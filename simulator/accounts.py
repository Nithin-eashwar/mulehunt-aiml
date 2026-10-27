"""Normal and merchant population for the simulator.

Accounts get temporary integer ids here; simulator.generate assigns the final
ACC/TXN/DEV ids. Internal columns (not part of accounts.csv):
    primary_device   device the account normally pays from (shared within 4% of households)
    second_device    new device after a phone change, or "" if the phone never changes
    device_change_at time of the phone change, or NaT
    split            train / val / test (plan D8)
    household        family group for personal accounts, -1 for merchants
    shop_group       shop whose owner and staff share one phone, -1 if none (hard negative)
    profile          "collector", "night_worker" or "" (hard-negative behaviour, plan rev 4)
"""

import numpy as np
import pandas as pd

from simulator.schema import ACCOUNT_COLUMNS, IST_TZ

MERCHANT_TYPES = ("small_merchant", "large_merchant")
NEW_ACCOUNT_MAX_DAY = 80
OLD_ACCOUNT_AGE_DAYS = (30, 8 * 365)


def window_start(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["window"]["start"], tz=IST_TZ)


def allocate(n: int, shares: dict, rng: np.random.Generator) -> np.ndarray:
    """Exactly round(share * n) labels per key (largest remainder), shuffled."""
    keys = list(shares)
    raw = np.array([shares[k] for k in keys], dtype=float) * n
    counts = np.floor(raw).astype(int)
    counts[np.argsort(raw - counts)[::-1][: n - counts.sum()]] += 1
    return rng.permutation(np.repeat(keys, counts))


def device_name(i: int) -> str:
    return f"DEV{i:06d}"


def _households(personal_ids: np.ndarray, cfg: dict, rng: np.random.Generator) -> dict:
    lo, hi = cfg["population"]["family_size"]
    order = rng.permutation(personal_ids)
    household, pos, h = {}, 0, 0
    while pos < len(order):
        size = int(rng.integers(lo, hi + 1))
        for acc in order[pos:pos + size]:
            household[acc] = h
        pos, h = pos + size, h + 1
    return household


def make_population(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    pop = cfg["population"]
    n = pop["n_accounts"]
    start = window_start(cfg)
    types = allocate(n, pop["type_mix"], rng)
    is_merchant = np.isin(types, MERCHANT_TYPES)

    df = pd.DataFrame({
        "account_id": np.arange(n),
        "account_type": types,
        "bank": rng.choice(list(cfg["bank_mix"]), size=n, p=list(cfg["bank_mix"].values())),
        "state": rng.choice(list(cfg["state_mix"]), size=n, p=list(cfg["state_mix"].values())),
        "role": np.where(is_merchant, "merchant", "normal"),
        "is_fraud": 0,
    })

    # Most accounts are old; a share of personal accounts opens during the window (D24).
    age_days = rng.uniform(*OLD_ACCOUNT_AGE_DAYS, size=n)
    created = start - pd.to_timedelta(np.round(age_days * 86400), unit="s")
    personal = np.flatnonzero(~is_merchant)
    new = rng.choice(personal, size=round(pop["new_account_share"] * len(personal)), replace=False)
    new_offset = pd.to_timedelta(np.round(rng.uniform(0, NEW_ACCOUNT_MAX_DAY, size=len(new)) * 86400), unit="s")
    created = pd.Series(created)
    created.iloc[new] = start + new_offset
    df["created_at"] = created.dt.floor("s").to_numpy()
    df["created_at"] = df["created_at"].dt.tz_convert(IST_TZ)

    df["split"] = allocate(n, cfg["splits"]["shares"], rng)
    household = _households(df.account_id.to_numpy()[personal], cfg, rng)
    df["household"] = df.account_id.map(household).fillna(-1).astype(int)

    # One device per account, then 2-3 members of some households share a phone (D24).
    device = np.arange(1, n + 1)
    old_personal = df[(df.role == "normal") & (df.created_at < start)]
    sizes = df[df.role == "normal"].groupby("household").size()
    multi = sizes.index[sizes >= 2]
    eligible = [h for h, g in old_personal.groupby("household") if len(g) >= 2]
    k = min(len(eligible), round(pop["shared_device_household_share"] * len(multi)))
    for h in rng.choice(eligible, size=k, replace=False) if k else []:
        members = old_personal.account_id[old_personal.household == h].to_numpy()
        sharers = rng.choice(members, size=min(len(members), int(rng.integers(2, 4))), replace=False)
        device[sharers] = device[sharers[0]]
    # Hard negatives (plan rev 4): a shop and 2-5 staff pay from one phone, like a mule ring.
    hard = cfg["hard_cases"]
    df["shop_group"] = -1
    small = df.account_id[df.account_type == "small_merchant"].to_numpy()
    shared_phone = pd.Series(device).duplicated(keep=False).to_numpy()
    staff_pool = list(rng.permutation(df.account_id[(df.role == "normal") & (df.created_at < start)
                                                    & ~shared_phone].to_numpy()))
    lo, hi = hard["shop_phone_group_size"]
    for g, shop in enumerate(rng.choice(small, size=round(hard["shop_phone_group_share"] * len(small)),
                                        replace=False)):
        staff = [staff_pool.pop() for _ in range(int(rng.integers(lo, hi + 1)) - 1)]
        members = [shop, *staff]
        device[members] = device[shop]
        df.loc[members, "shop_group"] = g
    df["primary_device"] = [device_name(d) for d in device]

    # Profiles: money collectors forward what they collect within hours; night workers
    # transact on a shifted day (plan rev 4).
    df["profile"] = ""
    old_personal_ids = df.account_id[(df.role == "normal") & (df.created_at < start)].to_numpy()
    collectors = rng.choice(old_personal_ids, size=round(hard["money_collector_share"] * len(personal)), replace=False)
    df.loc[collectors, "profile"] = "collector"
    gig = df.account_id[(df.account_type == "gig_worker") & (df.profile == "")].to_numpy()
    n_night = round(hard["night_worker_share"] * (df.account_type == "gig_worker").sum())
    df.loc[rng.choice(gig, size=n_night, replace=False), "profile"] = "night_worker"

    # About 2% of personal accounts change phone once during the window.
    changers = rng.choice(personal, size=round(cfg["behaviours"]["phone_change_share"] * len(personal)),
                          replace=False)
    df["second_device"] = ""
    df.loc[changers, "second_device"] = [device_name(n + 1 + i) for i in range(len(changers))]
    change_at = pd.Series(pd.NaT, index=df.index, dtype=f"datetime64[ns, {IST_TZ}]")
    offsets = pd.to_timedelta(np.round(rng.uniform(10, cfg["window"]["days"] - 10, size=len(changers)) * 86400),
                              unit="s")
    change_at.iloc[changers] = (start + offsets).floor("s")
    df["device_change_at"] = change_at.where(change_at > df.created_at)
    df.loc[df.device_change_at.isna(), "second_device"] = ""

    return df[ACCOUNT_COLUMNS + ["primary_device", "second_device", "device_change_at", "split", "household",
                                "profile", "shop_group"]]
