"""Normal (non-fraud) UPI behaviour for the population from simulator.accounts.

Every behaviour is generated vectorised: per-account, per-day Poisson counts
(scaled by festival multipliers and a slow start for new accounts), then hours
from the configured hour-of-day curve, then lognormal amounts.
"""

import numpy as np
import pandas as pd

from simulator.accounts import window_start
from simulator.schema import TRANSACTION_COLUMNS

NEW_ACCOUNT_RAMP_DAYS = 14
NEW_ACCOUNT_RAMP_FACTOR = 0.5
FAMILY_RECEIVER_SHARE = 0.7
SUNDAY = 6
OUT_COLUMNS = [c for c in TRANSACTION_COLUMNS if c != "txn_id"]


# ---------------------------------------------------------------- helpers

def festival_multiplier(day: pd.Timestamp, cfg: dict) -> float:
    d = pd.Timestamp(day).date()
    for fest in cfg["festivals"]:
        if pd.Timestamp(fest["start"]).date() <= d <= pd.Timestamp(fest["end"]).date():
            return 1.0 + fest["volume_uplift"]
    return 1.0


def sample_hours(n: int, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(cfg["hour_of_day"], dtype=float)
    return rng.choice(24, size=n, p=w / w.sum())


def _days(cfg):
    return pd.date_range(window_start(cfg), periods=cfg["window"]["days"], freq="D")


def _timestamps(day_idx, cfg, rng):
    hours = sample_hours(len(day_idx), cfg, rng)
    seconds = day_idx * 86400 + hours * 3600 + rng.integers(0, 3600, size=len(day_idx))
    return window_start(cfg) + pd.to_timedelta(seconds, unit="s")


def _lognormal(params, size, rng):
    return np.round(np.exp(rng.normal(np.log(params["median"]), params["sigma"], size=size)), 2).clip(1.0)


def _activity(accounts, cfg):
    """n_accounts x n_days factor: 0 before the account exists, 0.5 in its first 14 days."""
    start = window_start(cfg)
    first = ((accounts.created_at - start).dt.total_seconds() / 86400).to_numpy()[:, None]
    day = np.arange(cfg["window"]["days"])[None, :]
    factor = np.where(day + 1 <= first, 0.0, 1.0)
    return np.where((day >= first) & (day < first + NEW_ACCOUNT_RAMP_DAYS), NEW_ACCOUNT_RAMP_FACTOR, factor)


def _daily_events(rates, day_weight, activity, rng):
    """Poisson counts per (account row, day) -> (row index, day index) arrays."""
    counts = rng.poisson(rates[:, None] * day_weight[None, :] * activity)
    flat = np.repeat(np.arange(counts.size), counts.ravel())
    return flat // counts.shape[1], flat % counts.shape[1]


def _rows(sender, receiver, ts, amount, channel):
    return pd.DataFrame({"timestamp": ts, "sender_id": sender, "receiver_id": receiver,
                         "amount": amount, "channel": channel})


def _rate(accounts, cfg, key, per_month=False):
    table = cfg["behaviours"]
    r = accounts.account_type.map(lambda t: table[t].get(key, 0.0)).to_numpy(dtype=float)
    return r / 30.0 if per_month else r


# ---------------------------------------------------------------- behaviours

def _merchant_payments(accounts, cfg, rng, festival, activity):
    merchants = accounts[accounts.role == "merchant"]
    weight = merchants.account_type.map(cfg["behaviours"]["merchant_size_weight"]).to_numpy(dtype=float)
    rows, days = _daily_events(_rate(accounts, cfg, "p2m_per_day"), festival, activity, rng)
    receiver = rng.choice(merchants.account_id.to_numpy(), size=len(rows), p=weight / weight.sum())
    return _rows(accounts.account_id.to_numpy()[rows], receiver, _timestamps(days, cfg, rng),
                 _lognormal(cfg["ticket_size"]["p2m"], len(rows), rng), "P2M")


def _scheduled_business(accounts, cfg, rng, festival, activity, key, payer_mask, ticket, channel,
                        skip_sunday):
    """Payments to large merchants: bills by people (P2M, any day) and supplier
    transfers between businesses (P2P, never on Sundays)."""
    weekday = _days(cfg).dayofweek.to_numpy()
    day_weight = (np.where(weekday == SUNDAY, 0.0, 7 / 6) if skip_sunday else np.ones(len(weekday))) * festival
    rates = np.where(payer_mask, _rate(accounts, cfg, key, per_month=True), 0.0)
    rows, days = _daily_events(rates, day_weight, activity, rng)
    large = accounts.account_id[accounts.account_type == "large_merchant"].to_numpy()
    return _rows(accounts.account_id.to_numpy()[rows], rng.choice(large, size=len(rows)),
                 _timestamps(days, cfg, rng), _lognormal(cfg["ticket_size"][ticket], len(rows), rng), channel)


def _salaries(accounts, cfg, rng):
    """Monthly salary from an employer (a large merchant), 90% on days 1-3, the rest days 4-7.
    Only some employers pay over UPI; the rest use NEFT/IMPS and never appear in UPI data."""
    salaried = accounts[accounts.account_type == "salaried"]
    salaried = salaried[rng.random(len(salaried)) < cfg["behaviours"]["salary_via_upi_share"]]
    large = accounts.account_id[accounts.account_type == "large_merchant"].to_numpy()
    employer = rng.choice(large, size=len(salaried))
    pay = _lognormal(cfg["ticket_size"]["salary"], len(salaried), rng)
    days = _days(cfg)
    out = []
    for month_start in days[days.day == 1]:
        late = rng.random(len(salaried)) < 0.1
        offset = np.where(late, rng.integers(3, 7, size=len(salaried)), rng.integers(0, 3, size=len(salaried)))
        day_idx = (month_start - days[0]).days + offset
        out.append(_rows(employer, salaried.account_id.to_numpy(), _timestamps(day_idx, cfg, rng),
                         np.round(pay * rng.uniform(0.98, 1.02, size=len(pay)), 2), "P2P"))
    return pd.concat(out, ignore_index=True)


def _gig_payouts(accounts, cfg, rng):
    """Weekly payouts from a platform (a large merchant) on a fixed weekday per worker."""
    gig = accounts[accounts.account_type == "gig_worker"]
    large = accounts.account_id[accounts.account_type == "large_merchant"].to_numpy()
    platform = rng.choice(large, size=len(gig))
    first_day = rng.integers(0, 7, size=len(gig))
    out = []
    for week in range(0, cfg["window"]["days"], 7):
        day_idx = week + first_day
        keep = day_idx < cfg["window"]["days"]
        out.append(_rows(platform[keep], gig.account_id.to_numpy()[keep], _timestamps(day_idx[keep], cfg, rng),
                         _lognormal(cfg["ticket_size"]["gig_payout"], keep.sum(), rng), "P2P"))
    return pd.concat(out, ignore_index=True)


def _rent(accounts, cfg, rng):
    """Monthly rent on days 1-15 to a landlord (an older salaried account)."""
    share = accounts.account_type.map(lambda t: cfg["behaviours"][t].get("rent_share", 0.0)).to_numpy()
    payers = accounts[rng.random(len(accounts)) < share]
    landlords = accounts.account_id[(accounts.account_type == "salaried")
                                    & (accounts.created_at < window_start(cfg))].to_numpy()
    landlord = rng.choice(landlords, size=len(payers))
    rent = _lognormal(cfg["ticket_size"]["rent"], len(payers), rng)
    days = _days(cfg)
    out = []
    for month_start in days[days.day == 1]:
        day_idx = (month_start - days[0]).days + rng.integers(0, 15, size=len(payers))
        keep = day_idx < len(days)
        out.append(_rows(payers.account_id.to_numpy()[keep], landlord[keep], _timestamps(day_idx[keep], cfg, rng),
                         rent[keep], "P2P"))
    return pd.concat(out, ignore_index=True)


def _family_transfers(accounts, cfg, rng, festival, activity, n_needed):
    """Person-to-person transfers, scaled so the overall P2P/P2M mix matches channel_mix."""
    base = _rate(accounts, cfg, "p2p_per_day")
    expected = (base[:, None] * festival[None, :] * activity).sum()
    rows, days = _daily_events(base * (max(n_needed, 0) / expected), festival, activity, rng)
    ids = accounts.account_id.to_numpy()
    sender = ids[rows]
    personal = accounts.account_id[accounts.role == "normal"].to_numpy()
    receiver = rng.choice(personal, size=len(rows))
    members = accounts[accounts.household >= 0].groupby("household").account_id.apply(np.array)
    hh = accounts.household.to_numpy()[rows]
    to_family = (rng.random(len(rows)) < FAMILY_RECEIVER_SHARE) & (hh >= 0)
    for i in np.flatnonzero(to_family):
        fam = members[hh[i]]
        if len(fam) > 1:
            receiver[i] = rng.choice(fam[fam != sender[i]])
    return _rows(sender, receiver, _timestamps(days, cfg, rng),
                 _lognormal(cfg["ticket_size"]["p2p_family"], len(rows), rng), "P2P")


def _collections(accounts, cfg, rng):
    """Hard negative: money collectors gather small P2P payments from several people in a
    day and pass most of it on within hours (group funds, family pooling)."""
    hard = cfg["hard_cases"]
    collectors = accounts.account_id[accounts.profile == "collector"].to_numpy()
    personal = accounts.account_id[(accounts.role == "normal") & (accounts.created_at < window_start(cfg))].to_numpy()
    rows = []
    n_days = cfg["window"]["days"]
    for c in collectors:
        for day in rng.choice(n_days, size=rng.poisson(hard["collections_per_month"] * n_days / 30), replace=True):
            n = int(rng.integers(hard["collection_size"][0], hard["collection_size"][1] + 1))
            payers = rng.choice(personal[personal != c], size=n, replace=False)
            times = np.sort(_timestamps(np.full(n, day), cfg, rng))
            amounts = _lognormal(hard["collection_amount"], n, rng)
            rows += [(t, p, c, a) for t, p, a in zip(times, payers, amounts)]
            total = amounts.sum() * rng.uniform(*hard["forward_share"])
            delay = np.exp(rng.normal(np.log(hard["forward_delay"]["median_h"]), hard["forward_delay"]["sigma"]))
            recipients = rng.choice(personal[personal != c], size=int(rng.integers(1, 3)), replace=False)
            t_out = (times[-1] + pd.Timedelta(hours=float(delay))).floor("s")
            for r, share in zip(recipients, rng.dirichlet(np.ones(len(recipients)))):
                rows.append((t_out, c, r, round(total * share, 2)))
    df = pd.DataFrame(rows, columns=["timestamp", "sender_id", "receiver_id", "amount"])
    df["channel"] = "P2P"
    end = window_start(cfg) + pd.Timedelta(days=n_days)
    return df[df.timestamp < end].reset_index(drop=True)


def _first_large_receipts(accounts, cfg, rng):
    """Hard negative: many new accounts get a large first receipt soon after opening
    (first salary, fees from parents, a family transfer), as fresh mules do."""
    hard = cfg["hard_cases"]
    start = window_start(cfg)
    new = accounts[(accounts.role == "normal") & (accounts.created_at >= start)]
    new = new[rng.random(len(new)) < hard["new_account_large_receipt_share"]]
    lo, hi = hard["new_account_large_receipt_days"]
    when = new.created_at + pd.to_timedelta(rng.uniform(lo, hi, size=len(new)) * 86400, unit="s")
    large = accounts.account_id[accounts.account_type == "large_merchant"].to_numpy()
    parents = accounts.account_id[(accounts.role == "normal") & (accounts.created_at < start)].to_numpy()
    sender = np.where(rng.random(len(new)) < 0.5, rng.choice(large, size=len(new)), rng.choice(parents, size=len(new)))
    df = _rows(sender, new.account_id.to_numpy(), when.dt.floor("s").reset_index(drop=True),
               np.round(rng.uniform(*hard["new_account_large_receipt_amount"], size=len(new)), 2), "P2P")
    return df[df.timestamp < start + pd.Timedelta(days=cfg["window"]["days"])].reset_index(drop=True)


def _night_shift(tx, accounts, cfg, rng):
    """Night workers' own payments follow the hour curve shifted by night_shift_hours."""
    night = set(accounts.account_id[accounts.profile == "night_worker"])
    mask = tx.sender_id.isin(night).to_numpy()
    if not mask.any():
        return tx
    tx = tx.copy()
    hours = (sample_hours(mask.sum(), cfg, rng) + cfg["hard_cases"]["night_shift_hours"]) % 24
    day = tx.timestamp[mask].dt.normalize()
    tx.loc[mask, "timestamp"] = (day + pd.to_timedelta(hours * 3600 + rng.integers(0, 3600, size=mask.sum()),
                                                       unit="s")).to_numpy()
    return tx


def outlier_events(accounts: pd.DataFrame, txns: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Genuine one-off spikes (wedding, hospital bill, phone): one large P2M payment
    at 10-40x the account's median amount, for about 1% of personal accounts."""
    personal = accounts[accounts.role == "normal"]
    medians = txns.groupby("sender_id").amount.median()
    eligible = personal.account_id[personal.account_id.isin(medians.index)].to_numpy()
    k = min(len(eligible), round(cfg["outliers"]["account_rate"] * len(personal)))
    chosen = rng.choice(eligible, size=k, replace=False)
    lo, hi = cfg["outliers"]["multiplier"]
    amount = np.round(medians.loc[chosen].to_numpy() * rng.uniform(lo, hi, size=k), 2)
    created = accounts.set_index("account_id").created_at.loc[chosen]
    start = window_start(cfg)
    first_day = np.maximum(0, np.ceil((created - start).dt.total_seconds() / 86400).to_numpy()).astype(int)
    day_idx = rng.integers(first_day, cfg["window"]["days"])
    large = accounts.account_id[accounts.account_type == "large_merchant"].to_numpy()
    return _rows(chosen, rng.choice(large, size=k), _timestamps(day_idx, cfg, rng), amount, "P2M")


# ---------------------------------------------------------------- entry point

def _attach_devices(tx, accounts):
    a = accounts.set_index("account_id")
    primary = tx.sender_id.map(a.primary_device)
    second = tx.sender_id.map(a.second_device)
    changed = tx.timestamp >= tx.sender_id.map(a.device_change_at)
    return primary.where(~changed.fillna(False).astype(bool), second)


def generate_normal_transactions(accounts: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    days = _days(cfg)
    festival = np.array([festival_multiplier(d, cfg) for d in days])
    activity = _activity(accounts, cfg)

    personal = (accounts.role == "normal").to_numpy()
    p2m = pd.concat([
        _merchant_payments(accounts, cfg, rng, festival, activity),
        _scheduled_business(accounts, cfg, rng, festival, activity, "bills_per_month", personal, "bills",
                            "P2M", skip_sunday=False),
    ], ignore_index=True)
    p2p = pd.concat([
        _scheduled_business(accounts, cfg, rng, festival, activity, "supplier_per_month", ~personal, "supplier",
                            "P2P", skip_sunday=True),
        _salaries(accounts, cfg, rng), _gig_payouts(accounts, cfg, rng), _rent(accounts, cfg, rng),
        _collections(accounts, cfg, rng), _first_large_receipts(accounts, cfg, rng),
    ], ignore_index=True)
    share = cfg["channel_mix"]
    n_needed = round(len(p2m) * share["P2P"] / share["P2M"]) - len(p2p)
    tx = pd.concat([p2m, p2p, _family_transfers(accounts, cfg, rng, festival, activity, n_needed)],
                   ignore_index=True)
    tx = _night_shift(tx, accounts, cfg, rng)

    created = accounts.set_index("account_id").created_at
    tx = tx[(tx.sender_id != tx.receiver_id)
            & (tx.timestamp >= tx.sender_id.map(created))
            & (tx.timestamp >= tx.receiver_id.map(created))]
    tx = pd.concat([tx, outlier_events(accounts, tx, cfg, rng)], ignore_index=True)
    tx = tx[tx.timestamp >= tx.receiver_id.map(created)]

    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)
    tx["device_id"] = _attach_devices(tx, accounts)
    tx["scenario"] = "normal"
    tx["is_fraud"] = 0
    return tx[OUT_COLUMNS]
