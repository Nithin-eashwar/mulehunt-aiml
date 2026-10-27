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


def _scheduled_business(accounts, cfg, rng, festival, activity, key, payer_mask, ticket):
    """Bills (personal) and supplier payments (merchants): P2M to large merchants, never on Sundays."""
    weekday = _days(cfg).dayofweek.to_numpy()
    day_weight = np.where(weekday == SUNDAY, 0.0, 7 / 6) * festival
    rates = np.where(payer_mask, _rate(accounts, cfg, key, per_month=True), 0.0)
    rows, days = _daily_events(rates, day_weight, activity, rng)
    large = accounts.account_id[accounts.account_type == "large_merchant"].to_numpy()
    return _rows(accounts.account_id.to_numpy()[rows], rng.choice(large, size=len(rows)),
                 _timestamps(days, cfg, rng), _lognormal(cfg["ticket_size"][ticket], len(rows), rng), "P2M")


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
    """Monthly rent on days 1-10 to a landlord (an older salaried account)."""
    share = accounts.account_type.map(lambda t: cfg["behaviours"][t].get("rent_share", 0.0)).to_numpy()
    payers = accounts[rng.random(len(accounts)) < share]
    landlords = accounts.account_id[(accounts.account_type == "salaried")
                                    & (accounts.created_at < window_start(cfg))].to_numpy()
    landlord = rng.choice(landlords, size=len(payers))
    rent = _lognormal(cfg["ticket_size"]["rent"], len(payers), rng)
    days = _days(cfg)
    out = []
    for month_start in days[days.day == 1]:
        day_idx = (month_start - days[0]).days + rng.integers(0, 10, size=len(payers))
        out.append(_rows(payers.account_id.to_numpy(), landlord, _timestamps(day_idx, cfg, rng), rent, "P2P"))
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
        _scheduled_business(accounts, cfg, rng, festival, activity, "bills_per_month", personal, "bills"),
        _scheduled_business(accounts, cfg, rng, festival, activity, "supplier_per_month", ~personal, "supplier"),
    ], ignore_index=True)
    p2p = pd.concat([_salaries(accounts, cfg, rng), _gig_payouts(accounts, cfg, rng), _rent(accounts, cfg, rng)],
                    ignore_index=True)
    share = cfg["channel_mix"]
    n_needed = round(len(p2m) * share["P2P"] / share["P2M"]) - len(p2p)
    tx = pd.concat([p2m, p2p, _family_transfers(accounts, cfg, rng, festival, activity, n_needed)],
                   ignore_index=True)

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
