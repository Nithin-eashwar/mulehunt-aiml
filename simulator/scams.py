"""The five scam scenarios and the mule rings that launder the money.

Flow: schedule_rings decides when and in which split each ring runs (plan D8),
recruit_members picks victims and sleeper mules from the population and opens
fresh mule, cash-out and fraudster accounts, and one function per scenario
writes the victim payments and the laundering hops (simulator.patterns).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simulator.accounts import allocate, device_name, window_start
from simulator.behaviors import OUT_COLUMNS, _lognormal
from simulator.patterns import chain, cycle, fan_in, fan_out, hop_delay, scatter_gather, split_under_threshold

SCENARIOS = ("fake_kyc", "digital_arrest", "job_task", "loan_app", "fake_investment")
LARGE_PAYMENT_SCENARIOS = ("fake_kyc", "digital_arrest", "fake_investment")  # can be split under thresholds
SINGLE_RECEIVER_SCENARIOS = ("fake_kyc", "digital_arrest")  # the victim pays one tier-1 mule
DURATION_DAYS = {"fake_kyc": (1, 2), "job_task": (1, 3), "loan_app": (3, 8), "fake_investment": (3, 8)}
MULE_TIERS = ("mule_t1", "mule_t2", "mule_t3")
MAIN_COLLECTOR_SHARE = 0.6  # job_task: share of victims paying the main tier-1 mule
VICTIM_MIN_AGE_DAYS = 30
VICTIM_HOURS = (9, 21)
FEE_RANGE = (500, 2000)
RETURN_WINDOW = 0.5  # fake_investment: "returns" happen in the first half of the ring
CAMOUFLAGE_PAYMENTS = (3, 8)
MAX_ATTEMPTS = 50


@dataclass
class Ring:
    ring_id: int
    scenario: str
    split: str
    start: pd.Timestamp
    fraudster: int = -1
    mules: dict = field(default_factory=dict)
    cashout: int = -1
    victims: list = field(default_factory=list)
    evasive: bool = False
    duration_days: int = 1
    cycle: bool = False
    device_of: dict = field(default_factory=dict)
    sleepers: list = field(default_factory=list)  # existing accounts recruited as mules; they use their own phone
    patient: list = field(default_factory=list)   # mules that wait 1-2 days before forwarding

    def mule_list(self, tier: str) -> list:
        return self.mules.get(tier, [])


# ---------------------------------------------------------------- scheduling

def ring_day_bounds(cfg: dict, split: str) -> tuple[int, int]:
    """First and last day (1-based) a ring in `split` may use: inside the split and
    inside its feature window, which ends on the split's last day (plan D8)."""
    a, b = cfg["splits"][split]
    return max(a, b - cfg["splits"]["lookback_days"] + 1), b


def _duration(scenario, cfg, rng):
    lo, hi = cfg["scenarios"]["digital_arrest"]["days"] if scenario == "digital_arrest" else DURATION_DAYS[scenario]
    return int(rng.integers(lo, min(hi, cfg["fraud"]["max_ring_days"]) + 1))


def schedule_rings(cfg: dict, rng: np.random.Generator) -> list[Ring]:
    start0 = window_start(cfg)
    rings = []
    for scenario in SCENARIOS:
        for split in allocate(cfg["fraud"]["rings_per_scenario"], cfg["splits"]["shares"], rng):
            duration = _duration(scenario, cfg, rng)
            lo, hi = ring_day_bounds(cfg, split)
            day = int(rng.integers(lo, hi - duration - cfg["splits"]["ring_tail_days"] + 1))
            start = start0 + pd.Timedelta(days=day - 1, hours=float(rng.uniform(8, 12)))
            rings.append(Ring(len(rings), scenario, str(split), start.floor("s"), duration_days=duration))

    eligible = [r for r in rings if r.scenario in LARGE_PAYMENT_SCENARIOS]
    k = round(cfg["fraud"]["threshold_split_share"] * len(rings))
    for i in rng.choice(len(eligible), size=min(k, len(eligible)), replace=False):
        eligible[i].evasive = True
    invest = [r for r in rings if r.scenario == "fake_investment"]
    n_cycle = round(cfg["scenarios"]["fake_investment"]["cycle_ring_share"] * len(invest))
    for i in rng.choice(len(invest), size=n_cycle, replace=False):
        invest[i].cycle = True
    return rings


# ---------------------------------------------------------------- recruitment

def _ring_shape(ring, cfg, rng):
    size = cfg["fraud"]["ring_size"]
    n = {tier: int(rng.integers(lo, hi + 1)) for tier, (lo, hi) in size.items()}
    if ring.scenario in SINGLE_RECEIVER_SCENARIOS:
        n["mule_t2"] += n["mule_t1"] - 1
        n["mule_t1"] = 1
    if ring.scenario == "digital_arrest":
        n["mule_t3"] = min(n["mule_t3"], 1)
    if ring.cycle:
        n["mule_t2"] = max(n["mule_t2"], 2)
    n["mule_t3"] = min(n["mule_t3"], n["mule_t2"])
    return n


def _new_account(ids, cfg, rng, account_type, role, created, split, device):
    return {"account_id": ids, "account_type": account_type,
            "bank": rng.choice(list(cfg["bank_mix"]), p=list(cfg["bank_mix"].values())),
            "state": rng.choice(list(cfg["state_mix"]), p=list(cfg["state_mix"].values())),
            "created_at": created.floor("s"), "role": role, "is_fraud": 1, "primary_device": device,
            "second_device": "", "device_change_at": pd.NaT, "split": split, "household": -1}


class _Pool:
    """Draws distinct existing accounts that satisfy a per-ring condition."""

    def __init__(self, ids, rng):
        self.ids = list(rng.permutation(ids))

    def take(self, k, ok, used):
        out = []
        for acc in self.ids:
            if len(out) == k:
                break
            if acc not in used and ok(acc):
                out.append(acc)
                used.add(acc)
        if len(out) < k:
            raise RuntimeError("not enough eligible accounts; increase population.n_accounts")
        return out


def recruit_members(rings, accounts, cfg, rng, normal_txns=None) -> pd.DataFrame:
    fraud = cfg["fraud"]
    accounts = accounts.copy()
    created = accounts.set_index("account_id").created_at
    own_device = accounts.set_index("account_id").primary_device
    personal = accounts.account_id[accounts.role == "normal"].to_numpy()
    if normal_txns is not None:
        first_seen = normal_txns.groupby("sender_id").timestamp.min()
    else:  # without transactions, an old account stands in for a long history
        first_seen = created.where(created < window_start(cfg))
    victim_pool, sleeper_pool = _Pool(personal, rng), _Pool(personal, rng)

    shapes = [_ring_shape(r, cfg, rng) for r in rings]
    slots = [(i, tier, j) for i, s in enumerate(shapes) for tier in MULE_TIERS for j in range(s[tier])]
    n_sleepers = round(fraud["sleeper_share"] * len(slots))
    sleeper_slots = {slots[i] for i in rng.choice(len(slots), size=n_sleepers, replace=False)}

    used, new_rows, role_of, split_of = set(), [], {}, {}
    next_id = int(accounts.account_id.max()) + 1
    devices = pd.concat([accounts.primary_device, accounts.second_device])
    next_dev = int(devices[devices != ""].str[3:].astype(int).max()) + 1

    for i, ring in enumerate(rings):
        lo_dev, hi_dev = fraud["mule_devices_per_ring"]
        ring_devices = [device_name(next_dev + k) for k in range(int(rng.integers(lo_dev, hi_dev + 1)))]
        next_dev += len(ring_devices)
        fresh_age = lambda: pd.Timedelta(days=float(rng.uniform(*fraud["fresh_mule_age_days"])))

        ring.fraudster = next_id
        new_rows.append(_new_account(next_id, cfg, rng, "salaried", "fraudster",
                                     ring.start - pd.Timedelta(days=float(rng.uniform(30, 365))),
                                     ring.split, ring_devices[0]))
        ring.device_of[next_id] = ring_devices[0]
        next_id += 1

        history = pd.Timedelta(days=fraud["sleeper_min_history_days"])
        for tier in MULE_TIERS:
            ring.mules[tier] = []
            for j in range(shapes[i][tier]):
                device = ring_devices[int(rng.integers(len(ring_devices)))]
                if (i, tier, j) in sleeper_slots:
                    acc = sleeper_pool.take(1, lambda a: first_seen.get(a, pd.NaT) <= ring.start - history,
                                            used)[0]
                    role_of[acc], split_of[acc] = tier, ring.split
                    device = own_device[acc]
                    ring.sleepers.append(acc)
                else:
                    acc = next_id
                    next_id += 1
                    new_rows.append(_new_account(acc, cfg, rng, str(rng.choice(["student", "gig_worker"])), tier,
                                                 ring.start - fresh_age(), ring.split, device))
                ring.mules[tier].append(acc)
                ring.device_of[acc] = device
                if rng.random() < fraud["patient_mule_share"]:
                    ring.patient.append(acc)

        ring.cashout = next_id
        device = ring_devices[int(rng.integers(len(ring_devices)))]
        new_rows.append(_new_account(next_id, cfg, rng, "gig_worker", "cashout", ring.start - fresh_age(),
                                     ring.split, device))
        ring.device_of[next_id] = device
        next_id += 1

        lo, hi = cfg["scenarios"][ring.scenario]["victims"]
        old_enough = ring.start - pd.Timedelta(days=VICTIM_MIN_AGE_DAYS)
        ring.victims = victim_pool.take(int(rng.integers(lo, hi + 1)), lambda a: created[a] <= old_enough, used)
        for acc in ring.victims:
            role_of[acc], split_of[acc] = "victim", ring.split

    idx = accounts.account_id.isin(role_of)
    accounts.loc[idx, "role"] = accounts.account_id[idx].map(role_of)
    accounts.loc[idx, "split"] = accounts.account_id[idx].map(split_of)
    accounts.loc[idx, "is_fraud"] = (accounts.role[idx] != "victim").astype(int)
    new = pd.DataFrame(new_rows, columns=accounts.columns)
    new["created_at"] = pd.to_datetime(new.created_at).dt.tz_convert(accounts.created_at.dt.tz)
    new["device_change_at"] = pd.Series(pd.NaT, index=new.index, dtype=accounts.device_change_at.dtype)
    return pd.concat([accounts, new], ignore_index=True)


# ---------------------------------------------------------------- scenario helpers

def _uniform(rng, lo, hi):
    return round(float(rng.uniform(lo, hi)), 2)


def _victim_time(ring, day_offset, rng):
    day = ring.start.normalize() + pd.Timedelta(days=int(day_offset))
    t = day + pd.Timedelta(hours=float(rng.uniform(*VICTIM_HOURS)))
    return max(t, ring.start).floor("s")


def _pay(ring, victim, mule, amount, t, cfg, rng) -> list[dict]:
    """A victim payment; evasive rings split it into parts just under a threshold."""
    if not ring.evasive:
        return fan_in([victim], mule, [round(amount, 2)], [t])
    lo, hi = cfg["fraud"]["threshold_band"]
    parts = split_under_threshold(amount, cfg["fraud"]["thresholds"], rng)
    in_band = [p for p in parts if any(lo * th <= p <= hi * th for th in cfg["fraud"]["thresholds"])]
    times = [t + pd.Timedelta(minutes=int(m)) for m in np.cumsum(rng.integers(2, 15, size=len(in_band)))]
    return fan_in([victim] * len(in_band), mule, in_band, times)


def _to_cashout(ring, i, amount, t, cfg, rng):
    """Tier-2 mule number i passes money via a tier-3 mule (if any) to the cash-out."""
    t2, t3s = ring.mule_list("mule_t2")[i], ring.mule_list("mule_t3")
    path = [t2] + ([t3s[i % len(t3s)]] if t3s else []) + [ring.cashout]
    return chain(path, amount, t, rng, cfg, set(ring.patient))


def _launder(ring, receipts, cfg, rng) -> list[dict]:
    """Move each tier-1 receipt (mule, amount, time) down to the cash-out account."""
    rows, t2s, slow = [], ring.mule_list("mule_t2"), set(ring.patient)
    for n, (t1, amount, t) in enumerate(receipts):
        if ring.scenario == "digital_arrest":
            t3s = ring.mule_list("mule_t3")
            gather = t3s[0] if t3s else ring.cashout
            hops = scatter_gather(t1, t2s, gather, amount, t, rng, cfg, slow)
            rows += hops
            if t3s:
                for r in hops:
                    if r["receiver_id"] == gather:
                        rows += chain([gather, ring.cashout], r["amount"], r["timestamp"], rng, cfg, slow)
            continue
        spread = fan_out(t1, t2s, amount, t, rng, cfg, slow)
        rows += spread
        for r in spread:
            i = t2s.index(r["receiver_id"])
            if ring.cycle and n == 0 and i == 0:
                loop = cycle(t2s, r["amount"], r["timestamp"], rng, cfg, slow)
                rows += loop
                rows += _to_cashout(ring, 0, loop[-1]["amount"], loop[-1]["timestamp"], cfg, rng)
            else:
                rows += _to_cashout(ring, i, r["amount"], r["timestamp"], cfg, rng)
    return rows


def _daily_batches(payments) -> list[tuple]:
    """Tier-1 mules forward each day's takings after the day's last payment."""
    df = pd.DataFrame(payments)
    df["day"] = df.timestamp.dt.date
    g = df.groupby(["receiver_id", "day"]).agg(amount=("amount", "sum"), t=("timestamp", "max")).reset_index()
    return [(r.receiver_id, round(r.amount, 2), r.t) for r in g.sort_values("t").itertuples()]


def _receipts(payments) -> list[tuple]:
    return [(p["receiver_id"], p["amount"], p["timestamp"]) for p in payments]


# ---------------------------------------------------------------- the five scenarios

def fake_kyc(ring: Ring, cfg: dict, rng) -> list[dict]:
    """One large payment by one victim, then a fast fan-out towards the cash-out."""
    p = cfg["scenarios"]["fake_kyc"]
    t1 = ring.mule_list("mule_t1")[0]
    pays = _pay(ring, ring.victims[0], t1, _uniform(rng, *p["amount"]), _victim_time(ring, 0, rng), cfg, rng)
    return _tag(ring, pays + _launder(ring, _receipts(pays), cfg, rng))


def digital_arrest(ring: Ring, cfg: dict, rng) -> list[dict]:
    """Repeated large payments by one victim over days, always to the same tier-1 mule."""
    p = cfg["scenarios"]["digital_arrest"]
    n = int(rng.integers(p["payments_per_victim"][0], p["payments_per_victim"][1] + 1))
    days = np.sort(np.concatenate([[0, ring.duration_days - 1], rng.integers(0, ring.duration_days, size=n - 2)]))
    t1, pays = ring.mule_list("mule_t1")[0], []
    for d in days:
        pays += _pay(ring, ring.victims[0], t1, _uniform(rng, *p["amount"]), _victim_time(ring, d, rng), cfg, rng)
    return _tag(ring, pays + _launder(ring, _receipts(pays), cfg, rng))


def job_task(ring: Ring, cfg: dict, rng) -> list[dict]:
    """Many victims pay small 'task deposits', gathered by one main tier-1 mule (fan-in)."""
    p = cfg["scenarios"]["job_task"]
    t1s, pays = ring.mule_list("mule_t1"), []
    for k, victim in enumerate(ring.victims):
        main = rng.random() < MAIN_COLLECTOR_SHARE or len(t1s) == 1
        mule = t1s[0] if main else t1s[1 + k % (len(t1s) - 1)]
        t = _victim_time(ring, rng.integers(0, ring.duration_days), rng)
        pays += fan_in([victim], mule, [_uniform(rng, *p["amount"])], [t])
    return _tag(ring, pays + _launder(ring, _daily_batches(pays), cfg, rng))


def loan_app(ring: Ring, cfg: dict, rng) -> list[dict]:
    """Many victims of loan-app extortion pay repeatedly over several days."""
    p = cfg["scenarios"]["loan_app"]
    t1s, pays = ring.mule_list("mule_t1"), []
    for k, victim in enumerate(ring.victims):
        n = int(rng.integers(p["payments_per_victim"][0], p["payments_per_victim"][1] + 1))
        for d in np.sort(rng.integers(0, ring.duration_days, size=n)):
            pays += fan_in([victim], t1s[k % len(t1s)], [_uniform(rng, *p["amount"])], [_victim_time(ring, d, rng)])
    return _tag(ring, pays + _launder(ring, _daily_batches(pays), cfg, rng))


def fake_investment(ring: Ring, cfg: dict, rng) -> list[dict]:
    """Small 'returns' from the fraudster build trust, then each victim makes one large payment."""
    p = cfg["scenarios"]["fake_investment"]
    t1s, returns, pays = ring.mule_list("mule_t1"), [], []
    trust_days = max(1, int(ring.duration_days * RETURN_WINDOW))
    for k, victim in enumerate(ring.victims):
        n = int(rng.integers(p["returns_per_victim"][0], p["returns_per_victim"][1] + 1))
        times = sorted(_victim_time(ring, d, rng) for d in rng.integers(0, trust_days, size=n))
        returns += [{"timestamp": t, "sender_id": ring.fraudster, "receiver_id": victim,
                     "amount": _uniform(rng, *p["return_amount"]), "channel": "P2P"} for t in times]
        pay_day = int(rng.integers(trust_days, ring.duration_days)) if ring.duration_days > trust_days else trust_days
        t = max(_victim_time(ring, pay_day, rng), times[-1] + pd.Timedelta(hours=1))
        pays += _pay(ring, victim, t1s[k % len(t1s)], _uniform(rng, *p["amount"]), t, cfg, rng)
    return _tag(ring, returns + pays + _launder(ring, _receipts(pays), cfg, rng))


SCENARIO_FUNCTIONS = {"fake_kyc": fake_kyc, "digital_arrest": digital_arrest, "job_task": job_task,
                      "loan_app": loan_app, "fake_investment": fake_investment}


def _tag(ring, rows):
    for r in rows:
        r["scenario"], r["is_fraud"] = ring.scenario, 1
    return rows


def _mule_fees(ring, rows, cfg, rng):
    """After the cash-out the fraudster pays every mule its fee."""
    t_end = max(r["timestamp"] for r in rows)
    mules = [m for tier in MULE_TIERS for m in ring.mule_list(tier)]
    return _tag(ring, [{"timestamp": (t_end + pd.Timedelta(hours=float(h))).floor("s"),
                        "sender_id": ring.fraudster, "receiver_id": m,
                        "amount": _uniform(rng, *FEE_RANGE), "channel": "P2P"}
                       for m, h in zip(mules, hop_delay(rng, cfg, size=len(mules)))])


def _camouflage(ring, accounts, cfg, rng, ring_end):
    """Fresh mules make a few ordinary shop payments so they don't look empty."""
    start = window_start(cfg)
    merchants = accounts[accounts.role == "merchant"]
    weight = merchants.account_type.map(cfg["behaviours"]["merchant_size_weight"]).to_numpy(dtype=float)
    created = accounts.set_index("account_id").created_at
    rows = []
    for m in [m for tier in MULE_TIERS for m in ring.mule_list(tier)]:
        lo = max(created[m], start)
        if lo >= ring_end:
            continue
        n = int(rng.integers(CAMOUFLAGE_PAYMENTS[0], CAMOUFLAGE_PAYMENTS[1] + 1))
        span = (ring_end - lo).total_seconds()
        times = [(lo + pd.Timedelta(seconds=float(s))).floor("s") + pd.Timedelta(seconds=1)
                 for s in rng.uniform(0, span - 1, size=n)]
        for t, amount, merchant in zip(times, _lognormal(cfg["ticket_size"]["p2m"], n, rng),
                                       rng.choice(merchants.account_id.to_numpy(), size=n, p=weight / weight.sum())):
            rows.append({"timestamp": t, "sender_id": m, "receiver_id": merchant, "amount": float(amount),
                         "channel": "P2M", "scenario": "normal", "is_fraud": 0})
    return rows


# ---------------------------------------------------------------- entry point

def generate_fraud(accounts: pd.DataFrame, cfg: dict, rng: np.random.Generator,
                   normal_txns: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[Ring]]:
    rings = schedule_rings(cfg, rng)
    accounts = recruit_members(rings, accounts, cfg, rng, normal_txns)
    start = window_start(cfg)
    fresh = set(accounts.account_id[accounts.created_at >= start])
    rows = []
    for ring in rings:
        lo, hi = ring_day_bounds(cfg, ring.split)
        earliest, latest = start + pd.Timedelta(days=lo - 1), start + pd.Timedelta(days=hi)
        for _ in range(MAX_ATTEMPTS):
            ring_rows = SCENARIO_FUNCTIONS[ring.scenario](ring, cfg, rng)
            ring_rows += _mule_fees(ring, ring_rows, cfg, rng)
            times = [r["timestamp"] for r in ring_rows]
            if min(times) >= earliest and max(times) < latest:
                break
        else:
            raise RuntimeError(f"ring {ring.ring_id} kept spilling out of its split; check ring_tail_days")
        for r in ring_rows:
            r["device_id"] = ring.device_of.get(r["sender_id"])
        rows += ring_rows
        camo_ring = Ring(ring.ring_id, ring.scenario, ring.split, ring.start,
                         mules={t: [m for m in ring.mule_list(t) if m in fresh] for t in MULE_TIERS})
        for r in _camouflage(camo_ring, accounts, cfg, rng, ring.start):
            r["device_id"] = ring.device_of[r["sender_id"]]
            rows.append(r)

    tx = pd.DataFrame(rows)
    a = accounts.set_index("account_id")
    own = tx.sender_id.map(a.primary_device)
    changed = tx.timestamp >= tx.sender_id.map(a.device_change_at)
    own = own.where(~changed.fillna(False).astype(bool), tx.sender_id.map(a.second_device))
    tx["device_id"] = tx.device_id.fillna(own)
    tx["amount"] = tx.amount.astype(float)
    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return accounts, tx[OUT_COLUMNS], rings
