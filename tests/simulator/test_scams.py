import copy

import numpy as np
import pandas as pd
import pytest

from simulator.accounts import make_population, window_start
from simulator.behaviors import generate_normal_transactions
from simulator.config import load_config
from simulator.scams import SCENARIOS, generate_fraud, schedule_rings

MULES = ("mule_t1", "mule_t2", "mule_t3")
MAX_HOP = pd.Timedelta(hours=48)


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config())
    c["population"]["n_accounts"] = 10000
    return c


@pytest.fixture(scope="module")
def world(cfg):
    rng = np.random.default_rng(3)
    accounts = make_population(cfg, rng)
    normal = generate_normal_transactions(accounts, cfg, rng)
    accounts2, fraud, rings = generate_fraud(accounts, cfg, rng, normal)
    return accounts2, normal, fraud, rings


def fraud_only(world):
    return world[2][world[2].is_fraud == 1]


def roles(world):
    return world[0].set_index("account_id").role


def day_bounds(cfg, split):
    """First and last allowed day (1-based) for a ring in this split: inside the
    split and inside its 30-day feature window (plan D8)."""
    a, b = cfg["splits"][split]
    return max(a, b - cfg["splits"]["lookback_days"] + 1), b


def test_every_scenario_in_every_split(cfg):
    rings = schedule_rings(cfg, np.random.default_rng(0))
    assert len(rings) == 5 * cfg["fraud"]["rings_per_scenario"]
    counts = pd.Series([(r.scenario, r.split) for r in rings]).value_counts()
    for scenario in SCENARIOS:
        for split in ("train", "val", "test"):
            assert counts.get((scenario, split), 0) >= 3, (scenario, split)
    shares = pd.Series([r.split for r in rings]).value_counts(normalize=True)
    for split, target in cfg["splits"]["shares"].items():
        assert shares[split] == pytest.approx(target, abs=0.05)
    assert all(1 <= r.duration_days <= cfg["fraud"]["max_ring_days"] for r in rings)


def test_ring_inside_feature_window(cfg, world):
    _, _, fraud, rings = world
    fraud = fraud[fraud.is_fraud == 1]
    start = window_start(cfg)
    for ring in rings:
        members = {ring.fraudster, ring.cashout, *ring.victims, *sum(ring.mules.values(), [])}
        rows = fraud[fraud.sender_id.isin(members) | fraud.receiver_id.isin(members)]
        assert len(rows) > 0, ring.ring_id
        day = (rows.timestamp - start).dt.days + 1
        lo, hi = day_bounds(cfg, ring.split)
        assert day.min() >= lo and day.max() <= hi, (ring.ring_id, ring.split, day.min(), day.max())
    train_starts = [(r.start - start).days + 1 for r in rings if r.split == "train"]
    assert min(train_starts) >= 31


def test_hop_timing_distribution(world):
    fraud = fraud_only(world)
    role = roles(world)
    delays = []
    for row in fraud[fraud.sender_id.map(role).isin(MULES)].itertuples():
        funding = fraud[(fraud.receiver_id == row.sender_id) & (fraud.timestamp < row.timestamp)
                        & (fraud.sender_id.map(role) != "fraudster")]
        assert len(funding) > 0, row
        delays.append(row.timestamp - funding.timestamp.max())
    delays = pd.Series(delays)
    assert (delays > pd.Timedelta(0)).all()
    assert (delays <= pd.Timedelta(hours=24)).mean() >= 0.90
    assert (delays <= MAX_HOP).all()


def test_sleepers_have_history(world):
    accounts, normal, fraud, _ = world
    role = roles(world)
    mules = accounts.account_id[accounts.role.isin(MULES)]
    first_fraud = fraud[fraud.is_fraud == 1].melt(id_vars="timestamp", value_vars=["sender_id", "receiver_id"]) \
        .groupby("value").timestamp.min()
    first_normal = normal.groupby("sender_id").timestamp.min()
    history = (first_fraud.reindex(mules) - first_normal.reindex(mules)).dt.days
    assert (history >= 21).mean() >= 0.25


def test_roles_consistent(world):
    accounts, _, _, _ = world
    fraud = fraud_only(world)
    role = roles(world)
    involved = set(fraud.sender_id) | set(fraud.receiver_id)
    for r in (*MULES, "fraudster", "cashout"):
        ids = set(accounts.account_id[accounts.role == r])
        assert ids and ids <= involved, r
    from_victim = fraud[fraud.sender_id.map(role) == "victim"]
    assert from_victim.receiver_id.map(role).isin(MULES).all()
    to_victim = fraud[fraud.receiver_id.map(role) == "victim"]
    assert (to_victim.sender_id.map(role) == "fraudster").all()
    assert (to_victim.scenario == "fake_investment").all()
    assert (accounts.is_fraud[accounts.role.isin([*MULES, "fraudster", "cashout"])] == 1).all()
    assert (accounts.is_fraud[accounts.role.isin(["normal", "merchant", "victim"])] == 0).all()


def test_members_share_ring_split(world):
    accounts, _, _, rings = world
    split = accounts.set_index("account_id").split
    for ring in rings:
        members = [ring.fraudster, ring.cashout, *ring.victims, *sum(ring.mules.values(), [])]
        assert set(split.loc[members]) == {ring.split}


def test_mules_share_devices_with_fraudster(world):
    fraud = fraud_only(world)
    rings = world[3]
    for ring in rings:
        senders = [ring.fraudster, *sum(ring.mules.values(), [])]
        devices = fraud[fraud.sender_id.isin(senders)].device_id.unique()
        assert 1 <= len(devices) <= 2, ring.ring_id


def test_scenario_signatures(world):
    fraud = fraud_only(world)
    role = roles(world)
    job = fraud[(fraud.scenario == "job_task") & (fraud.receiver_id.map(role) == "mule_t1")]
    assert job.groupby("receiver_id").sender_id.nunique().max() >= 15
    da = fraud[(fraud.scenario == "digital_arrest") & (fraud.sender_id.map(role) == "victim")]
    per_victim = da.groupby("sender_id").agg(n=("amount", "size"), days=("timestamp", lambda t: t.dt.date.nunique()))
    assert ((per_victim.n >= 3) & (per_victim.days >= 2)).any()
    fi = fraud[fraud.scenario == "fake_investment"]
    first_in = fi[fi.receiver_id.map(role) == "victim"].groupby("receiver_id").timestamp.min()
    first_out = fi[fi.sender_id.map(role) == "victim"].groupby("sender_id").timestamp.min()
    assert len(first_in) > 0
    assert (first_in < first_out.reindex(first_in.index)).all()


def test_evasive_share(cfg, world):
    fraud = fraud_only(world)
    rings = world[3]
    role = roles(world)
    evasive = [r for r in rings if r.evasive]
    assert len(evasive) / len(rings) == pytest.approx(cfg["fraud"]["threshold_split_share"], abs=0.10)
    lo, hi = cfg["fraud"]["threshold_band"]
    thresholds = np.array(cfg["fraud"]["thresholds"])
    for ring in evasive:
        pay = fraud[fraud.sender_id.isin(ring.victims) & (fraud.sender_id.map(role) == "victim")].amount
        assert len(pay) > 0
        for amount in pay:
            assert ((amount >= lo * thresholds) & (amount <= hi * thresholds)).any(), (ring.ring_id, amount)


def test_camouflage_spending_for_fresh_mules(cfg, world):
    accounts, _, fraud, _ = world
    start = window_start(cfg)
    fresh = accounts.account_id[accounts.role.isin(MULES) & (accounts.created_at >= start)]
    camo = fraud[(fraud.is_fraud == 0) & (fraud.channel == "P2M")]
    assert fresh.isin(camo.sender_id).mean() >= 0.9


def test_fraud_rows_after_account_creation(world):
    accounts, _, fraud, _ = world
    created = accounts.set_index("account_id").created_at
    assert (fraud.timestamp >= fraud.sender_id.map(created)).all()
    assert (fraud.timestamp >= fraud.receiver_id.map(created)).all()
