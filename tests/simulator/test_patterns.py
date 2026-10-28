import numpy as np
import pandas as pd
import pytest

from simulator.config import load_config
from simulator.patterns import chain, cycle, fan_in, fan_out, hop_delay, scatter_gather, split_under_threshold

T0 = pd.Timestamp("2025-09-20 10:00", tz="Asia/Kolkata")
MAX_HOP = pd.Timedelta(hours=48)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def rng():
    return np.random.default_rng(11)


def test_hop_delay_distribution(cfg, rng):
    h = hop_delay(rng, cfg, size=20000)
    assert (h > 0).all() and (h <= 48).all()
    assert np.median(h) == pytest.approx(2.0, rel=0.1)
    assert (h <= 24).mean() >= 0.93


def test_fan_out_shape(cfg, rng):
    rows = fan_out(0, [1, 2, 3], 30000.0, T0, rng, cfg)
    assert len(rows) == 3
    assert {r["sender_id"] for r in rows} == {0}
    assert {r["receiver_id"] for r in rows} == {1, 2, 3}
    assert sum(r["amount"] for r in rows) <= 30000.0
    assert sum(r["amount"] for r in rows) >= 0.92 * 30000.0 - 0.05
    assert all(T0 < r["timestamp"] <= T0 + MAX_HOP for r in rows)


def test_fan_in_shape():
    t0s = [T0 + pd.Timedelta(minutes=m) for m in (0, 30, 90)]
    rows = fan_in([5, 6, 7], 1, [500.0, 700.0, 900.0], t0s, None, None)
    assert [r["receiver_id"] for r in rows] == [1, 1, 1]
    assert [r["amount"] for r in rows] == [500.0, 700.0, 900.0]
    assert [r["timestamp"] for r in rows] == t0s


def test_chain_time_respecting(cfg, rng):
    rows = chain([0, 1, 2, 3], 10000.0, T0, rng, cfg)
    assert [(r["sender_id"], r["receiver_id"]) for r in rows] == [(0, 1), (1, 2), (2, 3)]
    times = [T0] + [r["timestamp"] for r in rows]
    for prev, cur in zip(times, times[1:]):
        assert prev < cur <= prev + MAX_HOP
    amounts = [10000.0] + [r["amount"] for r in rows]
    for prev, cur in zip(amounts, amounts[1:]):
        assert 0.92 * prev - 0.01 <= cur <= 0.99 * prev + 0.01


def test_scatter_gather(cfg, rng):
    rows = scatter_gather(0, [1, 2, 3], 9, 50000.0, T0, rng, cfg)
    out = [r for r in rows if r["sender_id"] == 0]
    back = [r for r in rows if r["receiver_id"] == 9]
    assert {r["receiver_id"] for r in out} == {1, 2, 3}
    assert {r["sender_id"] for r in back} == {1, 2, 3}
    arrive = {r["receiver_id"]: r["timestamp"] for r in out}
    assert all(r["timestamp"] > arrive[r["sender_id"]] for r in back)


def test_cycle_returns_to_start(cfg, rng):
    rows = cycle([4, 5, 6], 20000.0, T0, rng, cfg)
    assert [(r["sender_id"], r["receiver_id"]) for r in rows] == [(4, 5), (5, 6), (6, 4)]


def test_split_under_threshold(rng):
    thresholds = [10000, 25000, 50000]
    parts = split_under_threshold(60000, thresholds, rng)
    assert sum(parts) == pytest.approx(60000, abs=0.01)
    assert all(p < 50000 for p in parts)
    assert sum(0.90 * 50000 <= p <= 0.999 * 50000 for p in parts) == len(parts) - 1


def test_split_small_amount_unchanged(rng):
    assert split_under_threshold(4000, [10000, 25000, 50000], rng) == [4000]


def test_patient_sender_waits(cfg, rng):
    lo, hi = cfg["fraud"]["patient_delay_h"]
    rows = chain([0, 1, 2, 3], 10000.0, T0, rng, cfg, slow={1})
    times = [T0] + [r["timestamp"] for r in rows]
    waits = [(b - a).total_seconds() / 3600 for a, b in zip(times, times[1:])]
    assert lo <= waits[1] <= hi + 0.01          # mule 1 is patient
    assert waits[0] <= 48 and waits[2] <= 48
    spread = fan_out(5, [6, 7], 1000.0, T0, rng, cfg, slow={5})
    assert all(lo <= (r["timestamp"] - T0).total_seconds() / 3600 <= hi + 0.01 for r in spread)
