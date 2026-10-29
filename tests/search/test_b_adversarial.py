import math
from dataclasses import replace

import pandas as pd
import pytest

from mulehunt.adversarial import (GameParams, Stats, accounting, best_response, fixed_detector, full_depth,
                                  initial_state, measure_false_alarms, minimax, minimax_detector,
                                  minimax_fraudster, play_out)

FA_ZERO = GameParams(alert_cost=0)
FA_TOY = {(10000, False): 40.0, (10000, True): 300.0, (25000, False): 12.0, (25000, True): 272.0,
          (50000, False): 3.0, (50000, True): 263.0}


def test_one_round_by_hand():
    """X = 60k, R = 1, lambda = 0, fee 1,500, commission 5%. Last round: f = 1, so n is the choice.

    vs (10k, on):  n=1,2,4 parts >= 10k all caught -> -1,500 / -3,000 / -6,000.
                   n=8: parts 7,500 < 10k, velocity catches parts 4-8 (5 x 7,500 = 37,500),
                   uncaught 22,500 -> 22,500 x 0.95 - 8 x 1,500 = 9,375.   best = 9,375 (n=8)
    vs (25k, on):  n=4: parts 15k, 1 caught, 45,000 x 0.95 - 6,000 = 36,750 (best)
    vs (50k, *):   n=2: parts 30k, 57,000 - 3,000 = 54,000
    vs (10k, off): n=8: 57,000 - 12,000 = 45,000;  (25k, off): n=4: 51,000
    Detector minimises: (10k, on), value 9,375, fraudster splits into 8."""
    p = replace(FA_ZERO, X=60_000, R=1)
    v, d = minimax(initial_state(p), 2, p, alpha_beta=False)
    assert d == (10000, True) and math.isclose(v, 9375)
    val, plan = best_response((10000, True), p)
    assert plan == [(1.0, 8)] and math.isclose(val, 9375)
    assert math.isclose(best_response((50000, False), p)[0], 54000)


@pytest.mark.parametrize("lam", [0, 25, 100, 400])
def test_alpha_beta_matches_minimax(lam):
    p = replace(GameParams(fa=FA_TOY), alert_cost=lam, R=3)
    s1, s2, s3 = Stats(), Stats(), Stats()
    v1, m1 = minimax(initial_state(p), 6, p, False, False, s1)
    v2, m2 = minimax(initial_state(p), 6, p, True, False, s2)
    v3, m3 = minimax(initial_state(p), 6, p, True, True, s3)
    assert math.isclose(v1, v2, abs_tol=1e-6) and math.isclose(v1, v3, abs_tol=1e-6)
    assert m1 == m2
    assert s2.nodes < s1.nodes
    assert s3.nodes <= s2.nodes


def test_depth_limited_equals_full_when_deep_enough():
    p = replace(GameParams(fa=FA_TOY), R=2)
    full = minimax(initial_state(p), full_depth(p), p)
    deeper = minimax(initial_state(p), full_depth(p) + 4, p)
    assert full == deeper
    shallow = minimax(initial_state(p), 2, p)  # uses the evaluation function
    assert shallow[0] >= full[0] - 1e-9  # the evaluation is optimistic for the fraudster


@pytest.mark.parametrize("lam", [0, 100])
def test_accounting(lam):
    p = replace(GameParams(fa=FA_TOY), alert_cost=lam)
    for policy in (minimax_detector(p), fixed_detector((25000, False)), fixed_detector((10000, True))):
        acc = accounting(play_out(policy, minimax_fraudster(p), p), p)
        assert math.isclose(acc["total"], p.X, rel_tol=1e-9)


def test_fixed_threshold_is_split_under():
    p = GameParams(fa=FA_TOY)
    for T in (10000, 25000, 50000):
        val, plan = best_response((T, False), p)
        rem = p.X
        for f, n in plan:
            if f * rem / max(p.parts) < T:  # splitting under T is possible this round
                assert f * rem / n < T
            rem -= f * rem
        assert val > 0


def test_measure_false_alarms():
    ts = pd.to_datetime(["2025-09-01T10:00", "2025-09-01T11:00", "2025-09-02T10:00"] + ["2025-09-01T12:00"] * 4) \
        .tz_localize("Asia/Kolkata")
    tx = pd.DataFrame({"timestamp": ts, "sender_id": ["A", "B", "C", "D", "D", "D", "D"],
                       "amount": [60000, 12000, 30000, 1, 1, 1, 1], "scenario": "normal"})
    tx.loc[1, "scenario"] = "fake_kyc"
    fa = measure_false_alarms(tx)
    assert fa[(10000, False)] == 1.0  # 60k and 30k over 2 days; the scam row is ignored
    assert fa[(50000, False)] == 0.5
    assert fa[(10000, True)] == 1.5  # plus D's 4 payments on one day
