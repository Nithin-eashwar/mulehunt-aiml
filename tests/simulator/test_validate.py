import numpy as np
import pandas as pd
import pytest

from simulator.config import load_config
from simulator.schema import read_accounts, read_transactions
from simulator.validate import compare_ibm, detect_patterns, hour_tvd, run_checks, write_report

T0 = pd.Timestamp("2025-09-01 10:00", tz="Asia/Kolkata")
TOY = "data/generated/toy"


def edges(*hops):
    """hops: (sender, receiver, hours after T0)."""
    return pd.DataFrame({
        "timestamp": [T0 + pd.Timedelta(hours=h) for _, _, h in hops],
        "sender_id": [s for s, _, _ in hops],
        "receiver_id": [r for _, r, _ in hops],
        "amount": 1000.0, "channel": "P2P", "scenario": "fake_kyc", "is_fraud": 1,
    })


@pytest.fixture(scope="module")
def toy():
    return read_accounts(f"{TOY}/accounts.csv"), read_transactions(f"{TOY}/transactions.csv")


def test_detect_patterns_on_toy(toy):
    _, tx = toy
    p = detect_patterns(tx[tx.is_fraud == 1])
    assert p["fan_out"] >= 1
    assert p["fan_in"] >= 1
    assert p["chain"] >= 3


def test_detect_cycle():
    p = detect_patterns(edges(("A", "B", 0), ("B", "C", 2), ("C", "A", 4)))
    assert p["cycle"] == 1


def test_cycle_needs_time_order():
    p = detect_patterns(edges(("A", "B", 4), ("B", "C", 2), ("C", "A", 6)))
    assert p["cycle"] == 0


def test_chain_requires_time_order():
    ordered = detect_patterns(edges(("A", "B", 0), ("B", "C", 1), ("C", "D", 2)))
    assert ordered["chain"] == 1
    broken = detect_patterns(edges(("A", "B", 0), ("B", "C", -1), ("C", "D", 2)))
    assert broken["chain"] == 0


def test_chain_respects_window():
    p = detect_patterns(edges(("A", "B", 0), ("B", "C", 50), ("C", "D", 51)))
    assert p["chain"] == 0


def test_fan_out_and_fan_in_need_three_within_window():
    p = detect_patterns(edges(("A", "B", 0), ("A", "C", 1), ("A", "D", 2), ("X", "Z", 0), ("Y", "Z", 1)))
    assert p["fan_out"] == 1 and p["fan_in"] == 0
    late = detect_patterns(edges(("A", "B", 0), ("A", "C", 1), ("A", "D", 60)))
    assert late["fan_out"] == 0


def test_scatter_gather():
    p = detect_patterns(edges(("S", "M1", 0), ("S", "M2", 0.5), ("M1", "D", 2), ("M2", "D", 3)))
    assert p["scatter_gather"] == 1


def test_structural_patterns_optional():
    p = detect_patterns(edges(("A", "B", 0), ("B", "A", 1)), structural=False)
    assert set(p) == {"fan_out", "fan_in", "chain"}


def test_hour_tvd_zero_for_identical():
    w = np.arange(1, 25, dtype=float)
    assert hour_tvd(w, w * 3) == pytest.approx(0.0)
    assert hour_tvd([1] + [0] * 23, [0] * 23 + [1]) == pytest.approx(1.0)


def test_checks_table_columns(toy):
    accounts, tx = toy
    checks = run_checks(accounts, tx, load_config())
    assert list(checks.columns) == ["check", "target", "generated", "gap", "passed", "source"]
    names = set(checks.check)
    for name in ("mean_ticket_inr", "p2m_share", "hour_of_day_tvd", "txn_fraud_share", "account_fraud_share"):
        assert name in names
    assert {f"fraud_txns_{s}" for s in ("fake_kyc", "job_task", "digital_arrest")} <= names


def test_festival_check_uses_configured_baseline():
    # 100 transactions a day, but 150 on 1-5 Oct: Dussehra (2 Oct) must be compared with
    # its own baseline (22 Sep - 1 Oct) as NPCI measured it, not with the days before Diwali.
    cfg = load_config()
    days = pd.date_range("2025-09-15", "2025-10-31", tz="Asia/Kolkata")
    per_day = [150 if (d.month == 10 and d.day <= 5) else 100 for d in days]
    stamps = [d + pd.Timedelta(hours=12) for d, n in zip(days, per_day) for _ in range(n)]
    tx = pd.DataFrame({"timestamp": stamps, "sender_id": "A", "receiver_id": "B", "amount": 100.0,
                       "channel": "P2M", "scenario": "normal", "is_fraud": 0})
    accounts = pd.DataFrame({"account_id": ["A", "B"], "role": ["normal", "merchant"], "is_fraud": [0, 0]})
    checks = run_checks(accounts, tx, cfg).set_index("check")
    expected = 150 / ((9 * 100 + 150) / 10) - 1
    assert checks.loc["festival_Dussehra", "generated"] == pytest.approx(expected)


def test_ibm_missing_is_skipped(tmp_path):
    out = compare_ibm({"fan_out": 3, "fan_in": 2, "scatter_gather": 1, "cycle": 1, "chain": 5},
                      tmp_path / "missing.txt")
    assert (out.status == "SKIPPED").all()


def write_ibm(path, counts):
    lines = []
    for name, n in counts.items():
        for _ in range(n):
            lines += [f"BEGIN LAUNDERING ATTEMPT - {name}:  Max 4-degree", "2022/09/01 00:00,1,A,2,B,10,US Dollar,10,US Dollar,ACH,1",
                      f"END LAUNDERING ATTEMPT - {name}", ""]
    path.write_text("\n".join(lines))


def test_ibm_parses_and_maps(tmp_path):
    path = tmp_path / "p.txt"
    write_ibm(path, {"FAN-OUT": 3, "FAN-IN": 1, "CYCLE": 2, "SCATTER-GATHER": 1, "STACK": 4})
    out = compare_ibm({"fan_out": 1, "fan_in": 1, "scatter_gather": 1, "cycle": 1, "chain": 9}, path).set_index("ibm_pattern")
    assert out.loc["FAN-OUT", "ibm_count"] == 3
    assert out.loc["FAN-OUT", "status"] == "PASS"
    assert out.loc["STACK", "status"] == "not implemented"
    assert "chain (ours only)" in out.index


def test_ibm_mix_not_pass_fail(tmp_path):
    path = tmp_path / "p.txt"
    write_ibm(path, {"FAN-OUT": 100, "FAN-IN": 1, "CYCLE": 1, "SCATTER-GATHER": 1})
    out = compare_ibm({"fan_out": 1, "fan_in": 500, "scatter_gather": 1, "cycle": 300, "chain": 0}, path)
    mapped = out[out.our_pattern.isin(["fan_out", "fan_in", "scatter_gather", "cycle"])]
    assert (mapped.status == "PASS").all()
    zero = compare_ibm({"fan_out": 0, "fan_in": 1, "scatter_gather": 1, "cycle": 1, "chain": 0}, path)
    assert zero.set_index("our_pattern").loc["fan_out", "status"] == "FAIL"


def test_write_report(toy, tmp_path):
    accounts, tx = toy
    checks = run_checks(accounts, tx, load_config())
    report = tmp_path / "r.md"
    write_report(report, checks, detect_patterns(tx[tx.is_fraud == 1]), detect_patterns(tx, structural=False),
                 compare_ibm({}, None), figures={}, unsourced=["a.b"])
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert "| check |" in text and "mean_ticket_inr" in text
    assert "SKIPPED" in text and "a.b" in text


def test_expected_hour_curve_blends_night_workers():
    from simulator.validate import expected_hour_curve

    cfg = load_config()
    w = np.asarray(cfg["hour_of_day"], dtype=float)
    cfg["hard_cases"]["night_worker_share"] = 0.0
    assert np.allclose(expected_hour_curve(cfg), w / w.sum())
    cfg["hard_cases"]["night_worker_share"] = 0.3
    curve = expected_hour_curve(cfg)
    assert curve.sum() == pytest.approx(1.0)
    night = [23, 0, 1, 2, 3, 4]
    assert curve[night].sum() > (w / w.sum())[night].sum()
