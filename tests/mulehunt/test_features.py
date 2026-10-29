import shutil

import numpy as np
import pandas as pd
import pytest

from mulehunt.features import (
    FEATURE_COLUMNS,
    FEATURES_CHECKSUM,
    FEATURE_GROUPS,
    TXN_ONLY,
    account_scenarios,
    build_feature_table,
    compute_account_features,
    content_checksum,
    fifo_match,
    split_windows,
)
from mulehunt.features import main as features_main
from simulator.config import load_config
from simulator.schema import read_accounts, read_transactions

IST = "Asia/Kolkata"
T = pd.Timestamp("2025-09-20 10:00", tz=IST)
START = pd.Timestamp("2025-09-12", tz=IST)
END = pd.Timestamp("2025-10-12", tz=IST)
TOY = "data/generated/toy"


def accounts(*ids, state="Karnataka", created="2024-01-01"):
    return pd.DataFrame({"account_id": list(ids), "account_type": "student", "bank": "SBI",
                         "state": state, "created_at": pd.Timestamp(created, tz=IST),
                         "role": "normal", "is_fraud": 0})


def txns(*rows):
    """rows: (sender, receiver, amount, time[, device])."""
    return pd.DataFrame({
        "txn_id": [f"T{i}" for i in range(len(rows))],
        "timestamp": [r[3] for r in rows],
        "sender_id": [r[0] for r in rows], "receiver_id": [r[1] for r in rows],
        "amount": [float(r[2]) for r in rows], "channel": "P2P",
        "device_id": [r[4] if len(r) > 4 else f"D{r[0]}" for r in rows],
        "scenario": "normal", "is_fraud": 0,
    })


def features(tx, acc, ids, start=START, end=END):
    return compute_account_features(tx, acc, list(ids), start, end)


def h(hours):
    return T + pd.Timedelta(hours=hours)


def test_feature_groups():
    assert FEATURE_COLUMNS == [c for g in FEATURE_GROUPS.values() for c in g]
    assert TXN_ONLY == FEATURE_GROUPS["flow"] + FEATURE_GROUPS["behavioural"] + FEATURE_GROUPS["device_location"]
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS) == 20


def test_split_windows():
    w = split_windows(load_config())
    start = pd.Timestamp("2025-08-13", tz=IST)
    assert w["train"] == (start + pd.Timedelta(days=30), start + pd.Timedelta(days=60))
    assert w["val"] == (start + pd.Timedelta(days=45), start + pd.Timedelta(days=75))
    assert w["test"] == (start + pd.Timedelta(days=60), start + pd.Timedelta(days=90))


def test_pass_through_full():
    tx = txns(("X", "A", 10000, h(0)), ("A", "Y", 9500, h(2)))
    f = features(tx, accounts("A", "X", "Y"), ["A"], end=h(10)).loc["A"]
    assert f.pass_through_ratio == pytest.approx(0.95)
    assert f.avg_dwell_hours == pytest.approx((9500 * 2 + 500 * 10) / 10000)
    assert f.total_inflow == 10000 and f.total_outflow == 9500


def test_pass_through_not_double_counted():
    rows = [(f"S{i}", "A", 100, h(i / 60)) for i in range(30)] + [("A", "Y", 100, h(1.5))]
    f = features(txns(*rows), accounts("A", "Y", *[f"S{i}" for i in range(30)]), ["A"]).loc["A"]
    assert f.pass_through_ratio == pytest.approx(1 / 30)


def test_fifo_order():
    inflows = pd.DataFrame({"timestamp": [h(0), h(5)], "amount": [100.0, 100.0]})
    outflows = pd.DataFrame({"timestamp": [h(6)], "amount": [150.0]})
    m = fifo_match(inflows, outflows)
    assert list(m.amount) == [100.0, 50.0]
    assert list(m.inflow_time) == [h(0), h(5)]
    assert list(m.outflow_time) == [h(6), h(6)]


def test_outflow_without_inflow_ignored():
    base = txns(("X", "A", 1000, h(5)), ("A", "Y", 500, h(6)))
    extra = txns(("A", "Z", 700, h(1)), ("X", "A", 1000, h(5)), ("A", "Y", 500, h(6)))
    acc = accounts("A", "X", "Y", "Z")
    assert features(extra, acc, ["A"]).loc["A"].pass_through_ratio == \
        features(base, acc, ["A"]).loc["A"].pass_through_ratio == pytest.approx(0.5)


def test_pass_through_late_outflow_ignored():
    tx = txns(("X", "A", 1000, h(0)), ("A", "Y", 1000, h(30)))
    f = features(tx, accounts("A", "X", "Y"), ["A"]).loc["A"]
    assert f.pass_through_ratio == 0
    assert f.avg_dwell_hours == pytest.approx(30)


def test_fan_in_max_24h():
    rows = [(f"S{i}", "A", 100, h(2 * i)) for i in range(5)] + [("S9", "A", 100, h(72))]
    f = features(txns(*rows), accounts("A", *[f"S{i}" for i in (0, 1, 2, 3, 4, 9)]), ["A"]).loc["A"]
    assert f.fan_in_max_24h == 5
    assert f.in_degree == 6


def test_fan_out_max_24h():
    rows = [("A", f"R{i}", 100, h(i)) for i in range(4)]
    f = features(txns(*rows), accounts("A", *[f"R{i}" for i in range(4)]), ["A"]).loc["A"]
    assert f.fan_out_max_24h == 4 and f.out_degree == 4


def test_below_threshold():
    rows = [("A", "Y", a, h(i)) for i, a in enumerate([9100, 23000, 5000, 49990, 10000])]
    assert features(txns(*rows), accounts("A", "Y"), ["A"]).loc["A"].below_threshold_share == pytest.approx(0.6)


def test_new_counterparty_history_fixed():
    acc = accounts("A", "OLD", "RECENT", "NEW")
    for start in [START, START + pd.Timedelta(days=15)]:
        tx = txns(("A", "OLD", 100, start - pd.Timedelta(days=31)),
                  ("A", "RECENT", 100, start - pd.Timedelta(days=29)),
                  ("A", "OLD", 100, start + pd.Timedelta(days=1)),
                  ("A", "RECENT", 100, start + pd.Timedelta(days=1)),
                  ("A", "NEW", 100, start + pd.Timedelta(days=2)))
        f = features(tx, acc, ["A"], start=start, end=start + pd.Timedelta(days=30)).loc["A"]
        assert f.new_counterparty_share == pytest.approx(2 / 3)  # OLD and NEW are new; RECENT is not


def test_round_amount():
    rows = [("A", "Y", a, h(i)) for i, a in enumerate([1000, 1500, 1234, 500])]
    assert features(txns(*rows), accounts("A", "Y"), ["A"]).loc["A"].round_amount_share == pytest.approx(0.5)


def test_night_share():
    day = T.normalize()
    rows = [("A", "Y", 100, day + pd.Timedelta(hours=hr)) for hr in (23, 26, 12, 18)]  # 26 = 2 am next day
    assert features(txns(*rows), accounts("A", "Y"), ["A"]).loc["A"].night_txn_share == pytest.approx(0.5)


def test_burstiness():
    even = [("A", "Y", 100, h(i)) for i in range(5)]
    f = features(txns(*even), accounts("A", "Y"), ["A"]).loc["A"]
    assert f.burstiness == pytest.approx(-1.0)  # perfectly regular gaps
    two = features(txns(*even[:2]), accounts("A", "Y"), ["A"]).loc["A"]
    assert two.burstiness == 0


def test_large_receipt_age():
    acc = accounts("A", "X", created="2025-09-01")
    tx = txns(("X", "A", 500, h(0)), ("X", "A", 20000, h(48)))
    f = features(tx, acc, ["A"]).loc["A"]
    assert f.has_large_receipt == 1
    assert f.account_age_at_first_large_receipt_days == pytest.approx((h(48) - pd.Timestamp("2025-09-01", tz=IST)).total_seconds() / 86400)


def test_device_features():
    tx = txns(("A", "Y", 100, h(0), "D1"), ("B", "Y", 100, h(1), "D1"), ("A", "Y", 100, h(2), "D2"))
    f = features(tx, accounts("A", "B", "Y"), ["A", "B", "Y"])
    assert f.loc["A"].device_shared_accounts == 1 and f.loc["B"].device_shared_accounts == 1
    assert f.loc["A"].device_count == 2 and f.loc["Y"].device_count == 0


def test_state_mismatch():
    acc = pd.concat([accounts("A", "Y"), accounts("Z", state="Delhi")], ignore_index=True)
    tx = txns(("A", "Y", 100, h(0)), ("A", "Z", 100, h(1)))
    assert features(tx, acc, ["A"]).loc["A"].counterparty_state_mismatch_share == pytest.approx(0.5)


def test_neighbor_suspicious_share():
    # M passes 95% on within 2 h; N keeps its money. A pays both.
    tx = txns(("A", "M", 1000, h(0)), ("M", "Q", 950, h(2)), ("A", "N", 1000, h(0)))
    f = features(tx, accounts("A", "M", "N", "Q"), ["A"]).loc["A"]
    assert f.neighbor_suspicious_share == pytest.approx(0.5)


def test_idle_account_is_finite():
    acc = accounts("A", "IDLE", created="2025-09-01")
    f = features(txns(("A", "A2", 100, h(0))), pd.concat([acc, accounts("A2")]), ["IDLE"]).loc["IDLE"]
    assert np.isfinite(f.to_numpy(dtype=float)).all()
    assert f.pass_through_ratio == 0 and f.total_inflow == 0 and f.txn_count == 0
    assert f.avg_dwell_hours == pytest.approx((END - START).total_seconds() / 3600)
    assert f.burstiness == 0 and f.has_large_receipt == 0
    assert f.account_age_at_first_large_receipt_days == pytest.approx(
        (END - pd.Timestamp("2025-09-01", tz=IST)).total_seconds() / 86400)


def test_window_respected():
    tx = txns(("X", "A", 1000, h(0)), ("A", "Y", 900, h(2)))
    late = pd.concat([tx, txns(("A", "Z", 5000, END + pd.Timedelta(seconds=1)))], ignore_index=True)
    at_end = pd.concat([tx, txns(("A", "Z", 5000, END))], ignore_index=True)
    acc = accounts("A", "X", "Y", "Z")
    pd.testing.assert_frame_equal(features(tx, acc, ["A"]), features(late, acc, ["A"]))
    pd.testing.assert_frame_equal(features(tx, acc, ["A"]), features(at_end, acc, ["A"]))


@pytest.fixture(scope="module")
def toy():
    acc = read_accounts(f"{TOY}/accounts.csv")
    tx = read_transactions(f"{TOY}/transactions.csv")
    splits = pd.Series(np.random.default_rng(0).choice(["train", "val", "test"], size=len(acc), p=[0.5, 0.25, 0.25]),
                       index=acc.account_id)
    return acc, tx, splits


def test_no_label_columns_used(toy):
    acc, tx, _ = toy
    start, end = split_windows(load_config())["train"]
    rng = np.random.default_rng(1)
    acc2 = acc.assign(role=rng.permutation(acc.role.to_numpy()), is_fraud=rng.permutation(acc.is_fraud.to_numpy()))
    tx2 = tx.assign(scenario=rng.permutation(tx.scenario.to_numpy()), is_fraud=rng.permutation(tx.is_fraud.to_numpy()))
    ids = list(acc.account_id[:200])
    pd.testing.assert_frame_equal(compute_account_features(tx, acc, ids, start, end),
                                  compute_account_features(tx2, acc2, ids, start, end))


def test_table_layout(toy):
    acc, tx, splits = toy
    table = build_feature_table(acc, tx, splits, load_config())
    assert list(table.columns) == ["account_id", *FEATURE_COLUMNS, "label", "split"]
    assert table.account_id.is_unique and len(table) == len(acc)
    assert set(table.split) <= {"train", "val", "test"}
    assert (table.label == table.account_id.map(acc.set_index("account_id").is_fraud)).all()
    assert np.isfinite(table[FEATURE_COLUMNS].to_numpy(dtype=float)).all()


def test_splits_disjoint_and_time_safe(toy):
    acc, tx, splits = toy
    cfg = load_config()
    _, train_end = split_windows(cfg)["train"]
    train_ids = splits.index[splits == "train"]
    later = txns((train_ids[0], train_ids[1], 77777, train_end), (train_ids[2], train_ids[0], 5555, train_end + pd.Timedelta(hours=1)))
    before = build_feature_table(acc, tx, splits, cfg)
    after = build_feature_table(acc, pd.concat([tx, later], ignore_index=True), splits, cfg)
    cols = ["account_id", *FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(before[before.split == "train"][cols].reset_index(drop=True),
                                  after[after.split == "train"][cols].reset_index(drop=True))


def test_account_scenarios(toy):
    _, tx, _ = toy
    s = account_scenarios(tx)
    assert set(s.unique()) == {"fake_kyc", "job_task", "digital_arrest"}
    fraud = tx[tx.is_fraud == 1]
    assert set(s.index) == set(fraud.sender_id) | set(fraud.receiver_id)


def small_table():
    return pd.DataFrame({"account_id": ["A", "B"], "total_inflow": [1.5, np.nan],
                         "pass_through_ratio": [0.0, 0.25], "label": [0, 1], "split": ["train", "test"]})


def test_content_checksum_ignores_file_bytes(tmp_path):
    # Two writes of the same table can differ byte for byte (the writer records its version);
    # the content checksum must not.
    df = small_table()
    df.to_parquet(tmp_path / "a.parquet", index=False, compression="snappy")
    df.to_parquet(tmp_path / "b.parquet", index=False, compression="gzip")
    assert (tmp_path / "a.parquet").read_bytes() != (tmp_path / "b.parquet").read_bytes()
    assert content_checksum(pd.read_parquet(tmp_path / "a.parquet")) == \
        content_checksum(pd.read_parquet(tmp_path / "b.parquet"))


def test_content_checksum_sees_values_and_names():
    base = content_checksum(small_table())
    assert base == content_checksum(small_table().assign(pass_through_ratio=[-0.0, 0.25]))  # -0.0 == 0.0
    assert base != content_checksum(small_table().assign(pass_through_ratio=[0.0, 0.25 + 1e-15]))
    assert base != content_checksum(small_table().rename(columns={"label": "y"}))
    assert base != content_checksum(small_table().iloc[::-1])


def run_features_main(tmp_path, out):
    data = tmp_path / "data/generated"
    shutil.copytree(TOY, data)
    pd.DataFrame({"account_id": read_accounts(f"{TOY}/accounts.csv").account_id, "split": "train"}).to_csv(
        data.parent / "split.csv", index=False)
    (data / "_meta").mkdir()
    shutil.copy(data.parent / "split.csv", data / "_meta/account_split.csv")
    (data / "CHECKSUMS.txt").write_text("abc  accounts.csv\n", encoding="utf-8")
    features_main(["--data", str(data), "--out", str(out)])
    return data


def test_main_records_content_checksum_for_the_default_output(tmp_path, capsys):
    out = tmp_path / "data/processed/features.parquet"
    data = run_features_main(tmp_path, out)
    assert (data / "CHECKSUMS.txt").read_text(encoding="utf-8") == "abc  accounts.csv\n"  # left alone
    expected = content_checksum(pd.read_parquet(out))
    assert (data / FEATURES_CHECKSUM).read_text(encoding="utf-8").split()[0] == expected
    features_main(["--data", str(data), "--out", str(out), "--verify"])
    assert "OK" in capsys.readouterr().out


def test_main_elsewhere_writes_no_checksum(tmp_path):
    data = run_features_main(tmp_path, tmp_path / "scratch/features.parquet")
    assert not (data / FEATURES_CHECKSUM).exists()


def test_verify_fails_on_changed_table(tmp_path):
    out = tmp_path / "data/processed/features.parquet"
    data = run_features_main(tmp_path, out)
    pd.read_parquet(out).assign(label=0).to_parquet(out, index=False)
    with pytest.raises(SystemExit) as exc:
        features_main(["--data", str(data), "--out", str(out), "--verify"])
    assert exc.value.code == 1
