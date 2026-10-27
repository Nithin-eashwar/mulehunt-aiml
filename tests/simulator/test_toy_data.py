import pandas as pd
import pytest

from simulator.make_toy_data import WINDOW_END, WINDOW_START, make_toy_data, write_toy_data
from simulator.schema import read_accounts, read_transactions, validate_accounts, validate_transactions

MAX_HOP = pd.Timedelta(hours=48)


@pytest.fixture(scope="module")
def toy():
    return make_toy_data()


def time_respecting_chain_exists(fraud_tx, roles):
    """True if some victim -> mule_t1 -> mule_t2 [-> mule_t3] -> cashout path has
    each hop later than the previous one and at most 48 h after it."""

    def step(frontier, allowed):
        nxt = []
        for node, t in frontier:
            hops = fraud_tx[(fraud_tx.sender_id == node)
                            & (fraud_tx.receiver_id.map(roles).isin(allowed))
                            & (fraud_tx.timestamp > t)
                            & (fraud_tx.timestamp <= t + MAX_HOP)]
            nxt += list(zip(hops.receiver_id, hops.timestamp))
        return nxt

    frontier = [(row.receiver_id, row.timestamp) for row in fraud_tx.itertuples()
                if roles[row.sender_id] == "victim" and roles[row.receiver_id] == "mule_t1"]
    frontier = step(frontier, {"mule_t2"})
    reached = step(frontier, {"cashout", "mule_t3"})
    via_t3 = step([(n, t) for n, t in reached if roles[n] == "mule_t3"], {"cashout"})
    return any(roles[n] == "cashout" for n, _ in reached) or bool(via_t3)


def test_toy_passes_schema(toy):
    accounts, tx = toy
    assert validate_accounts(accounts) == []
    assert validate_transactions(tx, accounts) == []


def test_toy_size(toy):
    accounts, _ = toy
    assert len(accounts) == 500


def test_toy_inside_window(toy):
    _, tx = toy
    assert tx.timestamp.min() >= WINDOW_START
    assert tx.timestamp.max() < WINDOW_END


def test_three_chains(toy):
    accounts, tx = toy
    roles = dict(zip(accounts.account_id, accounts.role))
    fraud = tx[tx.scenario != "normal"]
    assert sorted(fraud.scenario.unique()) == ["digital_arrest", "fake_kyc", "job_task"]
    assert (fraud.is_fraud == 1).all()
    for scenario, ring_tx in fraud.groupby("scenario"):
        assert time_respecting_chain_exists(ring_tx, roles), scenario


def test_ring_shapes(toy):
    accounts, tx = toy
    roles = dict(zip(accounts.account_id, accounts.role))
    fraud = tx[tx.scenario != "normal"]
    kyc = fraud[(fraud.scenario == "fake_kyc") & (fraud.sender_id.map(roles) == "mule_t1")]
    assert kyc.receiver_id.nunique() >= 3 or fraud[
        (fraud.scenario == "fake_kyc") & (fraud.sender_id.map(roles) == "victim")
    ].receiver_id.nunique() >= 3  # fan-out
    job_in = fraud[(fraud.scenario == "job_task") & (fraud.receiver_id.map(roles) == "mule_t1")]
    assert job_in.groupby("receiver_id").sender_id.nunique().max() >= 5  # fan-in


def test_toy_deterministic():
    a1, t1 = make_toy_data(seed=7)
    a2, t2 = make_toy_data(seed=7)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(t1, t2)


def test_written_files_roundtrip(toy, tmp_path):
    accounts, tx = toy
    write_toy_data(accounts, tx, tmp_path)
    text = (tmp_path / "transactions.csv").read_text().splitlines()[1]
    assert "T" in text.split(",")[1] and text.split(",")[1].endswith("+05:30")
    accounts2 = read_accounts(tmp_path / "accounts.csv")
    tx2 = read_transactions(tmp_path / "transactions.csv")
    assert validate_accounts(accounts2) == []
    assert validate_transactions(tx2, accounts2) == []
    pd.testing.assert_series_equal(tx2.amount, tx.amount)
    assert (tx2.timestamp == tx.timestamp).all()
