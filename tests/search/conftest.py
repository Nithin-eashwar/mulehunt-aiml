from pathlib import Path

import pandas as pd
import pytest

from mulehunt.graph import TxGraph
from simulator.schema import IST_TZ, read_accounts, read_transactions

BASE = pd.Timestamp("2025-09-01T00:00:00", tz=IST_TZ)
TOY = Path("data/generated/toy")


def build_graph(edges, roles=None, window=pd.Timedelta(hours=48)):
    """edges: (sender, receiver, hours_after_base, amount) tuples."""
    roles = roles or {}
    names = sorted({a for e in edges for a in e[:2]} | set(roles))
    accounts = pd.DataFrame({"account_id": names, "role": [roles.get(a, "normal") for a in names],
                             "is_fraud": 0})
    tx = pd.DataFrame({
        "txn_id": [f"TXN{i:08d}" for i in range(len(edges))],
        "timestamp": [BASE + pd.Timedelta(hours=h) for _, _, h, _ in edges],
        "sender_id": [s for s, *_ in edges], "receiver_id": [r for _, r, *_ in edges],
        "amount": [float(a) for *_, a in edges], "scenario": "normal", "is_fraud": 0})
    return TxGraph(accounts, tx, window)


@pytest.fixture
def make_graph():
    return build_graph


@pytest.fixture(scope="session")
def toy_graph():
    """Always the toy data, never the real data, so ring-shape assertions stay valid."""
    g = TxGraph(read_accounts(TOY / "accounts.csv"), read_transactions(TOY / "transactions.csv"))
    g.data_dir = TOY
    return g
