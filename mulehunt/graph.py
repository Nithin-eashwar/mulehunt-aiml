"""Time-respecting transaction graph for tracing money (Person B, Unit II search).

A hop from account u to v is only legal if it happens strictly after the money
arrived at u and at most `window` (B-D2: 48 h) after it. Edges are held as numpy
arrays in CSR layout sorted by (sender, time), plus an incoming index sorted by
(receiver, time), so the 1.8M real edges load quickly and `successors` is two
binary searches. NetworkX is used only for drawing elsewhere.

Labels (`role`, `is_fraud`, `scenario`) are kept in `eval_*` attributes for
evaluation only; search logic never reads them, except the documented B-D1 goal
set returned by `known_cashouts()`.
"""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

from mulehunt.b_params import GOAL_ROLE, HOP_WINDOW_H
from simulator.schema import IST_TZ, read_accounts, read_transactions

DATA_DIR = Path("data/generated")
TOY_DIR = DATA_DIR / "toy"
RISK_SCORES = Path("models/risk_scores.csv")

Edge = namedtuple("Edge", "txn_id sender receiver time_ns amount")


def to_ns(ts) -> np.ndarray:
    """tz-aware timestamps -> int64 nanoseconds since the epoch (UTC), whatever pandas' unit."""
    return pd.DatetimeIndex(ts).tz_convert("UTC").as_unit("ns").asi8.astype(np.int64)


def to_timestamp(time_ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(time_ns), unit="ns", tz="UTC").tz_convert(IST_TZ)


def _window_ns(window) -> int:
    return int(pd.Timedelta(window).value)


class TxGraph:
    def __init__(self, accounts: pd.DataFrame, tx: pd.DataFrame, window=pd.Timedelta(hours=HOP_WINDOW_H)):
        self.window_ns = _window_ns(window)
        self.ids = accounts.account_id.astype(str).to_numpy()
        self.index = {a: i for i, a in enumerate(self.ids)}
        n = len(self.ids)
        src = tx.sender_id.map(self.index).to_numpy(dtype=np.int64)
        dst = tx.receiver_id.map(self.index).to_numpy(dtype=np.int64)
        t = to_ns(tx.timestamp)
        order = np.lexsort((t, src))
        self.src, self.dst, self.t = src[order], dst[order], t[order]
        self.amount = tx.amount.to_numpy(dtype=float)[order]
        self.txn = tx.txn_id.astype(str).to_numpy()[order]
        self.out_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.src, minlength=n))])
        in_order = np.lexsort((self.t, self.dst))
        self.in_edges = in_order  # positions in the (sender, time) arrays, sorted by (receiver, time)
        self.in_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.dst, minlength=n))])
        self._txn_order = np.argsort(self.txn)
        self._txn_sorted = self.txn[self._txn_order]
        # evaluation-only labels
        self.eval_role = dict(zip(self.ids, accounts.role)) if "role" in accounts else {}
        self.eval_tx = tx[[c for c in ("txn_id", "sender_id", "receiver_id", "timestamp", "amount",
                                       "scenario", "is_fraud") if c in tx]]
        self._goal_ids = [a for a, r in self.eval_role.items() if r == GOAL_ROLE]

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, data_dir=None, window=pd.Timedelta(hours=HOP_WINDOW_H)):
        """(graph, "real" | "toy"); falls back to the toy data when the real CSVs are missing."""
        data_dir = Path(data_dir or DATA_DIR)
        kind = "real"
        if not ((data_dir / "accounts.csv").exists() and (data_dir / "transactions.csv").exists()):
            data_dir, kind = TOY_DIR, "toy"
        g = cls(read_accounts(data_dir / "accounts.csv"), read_transactions(data_dir / "transactions.csv"), window)
        g.data_dir = data_dir
        return g, kind

    # ------------------------------------------------------------ edges
    @property
    def n_accounts(self) -> int:
        return len(self.ids)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def _edge(self, i: int) -> Edge:
        return Edge(self.txn[i], self.ids[self.src[i]], self.ids[self.dst[i]], int(self.t[i]), float(self.amount[i]))

    def successors(self, account: str, arrival_ns=None, window=None) -> list[Edge]:
        """Outgoing edges strictly after `arrival_ns` and at most `window` after it
        (all outgoing edges when `arrival_ns` is None)."""
        i = self.index[account]
        lo, hi = self.out_ptr[i], self.out_ptr[i + 1]
        if arrival_ns is not None:
            w = self.window_ns if window is None else _window_ns(window)
            seg = self.t[lo:hi]
            lo, hi = lo + np.searchsorted(seg, arrival_ns, "right"), lo + np.searchsorted(seg, arrival_ns + w, "right")
        return [self._edge(k) for k in range(lo, hi)]

    def predecessors(self, account: str) -> list[Edge]:
        i = self.index[account]
        return [self._edge(k) for k in self.in_edges[self.in_ptr[i]:self.in_ptr[i + 1]]]

    def out_degree(self, account: str) -> int:
        i = self.index[account]
        return int(self.out_ptr[i + 1] - self.out_ptr[i])

    def in_degree(self, account: str) -> int:
        i = self.index[account]
        return int(self.in_ptr[i + 1] - self.in_ptr[i])

    def edge_by_txn(self, txn_id: str) -> Edge:
        k = int(np.searchsorted(self._txn_sorted, txn_id))
        if k >= len(self._txn_sorted) or self._txn_sorted[k] != txn_id:
            raise KeyError(txn_id)
        return self._edge(int(self._txn_order[k]))

    def start_state(self, txn_id: str) -> tuple[str, int]:
        """B-D3: the state right after this payment lands, (receiver, time_ns)."""
        e = self.edge_by_txn(txn_id)
        return e.receiver, e.time_ns

    # ------------------------------------------------------------ goals
    def known_cashouts(self) -> set[str]:
        """B-D1: accounts the bank has seen withdraw cash (role == "cashout")."""
        return set(self._goal_ids)

    def hop_distance_to(self, goals) -> dict[str, int]:
        """Reverse BFS, ignoring time: fewest hops from each account to any goal."""
        n = self.n_accounts
        dist = np.full(n, -1, dtype=np.int64)
        frontier = np.zeros(n, dtype=bool)
        for a in goals:
            if a in self.index:
                frontier[self.index[a]] = True
        dist[frontier] = 0
        pairs = np.unique(np.stack([self.src, self.dst], axis=1), axis=0) if self.n_edges else np.zeros((0, 2), int)
        s, d = pairs[:, 0], pairs[:, 1]
        level = 0
        while frontier.any():
            level += 1
            cand = s[frontier[d] & (dist[s] < 0)]
            frontier = np.zeros(n, dtype=bool)
            frontier[cand] = True
            dist[frontier] = level
        return {self.ids[i]: int(dist[i]) for i in np.flatnonzero(dist >= 0)}


# ---------------------------------------------------------------- risk scores

def load_risk_scores(path=None) -> dict[str, float] | None:
    path = Path(path or RISK_SCORES)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"account_id": str})
    return dict(zip(df.account_id, df.risk_prob.astype(float)))


def placeholder_risk_scores(graph: TxGraph) -> dict[str, float]:
    """Label-free stand-in for tests and toy runs: accounts that both receive and send
    (pass-through shape) look riskier. Always in [0.01, 0.5]."""
    out = np.diff(graph.out_ptr)
    inn = np.diff(graph.in_ptr)
    balance = np.minimum(out, inn) / (np.maximum(out, inn) + 1)
    risk = 0.01 + 0.49 * balance
    return dict(zip(graph.ids, risk.astype(float)))
