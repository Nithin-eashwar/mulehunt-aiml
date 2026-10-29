import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulehunt import graph as graph_mod
from mulehunt.graph import TxGraph, load_risk_scores, placeholder_risk_scores, to_timestamp

H = 3600 * 10**9
REAL = Path("data/generated/transactions.csv")


def test_successors_strictly_later_and_within_window(make_graph):
    g = make_graph([("A", "B", 0, 1), ("B", "C", 0, 1), ("B", "D", 10, 1), ("B", "E", 48, 1), ("B", "F", 48.01, 1)])
    t0 = g.start_state("TXN00000000")[1]
    got = [e.receiver for e in g.successors("B", t0)]
    assert got == ["D", "E"]  # +0 h illegal, +48 h legal, just over 48 h illegal


def test_window_parameter(make_graph):
    g = make_graph([("A", "B", 0, 1), ("B", "C", 10, 1), ("B", "D", 30, 1)])
    t0 = g.start_state("TXN00000000")[1]
    assert [e.receiver for e in g.successors("B", t0, window=pd.Timedelta(hours=12))] == ["C"]
    g12 = make_graph([("A", "B", 0, 1), ("B", "C", 10, 1), ("B", "D", 30, 1)], window=pd.Timedelta(hours=12))
    assert [e.receiver for e in g12.successors("B", t0)] == ["C"]
    assert len(g.successors("B")) == 2  # no arrival time: every outgoing edge


def test_toy_fallback(tmp_path, monkeypatch):
    g, kind = TxGraph.load(tmp_path)
    assert kind == "toy" and g.n_accounts == 500 and g.n_edges == 3016


def test_degrees_predecessors_and_lookup(make_graph):
    g = make_graph([("A", "B", 0, 5), ("C", "B", 1, 6), ("B", "D", 2, 7)])
    assert g.in_degree("B") == 2 and g.out_degree("B") == 1 and g.out_degree("D") == 0
    assert sorted(e.sender for e in g.predecessors("B")) == ["A", "C"]
    e = g.edge_by_txn("TXN00000002")
    assert (e.sender, e.receiver, e.amount) == ("B", "D", 7.0)
    assert g.start_state("TXN00000002") == ("D", e.time_ns)
    assert to_timestamp(e.time_ns).tz is not None


def test_reverse_bfs_distances(make_graph):
    g = make_graph([("A", "B", 0, 1), ("B", "C", 1, 1), ("C", "Z", 2, 1), ("A", "Z", 0, 1), ("Q", "A", 0, 1)],
                   roles={"Z": "cashout"})
    assert g.known_cashouts() == {"Z"}
    d = g.hop_distance_to({"Z"})
    assert d == {"Z": 0, "C": 1, "A": 1, "B": 2, "Q": 2}


def test_placeholder_risk_bounded_and_label_free(make_graph):
    edges = [("A", "B", 0, 1), ("B", "C", 1, 1)]
    r1 = placeholder_risk_scores(make_graph(edges))
    r2 = placeholder_risk_scores(make_graph(edges, roles={"B": "mule_t1", "C": "cashout"}))
    assert r1 == r2
    assert all(0.01 <= v <= 0.5 for v in r1.values())
    assert r1["B"] > r1["A"]


def test_load_risk_scores_missing_file(tmp_path):
    assert load_risk_scores(tmp_path / "nope.csv") is None
    assert load_risk_scores() is not None  # the committed file


@pytest.mark.skipif(not REAL.exists(), reason="real data not generated")
def test_real_graph_load():
    import resource
    t0 = time.perf_counter()
    g, kind = TxGraph.load()
    elapsed = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = rss if sys.platform == "darwin" else rss * 1024  # bytes on macOS, KiB on Linux
    assert kind == "real" and g.n_edges == 1_803_905
    out = Path("reports/b_tables")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"edges": [g.n_edges], "accounts": [g.n_accounts], "load_s": [round(elapsed, 2)],
                  "peak_rss_mb_process": [round(peak / 2**20, 1)]}).to_csv(out / "graph_load_timing.csv", index=False)
    assert elapsed < 60 and peak < 2**30
