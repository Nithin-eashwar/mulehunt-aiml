import math
import re
from pathlib import Path

import pandas as pd
import pytest
from b_search_helpers import assert_time_respecting

from mulehunt import search
from mulehunt.search import (UNIT_COST, astar, bfs, dfs, dls, iddfs, layered_heuristic, make_cost_fn,
                             path_cost, reverse_bfs_heuristic, ucs)

UNINFORMED = [bfs, dfs, dls, iddfs]
ALL = UNINFORMED + [ucs, astar]


@pytest.fixture
def two_trails(make_graph):
    """S receives at t=0. Long trail S-a-b-c-Z leaves first (t=1); short S-x-Z leaves later (t=2)."""
    return make_graph([("V", "S", 0, 100), ("S", "a", 1, 1), ("a", "b", 2, 1), ("b", "c", 3, 1), ("c", "Z", 4, 1),
                       ("S", "x", 2, 1), ("x", "Z", 3, 1)], roles={"Z": "cashout"})


def start(g, txn="TXN00000000"):
    return g.start_state(txn)


def test_bfs_shortest_dfs_long(two_trails):
    g = two_trails
    s = start(g)
    rb, rd = bfs(g, s, {"Z"}), dfs(g, s, {"Z"})
    assert rb.found and rb.hops == 2 and [e.receiver for e in rb.path] == ["x", "Z"]
    assert rd.found and rd.hops == 4
    for r in (rb, rd):
        assert_time_respecting(r.path, start_time_ns=s[1])


def test_dls_cutoff_vs_failure(two_trails, make_graph):
    g = two_trails
    r = dls(g, start(g), {"Z"}, limit=1)
    assert not r.found and r.cutoff
    r = dls(g, start(g), {"Z"}, limit=2)
    assert r.found and r.hops == 2
    dead = make_graph([("V", "S", 0, 1), ("S", "a", 1, 1)], roles={"Z": "cashout"})
    r = dls(dead, start(dead), {"Z"}, limit=5)
    assert not r.found and not r.cutoff


def test_iddfs_depth_equals_bfs(two_trails, toy_graph):
    assert iddfs(two_trails, start(two_trails), {"Z"}).hops == bfs(two_trails, start(two_trails), {"Z"}).hops
    goals = toy_graph.known_cashouts()
    for txn in ("TXN00000717", "TXN00002374", "TXN00001531"):
        s = toy_graph.start_state(txn)
        assert iddfs(toy_graph, s, goals).hops == bfs(toy_graph, s, goals).hops


def test_time_rules_block_trails(make_graph):
    out_of_order = make_graph([("V", "S", 5, 1), ("S", "a", 6, 1), ("a", "Z", 5.5, 1)], roles={"Z": "cashout"})
    over_window = make_graph([("V", "S", 0, 1), ("S", "a", 1, 1), ("a", "Z", 50, 1)], roles={"Z": "cashout"})
    for g in (out_of_order, over_window):
        for alg in ALL:
            assert not alg(g, start(g), {"Z"}).found, alg.__name__


def test_cycle_terminates(make_graph):
    g = make_graph([("V", "A", 0, 1), ("A", "B", 1, 1), ("B", "A", 2, 1), ("A", "B", 3, 1), ("B", "A", 4, 1),
                    ("B", "Z", 5, 1)], roles={"Z": "cashout"})
    for alg in ALL:
        r = alg(g, start(g), {"Z"})
        assert r.found, alg.__name__
    assert dfs(g, start(g), {"Z"}, graph_search=True).found


def test_find_all(make_graph):
    g = make_graph([("V", "S", 0, 1), ("S", "Y", 1, 1), ("S", "a", 2, 1), ("a", "Z", 3, 1)],
                   roles={"Y": "cashout", "Z": "cashout"})
    for alg in ALL:
        r = alg(g, start(g), {"Y", "Z"}, find_all=True)
        assert set(r.goal_paths) == {"Y", "Z"}, alg.__name__


def test_start_is_goal(make_graph):
    g = make_graph([("V", "Z", 0, 1)], roles={"Z": "cashout"})
    for alg in ALL:
        r = alg(g, start(g), {"Z"})
        assert r.found and r.path == [] and r.cost == 0


def test_max_expansions_truncates(two_trails):
    for alg in ALL:
        r = alg(two_trails, start(two_trails), {"Z"}, max_expansions=1)
        assert r.truncated and not r.found, alg.__name__


def test_measure_memory(two_trails):
    r = bfs(two_trails, start(two_trails), {"Z"}, measure_memory=True)
    assert r.peak_bytes and r.peak_bytes > 0
    assert bfs(two_trails, start(two_trails), {"Z"}).peak_bytes is None


def _victim_payments(g, scenario):
    tx = g.eval_tx
    rows = tx[(tx.is_fraud == 1) & (tx.scenario == scenario)]
    return [t for t, s, r in zip(rows.txn_id, rows.sender_id, rows.receiver_id)
            if g.eval_role[s] == "victim" and g.eval_role[r] == "mule_t1"]


@pytest.mark.parametrize("scenario", ["fake_kyc", "job_task", "digital_arrest"])
def test_every_algorithm_traces_toy_rings(toy_graph, scenario):
    goals = toy_graph.known_cashouts()
    cost = make_cost_fn(None)
    for alg in ALL:
        ok = False
        for txn in _victim_payments(toy_graph, scenario):
            s = toy_graph.start_state(txn)
            r = alg(toy_graph, s, goals, cost_fn=cost) if alg in (ucs, astar) else alg(toy_graph, s, goals)
            if r.found:
                assert_time_respecting(r.path, start_time_ns=s[1])
                assert r.path[0].sender == s[0]
                ok = True
        assert ok, (alg.__name__, scenario)


# ---------------------------------------------------------------- A*

def _toy_starts(g):
    return [g.start_state(t) for sc in ("fake_kyc", "job_task", "digital_arrest") for t in _victim_payments(g, sc)]


def test_astar_matches_ucs_and_expands_no_more(two_trails, toy_graph):
    risk = {"a": 0.9, "b": 0.9, "c": 0.9, "x": 0.2, "Z": 0.99}
    cost = make_cost_fn(risk)
    for g, starts in ((two_trails, [start(two_trails)]), (toy_graph, _toy_starts(toy_graph))):
        goals = {"Z"} if g is two_trails else g.known_cashouts()
        for h in ("reverse_bfs", "layered"):
            hf = reverse_bfs_heuristic(g, goals, cost) if h == "reverse_bfs" else layered_heuristic(g, goals, cost)
            for s in starts:
                u, a = ucs(g, s, goals, cost_fn=cost), astar(g, s, goals, cost_fn=cost, heuristic=hf)
                assert u.found == a.found
                assert math.isclose(u.cost, a.cost, abs_tol=1e-9)
                assert a.nodes_expanded <= u.nodes_expanded
                assert_time_respecting(a.path, start_time_ns=s[1])
    r = ucs(two_trails, start(two_trails), {"Z"}, cost_fn=cost)
    assert [e.receiver for e in r.path] == ["a", "b", "c", "Z"]  # most suspicious, not shortest
    assert math.isclose(r.cost, path_cost(r.path, cost))


def test_heuristic_values(make_graph, toy_graph):
    g = make_graph([("V", "S", 0, 1), ("S", "Z", 1, 1), ("Q", "R", 0, 1)], roles={"Z": "cashout"})
    cost = make_cost_fn({"Z": 0.5})
    for hf in (reverse_bfs_heuristic(g, {"Z"}, cost), layered_heuristic(g, {"Z"}, cost)):
        assert hf("Z") == 0 and hf("Q") == math.inf and hf("R") == math.inf
        assert hf("S") > 0


def test_admissibility_on_toy(toy_graph):
    from mulehunt.graph import placeholder_risk_scores
    goals = toy_graph.known_cashouts()
    cost = make_cost_fn(placeholder_risk_scores(toy_graph))
    h1, h2 = reverse_bfs_heuristic(toy_graph, goals, cost), layered_heuristic(toy_graph, goals, cost)
    for s in _toy_starts(toy_graph):
        r = ucs(toy_graph, s, goals, cost_fn=cost)
        if r.found:
            assert h1(s[0]) <= h2(s[0]) + 1e-12 <= r.cost + 2e-12


def test_search_module_reads_no_labels():
    src = Path(search.__file__).read_text()
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = "\n".join(line.split("#")[0] for line in code.splitlines())
    for word in ("role", "is_fraud", "scenario", "eval_"):
        assert word not in code, word
