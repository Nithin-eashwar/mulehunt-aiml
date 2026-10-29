import math
import re
from pathlib import Path

from mulehunt import search
from mulehunt.trace import Searchers, all_trail_accounts, load_rings, ring_starts, trace_ring
from mulehunt.graph import placeholder_risk_scores

TOY = Path("data/generated/toy")


def test_load_rings_toy():
    rings = load_rings(TOY)
    assert sorted(r["scenario"] for r in rings) == ["digital_arrest", "fake_kyc", "job_task"]
    for r in rings:
        assert r["cashout"] and r["mules"]["mule_t1"] and r["victims"]


def test_ring_starts_nonempty(toy_graph):
    for r in load_rings(TOY):
        starts = ring_starts(toy_graph, r)
        assert starts
        assert all(acc in r["mules"]["mule_t1"] for acc, _, _ in starts)
    job = next(r for r in load_rings(TOY) if r["scenario"] == "job_task")
    assert len(ring_starts(toy_graph, job)) == 6


def test_trace_ring_bfs(toy_graph):
    s = Searchers(toy_graph, placeholder_risk_scores(toy_graph), toy_graph.known_cashouts())
    for r in load_rings(TOY):
        for alg in ("bfs", "dfs", "dls", "iddfs", "ucs", "astar_h1", "astar_h2"):
            out = trace_ring(toy_graph, r, alg, searchers=s)
            assert out["found_own"], (alg, r["scenario"])
            p = out["trail_precision"]
            assert math.isnan(p) or 0 <= p <= 1
            assert out["members_found"] <= set(r["mules"]["mule_t1"] + r["mules"]["mule_t2"] +
                                               r["mules"]["mule_t3"] + [r["cashout"]])


def test_all_trails_recovers_fan_out(toy_graph):
    kyc = next(r for r in load_rings(TOY) if r["scenario"] == "fake_kyc")
    acc, t, _ = ring_starts(toy_graph, kyc)[0]
    found, truncated = all_trail_accounts(toy_graph, (acc, t), toy_graph.known_cashouts())
    assert not truncated
    assert set(kyc["mules"]["mule_t2"]) <= found  # all three tier-2 mules, not just one trail


def test_search_module_never_reads_role():
    code = re.sub(r'"""[\s\S]*?"""', "", Path(search.__file__).read_text())
    assert "role" not in code
