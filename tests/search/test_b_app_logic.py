from pathlib import Path

import pytest

from app.logic import demo_transactions, game_run, higher_score, run_trail, score_account, trail_svg, \
    transaction_check
from mulehunt.graph import TxGraph, placeholder_risk_scores
from mulehunt.trace import load_rings

TOY = Path("data/generated/toy")
PAGES = sorted(Path("app/views").glob("*.py"))


def test_transaction_scores_take_the_higher(toy_graph):
    risk = placeholder_risk_scores(toy_graph)
    res = transaction_check(toy_graph, "TXN00000717", risk, 0.3)
    e = res["edge"]
    assert res["sender"]["account"] == e.sender and res["receiver"]["account"] == e.receiver
    assert res["decision"]["risk_prob"] == max(risk[e.sender], risk[e.receiver])


def test_higher_score_handles_missing():
    a, b = {"risk_prob": None}, {"risk_prob": 0.2}
    assert higher_score(a, b) is b and higher_score(b, a) is b
    assert score_account("X", None, 0.5)["risk_prob"] is None


def test_toy_fallback(tmp_path):
    g, kind = TxGraph.load(tmp_path)
    assert kind == "toy"
    assert len(load_rings(g.data_dir)) == 3


def test_demo_list(toy_graph):
    demo = demo_transactions(toy_graph, load_rings(TOY))
    assert set(demo.kind.str.startswith("victim payment").unique()) == {True, False}
    assert demo.ring_id.notna().sum() == 3
    risk = placeholder_risk_scores(toy_graph)
    for txn in demo[demo.ring_id.notna()].txn_id:
        for alg in ("BFS", "IDDFS", "A*"):
            assert run_trail(toy_graph, txn, alg, 48, risk).found
        svg = trail_svg(toy_graph, txn, run_trail(toy_graph, txn, "BFS", 48, risk).path, risk,
                        load_rings(TOY), reveal=True)
        assert svg.startswith("<svg") and "#" not in svg.replace("url(#mh-arrow)", "")  # colours come from classes


def test_game_run_accounting():
    rows, acc, counts = game_run(100_000, 2, 100, "minimax", {})
    assert abs(acc["total"] - 100_000) < 1e-6
    assert counts["alpha-beta + ordering"] <= counts["minimax"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_runs(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


def test_trail_rows_chain_and_wait(toy_graph):
    from app.logic import trail_rows
    risk = placeholder_risk_scores(toy_graph)
    res = run_trail(toy_graph, "TXN00002374", "BFS", 48, risk)  # toy digital_arrest chain, 3 hops
    rows = trail_rows(toy_graph, "TXN00002374", res.path)
    assert rows[0]["hop"] == "victim payment" and rows[0]["waited_h"] is None
    assert len(rows) == len(res.path) + 1
    for a, b in zip(rows, rows[1:]):
        assert b["from"] == a["to"] and 0 < b["waited_h"] <= 48


def test_game_run_fixed_rule_uses_best_response():
    from mulehunt.adversarial import GameParams, best_response
    fa = {(T, v): 0.0 for T in (10000, 25000, 50000) for v in (False, True)}
    rows, acc, _ = game_run(200_000, 3, 0, (50000, False), fa)
    value, plan = best_response((50000, False), GameParams(alert_cost=0, fa=fa))
    assert abs(acc["payoff"] - value) < 1e-6
    assert list(zip(rows.fraction, rows.parts)) == plan
