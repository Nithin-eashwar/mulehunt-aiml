from pathlib import Path

import pytest

from app.logic import demo_transactions, game_run, higher_score, run_trail, score_account, trail_html, \
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
        html = trail_html(toy_graph, txn, run_trail(toy_graph, txn, "BFS", 48, risk).path, risk,
                          load_rings(TOY), reveal=True)
        assert "<html>" in html


def test_game_run_accounting():
    rows, acc, counts = game_run(100_000, 2, 100, "minimax", {})
    assert abs(acc["total"] - 100_000) < 1e-6
    assert counts["alpha-beta + ordering"] <= counts["minimax"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_runs(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
