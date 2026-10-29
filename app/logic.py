"""Plain, testable logic behind the dashboard pages (no Streamlit here)."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from mulehunt import b_params as P
from mulehunt.search import astar, bfs, iddfs, layered_heuristic, make_cost_fn
from mulehunt.trace import ring_members, ring_starts

FEATURES = Path("data/processed/features.parquet")
MULE_TIERS = ("mule_t1", "mule_t2", "mule_t3")


# ---------------------------------------------------------------- page 1: transaction check

def score_account(account, risk, threshold, features=None, bundle=None) -> dict:
    """risk_prob (from risk_scores.csv), the model's flag, and explain() when features exist."""
    prob = None if risk is None else risk.get(account)
    out = {"account": account, "risk_prob": prob, "flagged": prob is not None and prob >= threshold,
           "explain": None}
    if features is not None and bundle is not None and account in features.index:
        from mulehunt.models import explain
        out["explain"] = explain(bundle, features.loc[account])
    return out


def higher_score(a: dict, b: dict) -> dict:
    """B-D19: models score accounts, so a payment gets the higher of its two accounts' scores."""
    pa = -1 if a["risk_prob"] is None else a["risk_prob"]
    pb = -1 if b["risk_prob"] is None else b["risk_prob"]
    return a if pa >= pb else b


def transaction_check(graph, txn_id, risk, threshold, features=None, bundle=None) -> dict:
    e = graph.edge_by_txn(txn_id)
    s = score_account(e.sender, risk, threshold, features, bundle)
    r = score_account(e.receiver, risk, threshold, features, bundle)
    return {"edge": e, "sender": s, "receiver": r, "decision": higher_score(s, r)}


def demo_transactions(graph, rings, per_scenario=2, n_genuine=3, seed=0) -> pd.DataFrame:
    """Flagged victim payments (first ones of a few rings per scenario) plus random genuine ones."""
    rows = []
    for scen in sorted({r["scenario"] for r in rings}):
        for ring in [r for r in rings if r["scenario"] == scen][:per_scenario]:
            starts = ring_starts(graph, ring)
            if starts:
                acc, t, txn = starts[0]
                rows.append({"txn_id": txn, "kind": f"victim payment ({scen})", "ring_id": ring["ring_id"]})
    tx = graph.eval_tx
    genuine = tx[tx.scenario == "normal"]
    if len(genuine):
        pick = genuine.sample(min(n_genuine, len(genuine)), random_state=seed)
        rows += [{"txn_id": t, "kind": "genuine payment", "ring_id": None} for t in pick.txn_id]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- page 2: mule trail

def run_trail(graph, txn_id, algorithm, window_h, risk, max_expansions=P.MAX_EXPANSIONS, heuristic=None):
    state = graph.start_state(txn_id)
    goals = graph.known_cashouts()
    window = pd.Timedelta(hours=window_h)
    if algorithm == "BFS":
        return bfs(graph, state, goals, window=window, max_expansions=max_expansions)
    if algorithm == "IDDFS":
        return iddfs(graph, state, goals, window=window, max_expansions=max_expansions)
    cost = make_cost_fn(risk, P.DEFAULT_RISK)
    h = heuristic or layered_heuristic(graph, goals, cost)
    return astar(graph, state, goals, cost_fn=cost, heuristic=h, window=window, max_expansions=max_expansions)


def trail_context(graph, txn_id, path, max_context=6):
    """Nodes and edges to draw: the victim payment, the trail, and up to `max_context`
    one-hop neighbours of each trail account."""
    first = graph.edge_by_txn(txn_id)
    edges = [first] + list(path)
    on_trail = {first.sender} | {e.receiver for e in edges}
    context = []
    for a in [e.receiver for e in edges]:
        for e in graph.successors(a)[:max_context]:
            if e.receiver not in on_trail:
                context.append(e)
    return edges, context


def ring_of(account, rings):
    for r in rings:
        m = ring_members(r)
        if account in m:
            return r, m[account]
    return None, None


def risk_colour(p):
    """Calm green-grey at 0 to signal vermilion at 1."""
    if p is None:
        return "#D6D2C8"
    p = min(1.0, max(0.0, p))
    stops = [(0x7F, 0x9A, 0x8A), (0xE0, 0xA8, 0x5E), (0xC2, 0x41, 0x0C)]  # muted green, amber, vermilion
    lo, hi, t = (stops[0], stops[1], p / 0.5) if p < 0.5 else (stops[1], stops[2], (p - 0.5) / 0.5)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(lo, hi))


from app.style import TIER as TIER_COLOURS  # noqa: E402


def trail_html(graph, txn_id, path, risk, rings=None, reveal=False, threshold=0.5, height="480px") -> str:
    from pyvis.network import Network
    edges, context = trail_context(graph, txn_id, path)
    net = Network(height=height, width="100%", directed=True, cdn_resources="in_line", bgcolor="#FFFFFF",
                  font_color="#1B2333")
    trail_nodes = {edges[0].sender} | {e.receiver for e in edges}
    for e in edges + context:
        for a in (e.sender, e.receiver):
            if a in net.get_nodes():
                continue
            p = None if risk is None else risk.get(a)
            title = f"{a}\nrisk {p:.3f}" if p is not None else a
            colour = risk_colour(p)
            label = a[-6:]
            if reveal and rings is not None:
                _, tier = ring_of(a, rings)
                colour = TIER_COLOURS.get(tier, "#EEEEEE")
                label += f"\n{tier or 'genuine'}"
                if tier in MULE_TIERS and (p is None or p < threshold):
                    label += "\n(classifier missed)"
            net.add_node(a, label=label, title=title, color=colour, size=20 if a in trail_nodes else 8,
                         font={"face": "IBM Plex Mono", "size": 12 if a in trail_nodes else 10,
                               "color": "#1B2333" if a in trail_nodes else "#8A8F98"},
                         borderWidth=0)
    for e in edges:
        net.add_edge(e.sender, e.receiver, title=f"Rs {e.amount:,.0f}", label=f"Rs {e.amount:,.0f}", width=2.5,
                     color="#1B2333", font={"face": "IBM Plex Mono", "size": 11, "strokeWidth": 4,
                                            "strokeColor": "#FFFFFF"})
    for e in context:
        net.add_edge(e.sender, e.receiver, color="#E2DED5", width=1)
    net.set_options('{"physics": {"barnesHut": {"springLength": 130}, "stabilization": {"iterations": 200}},'
                    ' "edges": {"arrows": {"to": {"scaleFactor": 0.6}}, "smooth": false},'
                    ' "interaction": {"hover": true}}')
    page = net.generate_html()
    fonts = ("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
             "&family=IBM+Plex+Sans:wght@400;600&display=swap'><style>body{margin:0}#mynetwork{border:0!important;"
             "border-radius:12px}div.vis-tooltip{font-family:'IBM Plex Mono',monospace;font-size:12px;"
             "background:#1B2333;color:#F7F5F0;border:0;border-radius:6px;padding:6px 8px}</style>")
    return page.replace("</head>", fonts + "</head>", 1)


def trail_metrics(result, cost_fn=None) -> pd.DataFrame:
    return pd.DataFrame([{"found": result.found, "goal": result.goal, "hops": result.hops,
                          "nodes expanded": result.nodes_expanded, "peak stored": result.peak_stored,
                          "time (ms)": round(1000 * result.elapsed_s, 1),
                          "cost": None if not result.found or math.isinf(result.cost) else round(result.cost, 4),
                          "truncated": result.truncated}])


# ---------------------------------------------------------------- page 3: game

def game_run(X, R, lam, policy, fa):
    from dataclasses import replace
    from mulehunt.adversarial import (GameParams, Stats, accounting, fixed_detector, full_depth, initial_state,
                                      minimax, minimax_detector, minimax_fraudster, play_out)
    p = GameParams(X=float(X), R=int(R), alert_cost=float(lam), fa=fa)
    pol = minimax_detector(p) if policy == "minimax" else fixed_detector(policy)
    rows = play_out(pol, minimax_fraudster(p), p)
    counts = {}
    if R <= 4:
        for label, ab, order in (("minimax", False, False), ("alpha-beta", True, False),
                                 ("alpha-beta + ordering", True, True)):
            st = Stats()
            minimax(initial_state(p), full_depth(p), p, ab, order, st)
            counts[label] = st.nodes
    return pd.DataFrame(rows), accounting(rows, p), counts
