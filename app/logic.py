"""Plain, testable logic behind the dashboard pages (no Streamlit here)."""


import pandas as pd

from mulehunt import b_params as P
from mulehunt.search import astar, bfs, iddfs, layered_heuristic, make_cost_fn
from mulehunt.trace import ring_members, ring_starts

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


def trail_rows(graph, txn_id, path) -> list[dict]:
    """The trail as ledger rows: the victim payment, then one row per hop, with how long
    the money sat in the sending account before moving on."""
    from mulehunt.graph import to_timestamp
    edges = [graph.edge_by_txn(txn_id)] + list(path)
    rows = []
    for i, e in enumerate(edges):
        wait = (e.time_ns - edges[i - 1].time_ns) / 3.6e12 if i else None
        rows.append({"hop": "victim payment" if i == 0 else f"hop {i}", "from": e.sender, "to": e.receiver,
                     "amount": e.amount, "time": to_timestamp(e.time_ns), "waited_h": wait})
    return rows


from app.style import TIER_LABEL  # noqa: E402


def risk_band(p) -> str:
    """The legend's three bands, as CSS classes defined from tokens."""
    if p is None:
        return "tier-genuine"
    return "risk-high" if p >= 0.67 else ("risk-mid" if p >= 0.33 else "risk-low")


def trail_svg(graph, txn_id, path, risk, rings=None, reveal=False, threshold=0.5, width=1000) -> str:
    """The trail as inline SVG: trail accounts left to right, each one's one-hop context
    fanned around it. Colours are token classes; hovering a node shows its id and risk."""
    import html as _html
    import math as _math
    edges, context = trail_context(graph, txn_id, path)
    chain = [edges[0].sender] + [e.receiver for e in edges]
    step = (width - 160) / max(len(chain) - 1, 1)
    height = 360
    pos = {a: (80 + i * step, height / 2) for i, a in enumerate(chain)}
    by_parent = {}
    for e in context:
        by_parent.setdefault(e.sender, []).append(e.receiver)
    for j, (parent, kids) in enumerate(by_parent.items()):
        px, py = pos[parent]
        side = -1 if j % 2 == 0 else 1  # alternate above and below the chain
        for k, kid in enumerate(kids):
            if kid in pos:
                continue
            ang = _math.radians(-150 + 120 * (k + 0.5) / len(kids)) * side
            pos[kid] = (px + 95 * _math.cos(ang), py + 110 * _math.sin(ang))

    def cls(a):
        p = None if risk is None else risk.get(a)
        if reveal and rings is not None:
            tier = ring_of(a, rings)[1]
            return f"tier-{tier or 'genuine'}", tier, p
        return risk_band(p), None, p

    parts = [f"<svg class='mh-graph' viewBox='0 0 {width} {height}' role='img' "
             f"aria-label='Trail of {len(edges)} payments from {chain[0]} to {chain[-1]}'>",
             "<defs><marker id='mh-arrow' viewBox='0 0 8 8' refX='7' refY='4' markerWidth='7' markerHeight='7' "
             "orient='auto'><path d='M0,0 L8,4 L0,8 z' class='arrow'/></marker></defs>"]
    for e in context:
        (x1, y1), (x2, y2) = pos[e.sender], pos[e.receiver]
        parts.append(f"<line class='ctx' x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}'/>")
    for e in edges:
        (x1, y1), (x2, y2) = pos[e.sender], pos[e.receiver]
        parts.append(f"<line class='hop' x1='{x1 + 16:.1f}' y1='{y1:.1f}' x2='{x2 - 18:.1f}' y2='{y2:.1f}' "
                     "marker-end='url(#mh-arrow)'/>")
        parts.append(f"<text class='amt' x='{(x1 + x2) / 2:.1f}' y='{y1 - 12:.1f}'>Rs {e.amount:,.0f}</text>")
    on_chain = set(chain)
    for a, (x, y) in pos.items():
        c, tier, p = cls(a)
        big = a in on_chain
        tip = f"{a}, risk {p:.3f}" if p is not None else f"{a}, no score"
        label = a[-6:]
        sub = ""
        if reveal and rings is not None and big:
            sub = TIER_LABEL.get(tier, "genuine")
            if tier in MULE_TIERS and (p is None or p < threshold):
                sub += ", classifier missed"
        parts.append(f"<g class='node{' on' if big else ''}'><title>{_html.escape(tip)}</title>"
                     f"<circle class='{c}' cx='{x:.1f}' cy='{y:.1f}' r='{14 if big else 6}'/>")
        if big:
            parts.append(f"<text class='id' x='{x:.1f}' y='{y + 32:.1f}'>{label}</text>")
            if sub:
                parts.append(f"<text class='sub' x='{x:.1f}' y='{y + 48:.1f}'>{_html.escape(sub)}</text>")
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- page 3: game

def game_run(X, R, lam, policy, fa):
    from mulehunt.adversarial import (GameParams, Stats, accounting, best_response, fixed_detector, full_depth,
                                      initial_state, minimax, minimax_detector, minimax_fraudster, play_out)
    p = GameParams(X=float(X), R=int(R), alert_cost=float(lam), fa=fa)
    if policy == "minimax":
        rows = play_out(minimax_detector(p), minimax_fraudster(p), p)
    else:
        # against a rule that never changes, the fraudster's best reply is the exact best response
        # (minimax_fraudster would assume the bank switches to minimax on later days)
        _, plan = best_response(policy, p)
        rows = play_out(fixed_detector(policy), lambda state, d: plan[state[0]], p)
    counts = {}
    if R <= 4:
        for label, ab, order in (("minimax", False, False), ("alpha-beta", True, False),
                                 ("alpha-beta + ordering", True, True)):
            st = Stats()
            minimax(initial_state(p), full_depth(p), p, ab, order, st)
            counts[label] = st.nodes
    return pd.DataFrame(rows), accounting(rows, p), counts
