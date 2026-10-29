"""Tracing study: run every search from every flagged victim payment of every ring.

    python -m mulehunt.trace [--data data/generated] [--quick]

Writes reports/b_02_search.md, reports/b_tables/search_*.csv and
reports/b_figures/search_*.png. Ring membership, `role`, `scenario` and
`is_fraud` are read here for evaluation only (finding the starts and scoring the
trails); the searches themselves only see the graph and the B-D1 goal set.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mulehunt import b_params as P
from mulehunt.graph import TxGraph, load_risk_scores, placeholder_risk_scores, to_timestamp
from mulehunt.search import (bfs, dfs, dls, iddfs, layered_heuristic, astar, make_cost_fn, path_cost,
                             reverse_bfs_heuristic, ucs)

REPORT = Path("reports/b_02_search.md")
FIGURES = Path("reports/b_figures")
TABLES = Path("reports/b_tables")
MULE_TIERS = ("mule_t1", "mule_t2", "mule_t3")
TIER_COLOURS = {"victim": "#8C8C8C", "fraudster": "#C44E52", "mule_t1": "#DD8452", "mule_t2": "#CCB974",
                "mule_t3": "#55A868", "cashout": "#4C72B0", "genuine": "#FFFFFF"}
ALGO_ORDER = ["bfs", "dfs", "dls", "iddfs", "ucs", "astar_h1", "astar_h2"]


# ---------------------------------------------------------------- rings and starts

def load_rings(data_dir=None) -> list[dict]:
    """Ground-truth rings. Real data: _meta/rings.json. Toy data (no _meta): one ring per
    scam scenario, built from `role` and the fraud transactions."""
    data_dir = Path(data_dir or "data/generated")
    meta = data_dir / "_meta" / "rings.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    from simulator.schema import read_accounts, read_transactions
    if not (data_dir / "transactions.csv").exists():
        data_dir = Path("data/generated/toy")
    acc = read_accounts(data_dir / "accounts.csv").set_index("account_id").role
    tx = read_transactions(data_dir / "transactions.csv")
    fraud = tx[tx.is_fraud == 1]
    rings = []
    for i, (scenario, rows) in enumerate(fraud.groupby("scenario", sort=True)):
        members = sorted(set(rows.sender_id) | set(rows.receiver_id))
        tier = lambda t: [a for a in members if acc[a] == t]
        rings.append({"ring_id": i, "scenario": scenario, "split": "toy", "start": str(rows.timestamp.min()),
                      "duration_days": None, "evasive": False, "cycle": False,
                      "fraudster": (tier("fraudster") or [None])[0], "cashout": tier("cashout")[0],
                      "mules": {t: tier(t) for t in MULE_TIERS}, "victims": tier("victim")})
    return rings


def ring_members(ring: dict) -> dict[str, str]:
    """account -> tier for one ring (victims included)."""
    out = {v: "victim" for v in ring["victims"]}
    if ring.get("fraudster"):
        out[ring["fraudster"]] = "fraudster"
    for t in MULE_TIERS:
        out.update({a: t for a in ring["mules"].get(t, [])})
    out[ring["cashout"]] = "cashout"
    return out


def ring_starts(graph: TxGraph, ring: dict) -> list[tuple[str, int, str]]:
    """B-D3: one start per victim -> tier-1 mule fraud payment of this ring, as
    (receiver, time_ns, txn_id), in time order. Labels are used only to find them."""
    tx = graph.eval_tx
    rows = tx[(tx.is_fraud == 1) & tx.sender_id.isin(ring["victims"]) & tx.receiver_id.isin(ring["mules"]["mule_t1"])]
    out = [(*graph.start_state(t), t) for t in rows.txn_id]
    return sorted(out, key=lambda s: (s[1], s[2]))


# ---------------------------------------------------------------- one ring

class Searchers:
    """The seven algorithms with shared, prebuilt costs and heuristics."""

    def __init__(self, graph, risk, goals, max_expansions=P.MAX_EXPANSIONS):
        self.graph, self.goals, self.cap = graph, goals, max_expansions
        self.cost = make_cost_fn(risk, P.DEFAULT_RISK)
        self.h1 = reverse_bfs_heuristic(graph, goals, self.cost)
        self.h2 = layered_heuristic(graph, goals, self.cost)

    def run(self, name, state, window=None, **kw):
        g, goals, cap = self.graph, self.goals, self.cap
        kw = {"window": window, "max_expansions": cap, **kw}
        if name == "bfs":
            return bfs(g, state, goals, **kw)
        if name == "dfs":
            return dfs(g, state, goals, **kw)
        if name == "dls":
            return dls(g, state, goals, limit=P.DLS_LIMIT, **kw)
        if name == "iddfs":
            return iddfs(g, state, goals, max_depth=P.MAX_DEPTH, **kw)
        if name == "ucs":
            return ucs(g, state, goals, cost_fn=self.cost, **kw)
        if name == "astar_h1":
            return astar(g, state, goals, cost_fn=self.cost, heuristic=self.h1, **kw)
        if name == "astar_h2":
            return astar(g, state, goals, cost_fn=self.cost, heuristic=self.h2, **kw)
        raise KeyError(name)


def _trail_accounts(path) -> list[str]:
    return [e.receiver for e in path]


def trace_ring(graph, ring, algorithm, risk=None, searchers=None, starts=None, window=None, **kw) -> dict:
    """Run `algorithm` from every start of `ring`; ring-level summary plus per-start rows."""
    searchers = searchers or Searchers(graph, risk, graph.known_cashouts())
    members = ring_members(ring)
    starts = ring_starts(graph, ring) if starts is None else starts
    rows, found_members = [], set()
    for account, t, txn in starts:
        r = searchers.run(algorithm, (account, t), window=window, **kw)
        trail = _trail_accounts(r.path)
        precision = (sum(a in members for a in trail) / len(trail)) if trail else math.nan
        if r.found:
            found_members |= {account} | (set(trail) & set(members))
        rows.append({"ring_id": ring["ring_id"], "txn_id": txn, "algorithm": algorithm, "found_any": r.found,
                     "found_own": r.found and r.goal == ring["cashout"], "hops": r.hops,
                     "expanded": r.nodes_expanded, "generated": r.nodes_generated, "peak_stored": r.peak_stored,
                     "time_ms": 1000 * r.elapsed_s, "truncated": r.truncated, "trail_precision": precision,
                     "cost": r.cost if r.found else math.nan, "trail": tuple(trail)})
    df = pd.DataFrame(rows)
    summary = {k: ring[k] for k in ("ring_id", "scenario", "split", "evasive", "cycle")}
    summary["algorithm"] = algorithm
    if df.empty:
        return {**summary, "found_any": False, "found_own": False, "members_found": set(), "starts": df}
    summary.update({
        "found_any": bool(df.found_any.any()), "found_own": bool(df.found_own.any()),
        "hops": df.hops.mean(), "expanded": df.expanded.mean(), "generated": df.generated.mean(),
        "peak_stored": df.peak_stored.mean(), "time_ms": df.time_ms.mean(), "truncated": int(df.truncated.sum()),
        "trail_precision": df.trail_precision.mean(), "members_found": found_members, "starts": df})
    return summary


# ---------------------------------------------------------------- all trails (investigation mode)

def all_trail_accounts(graph, state, goals, limit=P.DLS_LIMIT, window=None, max_expansions=P.MAX_EXPANSIONS):
    """Every account on any time-respecting trail of <= `limit` hops from `state` to a goal
    (tree search that keeps going after each goal). Returns (accounts, truncated)."""
    on_trail, expanded = set(), 0
    stack = [(state, 0, ())]
    while stack:
        if expanded >= max_expansions:
            return on_trail, True
        (account, t), depth, path = stack.pop()
        if depth >= limit:
            continue
        expanded += 1
        for e in graph.successors(account, t, window):
            if e.receiver in goals:
                on_trail |= set(path) | {e.receiver}
            elif e.receiver not in path:
                stack.append(((e.receiver, e.time_ns), depth + 1, path + (e.receiver,)))
    return on_trail, False


# ---------------------------------------------------------------- study

def _ring_lookup(rings):
    owner = {}
    for r in rings:
        for a, tier in ring_members(r).items():
            owner[a] = (r["ring_id"], tier)
    return owner


def comparison_table(starts: pd.DataFrame) -> pd.DataFrame:
    g = starts.groupby("algorithm", sort=False)
    out = pd.DataFrame({
        "starts": g.size(), "success_rate": g.found_own.mean(), "found_any_rate": g.found_any.mean(),
        "mean_expanded": g.expanded.mean(), "median_expanded": g.expanded.median(),
        "mean_peak_stored": g.peak_stored.mean(), "mean_hops": g.hops.mean(),
        "trail_precision": g.trail_precision.mean(), "truncated": g.truncated.sum()}).reindex(
        [a for a in ALGO_ORDER if a in set(starts.algorithm)])
    return out.reset_index().round(4)


def timing_table(starts):
    return starts.groupby("algorithm", sort=False).time_ms.agg(["mean", "median", "max"]).round(3).reset_index()


def breakdown(starts, rings, by) -> pd.DataFrame:
    info = pd.DataFrame(rings).set_index("ring_id")[[by]]
    df = starts.join(info, on="ring_id")
    t = df.groupby([by, "algorithm"]).agg(success_rate=("found_own", "mean"), mean_expanded=("expanded", "mean"),
                                          trail_precision=("trail_precision", "mean")).round(4)
    return t.reset_index()


def classifier_vs_search(graph, rings, risk, threshold, trails_by_ring, label) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per ring: mules on trails, mules flagged by the random forest, mules found only by the trail."""
    owner = _ring_lookup(rings)
    rows = []
    for r in rings:
        mules = {a for t in MULE_TIERS for a in r["mules"].get(t, [])}
        on_trail = trails_by_ring.get(r["ring_id"], set())
        trail_mules = on_trail & mules
        flagged = {a for a in mules if risk.get(a, 0.0) >= threshold}
        genuine = {a for a in on_trail if a not in owner}
        rows.append({"ring_id": r["ring_id"], "scenario": r["scenario"], "split": r["split"], "mules": len(mules),
                     "trail_mules": len(trail_mules), "rf_flagged": len(flagged),
                     "trail_only": len(trail_mules - flagged), "either": len(trail_mules | flagged),
                     "genuine_on_trails": len(genuine)})
    per_ring = pd.DataFrame(rows)
    s = per_ring.groupby("scenario")[["mules", "trail_mules", "rf_flagged", "trail_only", "either",
                                      "genuine_on_trails"]].sum()
    s.loc["all"] = s.sum()
    s["rf_recall"] = (s.rf_flagged / s.mules).round(3)
    s["trail_recall"] = (s.trail_mules / s.mules).round(3)
    s["combined_recall"] = (s.either / s.mules).round(3)
    s.insert(0, "trails", label)
    return per_ring, s.reset_index()


def window_sensitivity(searchers, rings, graph, ring_start_lists) -> pd.DataFrame:
    rows = []
    patient = []
    for w in P.WINDOW_SENSITIVITY_H:
        found = n = 0
        for r in rings:
            for account, t, txn in ring_start_lists[r["ring_id"]]:
                res = searchers.run("bfs", (account, t), window=pd.Timedelta(hours=w))
                n += 1
                found += res.found and res.goal == r["cashout"]
        rows.append({"window_h": w, "starts": n, "success_rate": round(found / max(n, 1), 4)})
    return pd.DataFrame(rows)


def max_hop_gap_hours(graph, ring):
    """Longest wait between a mule receiving and forwarding inside this ring (evaluation)."""
    members = set(ring_members(ring))
    tx = graph.eval_tx
    f = tx[(tx.is_fraud == 1) & tx.sender_id.isin(members) & tx.receiver_id.isin(members)]
    gaps = []
    for _, row in f.iterrows():
        prev = f[(f.receiver_id == row.sender_id) & (f.timestamp < row.timestamp)]
        if len(prev):
            gaps.append((row.timestamp - prev.timestamp.max()).total_seconds() / 3600)
    return max(gaps) if gaps else 0.0


def astar_vs_bfs(starts, cost) -> pd.DataFrame:
    b = starts[starts.algorithm == "bfs"].set_index("txn_id")
    a = starts[starts.algorithm == "astar_h2"].set_index("txn_id")
    both = b.found_own & a.reindex(b.index).found_own.fillna(False).astype(bool)
    b, a = b[both], a.reindex(b.index)[both]
    b_cost = b.trail.map(lambda tr: sum(cost.account_cost(x) for x in tr))
    return pd.DataFrame({"trail": ["BFS (fewest hops)", "A* h2 (most suspicious)"],
                         "starts": [len(b), len(a)],
                         "mean_hops": [b.hops.mean(), a.hops.mean()],
                         "mean_neg_log_risk_cost": [b_cost.mean(), a.cost.mean()],
                         "trail_precision": [b.trail_precision.mean(), a.trail_precision.mean()],
                         "same_trail_share": [float((b.trail == a.trail).mean()), float((b.trail == a.trail).mean())]
                         }).round(4)


def heuristic_table(searchers, cost_label):
    h1, h2 = searchers.h1, searchers.h2
    rows = [{"quantity": "m (h1 slope)", "value": f"{h1.m:.6g}"}]
    rows += [{"quantity": f"L_{j}", "value": f"{x:.6g}"} for j, x in enumerate(h2.layers)]
    return pd.DataFrame(rows)


def heuristic_verdict(cmp, searchers) -> str:
    ratio = cmp.set_index("algorithm").expanded_vs_ucs
    near = sum(searchers.h2.layers[:3])
    return (f"Measured: A*(h1) expands {ratio['astar_h1']:.1%} and A*(h2) {ratio['astar_h2']:.1%} of UCS's nodes. "
            f"Victim payments start 2-3 hops from a cash-out, and the layers 0-2 each contain a ring account the "
            f"classifier scores near 1, so h2 at a start is at most {near:.4f}, too small to reorder the frontier. "
            "Both heuristics are admissible (checked on every toy start in the tests) but uninformative on this "
            "data; h2 only helps for states far from every cash-out, where L_4 and L_5 are large.")


def heuristic_comparison(starts):
    sub = starts[starts.algorithm.isin(["ucs", "astar_h1", "astar_h2"])]
    pivot = sub.pivot_table(index="txn_id", columns="algorithm", values="cost")
    agree = {a: float(np.isclose(pivot[a], pivot["ucs"], atol=1e-9, equal_nan=True).mean())
             for a in ("astar_h1", "astar_h2")}
    g = sub.groupby("algorithm")
    out = pd.DataFrame({"mean_expanded": g.expanded.mean(), "total_expanded": g.expanded.sum(),
                        "mean_time_ms": g.time_ms.mean()}).reindex(["ucs", "astar_h1", "astar_h2"])
    out["cost_agrees_with_ucs"] = [1.0, agree["astar_h1"], agree["astar_h2"]]
    out["expanded_vs_ucs"] = out.total_expanded / out.total_expanded["ucs"]
    return out.reset_index().round(4)


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def trail_figure(graph, ring, path, path_out):
    import networkx as nx
    plt = _plt()
    members = ring_members(ring)
    G = nx.DiGraph()
    labels = {}
    for e in path:
        G.add_edge(e.sender, e.receiver)
        labels[(e.sender, e.receiver)] = f"Rs {e.amount:,.0f}\n{to_timestamp(e.time_ns):%d %b %H:%M}"
    pos = {n: (i, 0) for i, n in enumerate([path[0].sender] + [e.receiver for e in path])} if path else {}
    fig, ax = plt.subplots(figsize=(2.4 * max(len(path), 1) + 1.5, 2.6))
    colours = [TIER_COLOURS[members.get(n, "genuine")] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=colours, edgecolors="black", node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: f"{n[-4:]}\n{members.get(n, 'genuine')}" for n in G.nodes},
                            font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=15, node_size=900)
    nx.draw_networkx_edge_labels(G, pos, labels, font_size=7, ax=ax)
    ax.set_title(f"{ring['scenario']} ring {ring['ring_id']}: BFS trail from the victim payment")
    ax.axis("off")
    path_out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_out, dpi=120)
    plt.close(fig)
    return path_out


def bar_figure(df, x, y, title, ylabel, path_out, log=False):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(df[x].astype(str), df[y], color="#4C72B0")
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path_out, dpi=120)
    plt.close(fig)
    return path_out


def scenario_figure(by_scen, path_out):
    plt = _plt()
    pivot = by_scen.pivot(index="scenario", columns="algorithm", values="success_rate")[
        [a for a in ALGO_ORDER if a in set(by_scen.algorithm)]]
    ax = pivot.plot(kind="bar", figsize=(8, 3.8), width=0.8)
    ax.set_ylabel("success rate (own cash-out reached)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Tracing success by scenario")
    ax.legend(fontsize=7, ncol=4)
    ax.tick_params(axis="x", rotation=0)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path_out, dpi=120)
    plt.close(fig)
    return path_out


# ---------------------------------------------------------------- report

FORMULATION = """
**States.** `(account, arrival_time)`: which account holds the traced money, and since when.
The same account reached at two different times is two different states, because the time
decides which onward payments are possible.

**Initial state (B-D3).** A flagged victim payment: the state right after a victim to tier-1
mule payment lands, `(receiver, timestamp)`.

**Actions / transition.** From `(u, t)`, follow any payment `u -> v` at time `t'` with
`t < t' <= t + W`, giving the state `(v, t')`. `W` is the hop window (B-D2, 48 h): Person A's
simulator caps every hop at 48 h, and patient mules wait 24-47 h.

**Goal test (B-D1).** The account is a known cash-out (`role == "cashout"` in `accounts.csv`,
as if the bank had seen the cash withdrawal). This is the single ground-truth read in search
logic; everything else (tiers, ring membership, scenario) is used only to score the results.

**Path cost.** Hops for BFS / DFS / DLS / IDDFS; `-log(risk_prob of the receiver)` per edge
(B-D4) for UCS and A*, so the cheapest trail is the one whose accounts the classifier finds
most suspicious overall (the product of the risks is largest).
"""

TIME_RULE = """
Worked example (it is a unit test in `tests/search/test_b_graph.py`): money lands at account B
at 00:00. B then pays C at 00:00 (same instant: illegal, the money had not arrived "before"),
D at 10:00 (legal), E at exactly 48:00 later (legal, the boundary is inclusive) and F at
48:00:36 (illegal, outside the window). So `successors(B, 00:00) = [D, E]`.

Why the rule matters: a static graph would let a search follow a payment that happened
*before* the stolen money arrived, which cannot be carrying that money. The time rule also
makes every trail strictly increasing in time, so a tree search can never loop.
"""

CONVENTIONS = f"""
| Algorithm | Frontier | Goal test | Complete? | Optimal? |
|---|---|---|---|---|
| BFS | FIFO queue, graph search (reached set of states) | at generation | yes | fewest hops |
| DFS | LIFO stack, tree search, successors in time order | at generation | yes here (time strictly increases, finite) | no |
| DLS ({P.DLS_LIMIT}) | DFS with depth limit | at generation | only within the limit | no |
| IDDFS (<= {P.MAX_DEPTH}) | DLS with limit 1, 2, ... | at generation | yes (up to max depth) | fewest hops |
| UCS | priority queue on g | at pop | yes | cheapest cost |
| A* | priority queue on g + h | at pop | yes | cheapest cost if h admissible |

- Every run has a budget of `MAX_EXPANSIONS = {P.MAX_EXPANSIONS:,}` expansions. A run that hits
  it is reported as **truncated**, never as "not found".
- `cutoff` distinguishes "the depth limit stopped a branch" from "there is no trail".
- `peak_stored` (frontier + reached/visited states at their largest) is the deterministic
  memory measure. Tree-search DFS/DLS/IDDFS store only the stack; BFS stores every reached state.
- Several victim payments can reach the same `(account, time)` state. Tree search re-expands it,
  graph search does not, so their node counts differ; both are documented choices.
- Cash-out accounts are terminal: once a goal is reached, it is not expanded further.
"""

PROOFS = """
**h1 (plan): `h1(n) = m * d(n)`.** `d(n)` is the reverse-BFS hop distance from account `n` to
the nearest goal in the *static* graph (time ignored). Every time-respecting trail is also a
static path, so the true number of remaining hops is at least `d(n)`. Every edge on a
goal-bound trail enters an account that can still reach a goal, and `m` is the minimum cost
over exactly those entries, so each remaining hop costs at least `m`. Hence
`h1(n) <= h*(n)`: **admissible**. **Consistent:** for an edge `n -> n'`, `d(n) <= d(n') + 1`, so
`h1(n) - h1(n') <= m <= cost(n -> n')`. Accounts with no static route to a goal get `h = inf`
and are pruned (no trail can exist).

**h2 (layered, PROPOSAL): `h2(n) = sum_{j=0}^{d(n)-1} L_j`**, with
`L_j = min{ cost into account a : d(a) = j }`. One hop lowers `d` by at most 1 and a trail ends at
`d = 0`, so a trail from `n` must contain receivers at every distance `d(n)-1, ..., 0`, each at a
different position on the trail. Its cost is therefore at least the cheapest entry into each
of those layers: `h2(n) <= h*(n)`, **admissible**. **Consistent:** for `n -> n'`, if
`d(n') = d(n) - 1` then `h2(n) - h2(n') = L_{d(n)-1} <= cost(n -> n')` (the receiver `n'` is in
layer `d(n)-1`); if `d(n') >= d(n)` then `h2(n') >= h2(n)` and the inequality holds trivially.
Since every `L_j >= m`, `h2 >= h1` everywhere: h2 is never weaker.

**Why h1 is weak here.** The most suspicious account in `models/risk_scores.csv` has risk
0.99985, so `m = -log(0.99985) ~ 1.5e-4`: h1 is almost zero and A*(h1) behaves like UCS. The
layered bound uses the cheapest entry *per layer*, and the layers far from any cash-out contain
no near-certain account, so h2 is much larger there.
"""


def run_trace_study(data_dir=None, out=REPORT, fig_dir=FIGURES, tab_dir=TABLES, quick=False) -> None:
    graph, kind = TxGraph.load(data_dir)
    rings = load_rings(graph.data_dir if kind == "toy" else data_dir)
    risk = load_risk_scores() if kind == "real" else placeholder_risk_scores(graph)
    from mulehunt.models import load_model
    threshold = float(load_model("random_forest")["threshold"]) if kind == "real" else 0.3
    goals = graph.known_cashouts()
    searchers = Searchers(graph, risk, goals)
    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        rings = [r for i, r in enumerate(rings) if i % 5 == 0]

    ring_start_lists = {r["ring_id"]: ring_starts(graph, r) for r in rings}
    per_start, ring_rows = [], []
    for alg in ALGO_ORDER:
        for r in rings:
            s = trace_ring(graph, r, alg, searchers=searchers, starts=ring_start_lists[r["ring_id"]])
            per_start.append(s.pop("starts"))
            s["members_found"] = len(s["members_found"])
            ring_rows.append(s)
    starts = pd.concat(per_start, ignore_index=True)
    rings_df = pd.DataFrame(ring_rows)

    comp = comparison_table(starts)
    comp.to_csv(tab_dir / "search_comparison.csv", index=False)
    timing_table(starts).to_csv(tab_dir / "search_timing.csv", index=False)
    by_scen = breakdown(starts, rings, "scenario")
    by_split = breakdown(starts, rings, "split")
    by_evasive = breakdown(starts, rings, "evasive")
    by_cycle = breakdown(starts, rings, "cycle")
    for name, t in [("scenario", by_scen), ("split", by_split), ("evasive", by_evasive), ("cycle", by_cycle)]:
        t.to_csv(tab_dir / f"search_by_{name}.csv", index=False)
    ring_level = rings_df.groupby("algorithm", sort=False).agg(rings_traced=("found_own", "mean")).reindex(
        ALGO_ORDER).round(4).reset_index()
    ring_level.to_csv(tab_dir / "search_ring_level.csv", index=False)

    # search vs classifier: BFS single trails, and all trails <= DLS_LIMIT hops
    bfs_trails = {}
    for row in starts[starts.algorithm == "bfs"].itertuples():
        if row.found_own:
            acc = graph.edge_by_txn(row.txn_id).receiver
            bfs_trails.setdefault(row.ring_id, set()).update({acc, *row.trail})
    all_trails, n_trunc = {}, 0
    for r in rings:
        for account, t, txn in ring_start_lists[r["ring_id"]]:
            accs, tr = all_trail_accounts(graph, (account, t), goals)
            n_trunc += tr
            if accs:
                all_trails.setdefault(r["ring_id"], set()).update(accs | {account})
    _, vs_bfs = classifier_vs_search(graph, rings, risk, threshold, bfs_trails, "BFS trail per start")
    per_ring_all, vs_all = classifier_vs_search(graph, rings, risk, threshold, all_trails,
                                                f"all trails <= {P.DLS_LIMIT} hops")
    vs = pd.concat([vs_bfs, vs_all], ignore_index=True)
    vs.to_csv(tab_dir / "search_vs_classifier.csv", index=False)
    per_ring_all.to_csv(tab_dir / "search_vs_classifier_per_ring.csv", index=False)

    window = window_sensitivity(searchers, rings, graph, ring_start_lists)
    window.to_csv(tab_dir / "search_window.csv", index=False)
    gaps = pd.DataFrame({"ring_id": [r["ring_id"] for r in rings],
                         "max_hop_gap_h": [round(max_hop_gap_hours(graph, r), 2) for r in rings]})
    gaps["needs_over_24h"] = gaps.max_hop_gap_h > 24
    gaps["needs_over_12h"] = gaps.max_hop_gap_h > 12
    gaps.to_csv(tab_dir / "search_hop_gaps.csv", index=False)

    heur = heuristic_table(searchers, "neg_log_risk")
    heur.to_csv(tab_dir / "search_heuristic_values.csv", index=False)
    heur_cmp = heuristic_comparison(starts)
    heur_cmp.drop(columns=["mean_time_ms"]).to_csv(tab_dir / "search_heuristics.csv", index=False)
    a_vs_b = astar_vs_bfs(starts, searchers.cost)
    a_vs_b.to_csv(tab_dir / "search_astar_vs_bfs.csv", index=False)

    # figures
    trail_figs = []
    for scen in sorted({r["scenario"] for r in rings}):
        ring = next(r for r in rings if r["scenario"] == scen)
        for account, t, txn in ring_start_lists[ring["ring_id"]]:
            res = searchers.run("bfs", (account, t))
            if res.found and res.path:
                trail_figs.append(trail_figure(graph, ring, [graph.edge_by_txn(txn)] + res.path,
                                               fig_dir / f"search_trail_{scen}.png"))
                break
    fig_exp = bar_figure(comp, "algorithm", "mean_expanded", "Mean nodes expanded per start", "nodes (log)",
                         fig_dir / "search_expanded.png", log=True)
    fig_scen = scenario_figure(by_scen, fig_dir / "search_success_by_scenario.png")

    headline = vs_all[vs_all.scenario == "all"].iloc[0]
    fk = vs_all[vs_all.scenario == "fake_kyc"]
    fk_line = ""
    if len(fk):
        fk = fk.iloc[0]
        fk_line = (f" For `fake_kyc`, the random forest flags {int(fk.rf_flagged)} of {int(fk.mules)} mules "
                   f"(recall {fk.rf_recall:.2f}); trails find {int(fk.trail_mules)} "
                   f"({fk.trail_recall:.2f}), {int(fk.trail_only)} of which the classifier missed.")
    bfs_row = comp[comp.algorithm == "bfs"].iloc[0]
    summary = (
        f"{P.HONEST_FRAMING}\n\nData: **{kind}** ({graph.n_accounts:,} accounts, {graph.n_edges:,} transactions, "
        f"{len(rings)} rings, {len(starts) // len(ALGO_ORDER):,} victim-payment starts). Generated by "
        f"`python -m mulehunt.trace`.\n\n"
        f"- BFS reaches the ring's own cash-out from {bfs_row.success_rate:.1%} of starts.\n"
        f"- Search vs classifier (all trails <= {P.DLS_LIMIT} hops): the random forest flags "
        f"{int(headline.rf_flagged)} of {int(headline.mules)} mules (recall {headline.rf_recall:.3f}); trails "
        f"from the victim payments find {int(headline.trail_mules)} ({headline.trail_recall:.3f}), "
        f"{int(headline.trail_only)} of them missed by the classifier, for a combined recall of "
        f"{headline.combined_recall:.3f}. The cost: {int(headline.genuine_on_trails)} genuine accounts lie on "
        f"those trails.{fk_line}")
    sections = [
        ("Summary", summary),
        ("Problem formulation", FORMULATION),
        ("The time-respecting rule", TIME_RULE),
        ("Algorithms and conventions", CONVENTIONS),
        ("Comparison over every start",
         ["`success_rate` = the ring's own cash-out was reached; `trail_precision` = share of trail accounts "
          "(after the start) that belong to the ring; times are in `b_tables/search_timing.csv`.", comp,
          "Ring level (share of rings traced from at least one start):", ring_level, fig_exp]),
        ("By scenario, evasive and cycle rings",
         [by_scen, fig_scen, "By split:", by_split, "Evasive rings (split payments under thresholds):", by_evasive,
          "Rings with a cycle among tier-2 mules:", by_cycle]),
        ("Search vs classifier",
         ["For each ring: mules (tiers 1-3) on found trails, mules the random forest flags at its validation-F1 "
          f"threshold ({threshold:.3f}), and mules found **only** by the trail. `genuine_on_trails` counts "
          "accounts outside every ring that the trails pass through: the false-alarm cost of following "
          f"money. The all-trails variant collects every account on any time-respecting trail of at most "
          f"{P.DLS_LIMIT} hops to a cash-out ({n_trunc} start(s) hit the expansion budget).", vs]),
        ("Window sensitivity",
         ["BFS success with shorter hop windows. Patient mules wait 24-47 h, so trails through them are "
          "lost at 12 h and 24 h.", window,
          f"Rings whose longest in-ring hop gap exceeds 24 h: {int(gaps.needs_over_24h.sum())} of {len(gaps)}; "
          f"exceeds 12 h: {int(gaps.needs_over_12h.sum())} (`b_tables/search_hop_gaps.csv`)."]),
        ("A* heuristics", [PROOFS, "Measured values of `m` and the layer minima `L_j`:", heur,
                           "UCS vs A*(h1) vs A*(h2) over every start (expansions and agreement of the found cost, "
                           "1e-9 tolerance):", heur_cmp.drop(columns=["mean_time_ms"]), heuristic_verdict(heur_cmp, searchers),
                           "What A* adds: on starts where both reach the own cash-out, BFS returns the fewest-hop "
                           "trail while A* returns the trail with the lowest summed `-log(risk)`, i.e. the most "
                           "suspicious one.", a_vs_b]),
        ("Limitations", f"""
- **B-D1 assumption:** the goal set is the true cash-out accounts, as if the bank had seen every
  withdrawal. In practice the goal set would be partial; the optional label-free "sink" goal
  (Task 3 Step 5) was not built in this version.
- **Budgets:** `MAX_EXPANSIONS = {P.MAX_EXPANSIONS:,}`; truncated runs are counted separately.
- **Sleepers and money collectors:** mules with normal history and benign forwarders create
  legal branches, which is why DFS wanders and why genuine accounts appear on the all-trails sets.
- **Score windows:** `risk_prob` describes each account inside its own split window. Rings are
  split-pure, but genuine accounts on a trail may come from any split.
- **Simulated data:** the rings come from Person A's simulator, not real UPI data.
"""),
        ("Parameters used", P.params_table(["GOAL_ROLE", "HOP_WINDOW_H", "WINDOW_SENSITIVITY_H", "DEFAULT_RISK",
                                            "DLS_LIMIT", "MAX_DEPTH", "MAX_EXPANSIONS"])),
    ]
    if trail_figs:
        sections.insert(3, ("Example trails", ["One BFS trail per scenario (first hop = the victim payment); "
                                              "nodes coloured by true tier from `rings.json`.", *trail_figs]))
    from mulehunt.report import write_markdown_report
    write_markdown_report(out, "Tracing stolen money: uninformed and informed search", sections)
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Person B tracing study")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=str(REPORT))
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--quick", action="store_true", help="every fifth ring only")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    run_trace_study(a.data, a.out, a.figures, a.tables, quick=a.quick)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
