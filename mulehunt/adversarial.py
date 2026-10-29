"""Fraudster-vs-detector game with minimax and alpha-beta pruning (Unit II, adversarial search).

    python -m mulehunt.adversarial [--data data/generated]

The fraudster must move X rupees over R rounds (days). Each round the detector
commits to a rule (threshold T, velocity on/off) first; the fraudster sees it and
chooses (f, n): move f x remaining this round, as n equal parts through n mules.
A part is caught if it is >= T, or if velocity is on and it is beyond the first
VELOCITY_K parts. Zero-sum payoff to the fraudster per round:

    moved - caught - n * MULE_FEE - COMMISSION * moved_uncaught + lambda * FA(T, v)

FA is the measured false-alarm rate of the rule on genuine transactions, so the
detector pays for strict rules. Writes reports/b_03_adversarial.md.
"""

import argparse
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from mulehunt import b_params as P

REPORT = Path("reports/b_03_adversarial.md")
FIGURES = Path("reports/b_figures")
TABLES = Path("reports/b_tables")


@dataclass(frozen=True)
class GameParams:
    X: float = P.GAME_X
    R: int = P.GAME_R
    thresholds: tuple = P.GAME_THRESHOLDS
    velocity_k: int = P.VELOCITY_K
    fractions: tuple = P.FRACTIONS
    parts: tuple = P.PARTS
    mule_fee: float = P.MULE_FEE
    commission: float = P.COMMISSION
    alert_cost: float = P.ALERT_COST
    fa: dict = field(default_factory=dict, hash=False, compare=False)  # (T, v) -> alerts per day


# ---------------------------------------------------------------- data

def measure_false_alarms(tx: pd.DataFrame, thresholds=P.GAME_THRESHOLDS, k=P.VELOCITY_K) -> dict:
    """Mean daily alerts on genuine (scenario == "normal") transactions for each rule.
    Velocity on adds the genuine senders making more than k payments in a day."""
    g = tx[tx.scenario == "normal"]
    day = g.timestamp.dt.tz_convert("Asia/Kolkata").dt.date
    n_days = day.nunique()
    busy = (g.groupby([g.sender_id, day]).size() > k).sum() / n_days
    out = {}
    for T in thresholds:
        big = (g.amount >= T).sum() / n_days
        out[(T, False)] = float(big)
        out[(T, True)] = float(big + busy)
    return out


# ---------------------------------------------------------------- rules

def initial_state(params) -> tuple:
    return (0, float(params.X))  # (round index, remaining money)


def is_terminal(state, params) -> bool:
    return state[0] >= params.R or state[1] <= 1e-9


def detector_moves(params) -> list[tuple]:
    return [(T, v) for T in params.thresholds for v in (False, True)]


def fraudster_moves(state, params) -> list[tuple]:
    last = state[0] == params.R - 1
    fractions = (1.0,) if last else params.fractions
    return [(f, n) for f in fractions for n in params.parts]


def round_outcome(state, d_move, f_move, params) -> dict:
    T, v = d_move
    f, n = f_move
    moved = f * state[1]
    part = moved / n
    caught_parts = n if part >= T else (max(0, n - params.velocity_k) if v else 0)
    caught = caught_parts * part
    uncaught = moved - caught
    fees = n * params.mule_fee
    commission = params.commission * uncaught
    fa_cost = params.alert_cost * params.fa.get((T, v), 0.0)
    return {"moved": moved, "caught": caught, "uncaught": uncaught, "fees": fees, "commission": commission,
            "fa_cost": fa_cost, "payoff": uncaught - fees - commission + fa_cost}


def apply(state, d_move, f_move, params) -> tuple[tuple, float]:
    o = round_outcome(state, d_move, f_move, params)
    return (state[0] + 1, state[1] - o["moved"]), o["payoff"]


def evaluate(state, params) -> float:
    """B-D12: optimistic for the fraudster: the rest moves through one mule uncaught."""
    rem = state[1]
    return rem * (1 - params.commission) - params.mule_fee if rem > 1e-9 else 0.0


# ---------------------------------------------------------------- minimax

class Stats:
    def __init__(self):
        self.nodes = 0
        self.cutoffs = 0


def _ordered_detector(state, params):
    """Most restrictive rule first: the fraudster's best immediate payoff against it, ascending."""
    moves = detector_moves(params)
    key = lambda d: max(round_outcome(state, d, f, params)["payoff"] - params.alert_cost * params.fa.get(d, 0.0)
                        for f in fraudster_moves(state, params))
    return sorted(moves, key=key)


def _ordered_fraudster(state, d, params):
    return sorted(fraudster_moves(state, params), key=lambda f: -round_outcome(state, d, f, params)["payoff"])


def _min_node(state, depth, params, ab, ordering, stats, alpha, beta):
    stats.nodes += 1
    if is_terminal(state, params):
        return 0.0, None
    if depth == 0:
        return evaluate(state, params), None
    best, best_move = math.inf, None
    for d in (_ordered_detector(state, params) if ordering else detector_moves(params)):
        val, _ = _max_node(state, d, depth - 1, params, ab, ordering, stats, alpha, beta)
        if val < best:
            best, best_move = val, d
        if ab:
            beta = min(beta, best)
            if alpha >= beta:
                stats.cutoffs += 1
                break
    return best, best_move


def _max_node(state, d, depth, params, ab, ordering, stats, alpha, beta):
    stats.nodes += 1
    if depth == 0:
        return evaluate(state, params), None
    best, best_move = -math.inf, None
    for f in (_ordered_fraudster(state, d, params) if ordering else fraudster_moves(state, params)):
        nxt, pay = apply(state, d, f, params)
        val, _ = _min_node(nxt, depth - 1, params, ab, ordering, stats, alpha - pay, beta - pay)
        val += pay
        if val > best:
            best, best_move = val, f
        if ab:
            alpha = max(alpha, best)
            if alpha >= beta:
                stats.cutoffs += 1
                break
    return best, best_move


def minimax(state, depth, params, alpha_beta=True, ordering=False, stats=None):
    """Value of a detector-to-move state (fraudster's payoff from here) and the detector's
    best rule. `depth` is in plies (one detector or fraudster move each); 2R plies = full game."""
    stats = stats if stats is not None else Stats()
    return _min_node(state, depth, params, alpha_beta, ordering, stats, -math.inf, math.inf)


def fraudster_reply(state, d_move, depth, params, alpha_beta=True, ordering=True, stats=None):
    stats = stats if stats is not None else Stats()
    return _max_node(state, d_move, depth, params, alpha_beta, ordering, stats, -math.inf, math.inf)


def full_depth(params, state=None) -> int:
    state = state or initial_state(params)
    return 2 * (params.R - state[0])


def best_response(fixed_detector_move, params) -> tuple[float, list]:
    """Fraudster's best plan when the detector always plays the same rule (exact search)."""
    def go(state):
        if is_terminal(state, params):
            return 0.0, []
        best = (-math.inf, None)
        for f in fraudster_moves(state, params):
            nxt, pay = apply(state, fixed_detector_move, f, params)
            v, plan = go(nxt)
            if v + pay > best[0]:
                best = (v + pay, [f] + plan)
        return best
    return go(initial_state(params))


# ---------------------------------------------------------------- play-outs

def minimax_detector(params, depth=None):
    return lambda state: minimax(state, depth or full_depth(params, state), params, True, True)[1]


def minimax_fraudster(params, depth=None):
    return lambda state, d: fraudster_reply(state, d, (depth or full_depth(params, state)) - 1, params)[1]


def fixed_detector(move):
    return lambda state: move


def play_out(policy_d, policy_f, params) -> list[dict]:
    state, rows = initial_state(params), []
    while not is_terminal(state, params):
        d = policy_d(state)
        f = policy_f(state, d)
        o = round_outcome(state, d, f, params)
        rows.append({"round": state[0] + 1, "remaining_before": state[1], "threshold": d[0], "velocity": d[1],
                     "fraction": f[0], "parts": f[1], **o})
        state, _ = apply(state, d, f, params)
    if state[1] > 1e-9:
        rows.append({"round": "frozen", "remaining_before": state[1], "moved": 0.0, "caught": 0.0,
                     "uncaught": 0.0, "fees": 0.0, "commission": 0.0, "fa_cost": 0.0, "payoff": 0.0})
    return rows


def accounting(rows, params) -> dict:
    df = pd.DataFrame(rows)
    left = df[df["round"] == "frozen"].remaining_before.sum() if "frozen" in set(df["round"]) else 0.0
    return {"uncaught": df.uncaught.sum(), "caught": df.caught.sum(), "left_over": left,
            "fees": df.fees.sum() + df.commission.sum(), "fa_cost": df.fa_cost.sum(), "payoff": df.payoff.sum(),
            "total": df.uncaught.sum() + df.caught.sum() + left}


# ---------------------------------------------------------------- study

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def tree_figure(params, path):
    """R = 1 game tree with 3 detector x 3 fraudster moves; pruned branches greyed out."""
    plt = _plt()
    small = replace(params, R=1, thresholds=params.thresholds, parts=(1, 4, 8))
    dmoves = [(T, False) for T in small.thresholds]
    state = initial_state(small)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    beta = math.inf
    root_val = math.inf
    xs = np.linspace(0.1, 0.9, 3)
    for i, d in enumerate(dmoves):
        alpha, best = -math.inf, -math.inf
        child_x = xs[i] + np.array([-0.1, 0, 0.1])
        ax.plot([0.5, xs[i]], [0.9, 0.55], color="black", lw=1)
        for j, f in enumerate([(1.0, n) for n in small.parts]):
            pruned = alpha >= beta
            colour = "#BBBBBB" if pruned else "black"
            ax.plot([xs[i], child_x[j]], [0.55, 0.2], color=colour, lw=1, ls="--" if pruned else "-")
            if pruned:
                ax.text(child_x[j], 0.13, "pruned", ha="center", fontsize=7, color="#999999")
                continue
            val = round_outcome(state, d, f, small)["payoff"]
            ax.text(child_x[j], 0.13, f"n={f[1]}\n{val:,.0f}", ha="center", va="top", fontsize=7)
            best = max(best, val)
            alpha = max(alpha, best)
        beta = min(beta, best)
        root_val = min(root_val, best)
        ax.text(xs[i], 0.58, f"T={d[0] // 1000}k\nmax={best:,.0f}", ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="#F2F2F2"))
    ax.text(0.5, 0.93, f"detector (MIN)\nvalue={root_val:,.0f}", ha="center", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="#DDE6F2"))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title(f"R = 1, X = Rs {small.X:,.0f}, velocity off, lambda = {small.alert_cost:g}: alpha-beta, left to right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_game_study(data_dir=None, out=REPORT, fig_dir=FIGURES, tab_dir=TABLES) -> None:
    from mulehunt.graph import TxGraph
    from mulehunt.report import write_markdown_report
    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir or "data/generated")
    kind = "real" if (data_dir / "transactions.csv").exists() else "toy"
    from simulator.schema import read_transactions
    tx = read_transactions((data_dir if kind == "real" else Path("data/generated/toy")) / "transactions.csv")
    fa = measure_false_alarms(tx)
    params = GameParams(fa=fa)
    fa_tab = pd.DataFrame([{"threshold": T, "velocity": v, "alerts_per_day": round(x, 2),
                            "cost_per_round_at_lambda": round(params.alert_cost * x, 0)} for (T, v), x in fa.items()])
    fa_tab.to_csv(tab_dir / "game_false_alarms.csv", index=False)

    # 1. fixed thresholds are beaten
    rows = []
    for d in detector_moves(params):
        val, plan = best_response(d, params)
        acc = accounting(play_out(fixed_detector(d), lambda s, _d, p=iter(plan): next(p), params), params)
        rows.append({"detector_rule": f"T={d[0]:,} v={'on' if d[1] else 'off'}", "game_value": round(val, 0),
                     "fraudster_plan (f, n) per round": " ".join(f"({f:g},{n})" for f, n in plan),
                     "uncaught": round(acc["uncaught"], 0), "caught": round(acc["caught"], 0),
                     "fees": round(acc["fees"], 0), "fa_cost": round(acc["fa_cost"], 0)})
    fixed = pd.DataFrame(rows)
    fixed.to_csv(tab_dir / "game_fixed_policies.csv", index=False)

    # 2. minimax policy
    stats = Stats()
    value, first = minimax(initial_state(params), full_depth(params), params, True, True, stats)
    mm_rows = play_out(minimax_detector(params), minimax_fraudster(params), params)
    mm = pd.DataFrame(mm_rows)
    money = ["remaining_before", "moved", "caught", "uncaught", "fees", "commission", "fa_cost", "payoff"]
    mm[money] = mm[money].round(0)
    mm.to_csv(tab_dir / "game_minimax_playout.csv", index=False)
    acc_mm = accounting(mm_rows, params)

    # 3. node counts
    node_rows, timing_rows = [], []
    for R in (2, 3, 4):
        p = replace(params, R=R)
        for label, ab, order in (("minimax", False, False), ("alpha-beta", True, False),
                                 ("alpha-beta + ordering", True, True)):
            st = Stats()
            t0 = time.perf_counter()
            v, m = minimax(initial_state(p), full_depth(p), p, ab, order, st)
            timing_rows.append({"R": R, "method": label, "seconds": round(time.perf_counter() - t0, 3)})
            node_rows.append({"R": R, "method": label, "nodes": st.nodes, "cutoffs": st.cutoffs,
                              "value": round(v, 2), "first_rule": str(m)})
    nodes = pd.DataFrame(node_rows)
    nodes.to_csv(tab_dir / "game_nodes.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(tab_dir / "game_timing.csv", index=False)

    # 4. lambda sweep
    sweep = []
    for lam in P.ALERT_COST_SWEEP:
        p = replace(params, alert_cost=lam)
        v, m = minimax(initial_state(p), full_depth(p), p, True, True)
        acc = accounting(play_out(minimax_detector(p), minimax_fraudster(p), p), p)
        rules = [f"{r['threshold'] // 1000}k/{'on' if r['velocity'] else 'off'}"
                 for r in play_out(minimax_detector(p), minimax_fraudster(p), p) if r["round"] != "frozen"]
        sweep.append({"lambda": lam, "game_value": round(v, 0), "detector_rules_by_round": " ".join(rules),
                      "uncaught": round(acc["uncaught"], 0), "caught": round(acc["caught"], 0)})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(tab_dir / "game_lambda_sweep.csv", index=False)

    # 5. deep game
    deep = []
    pd6 = replace(params, R=P.DEEP_R)
    for depth in P.DEEP_DEPTHS:
        st = Stats()
        t0 = time.perf_counter()
        v, m = minimax(initial_state(pd6), depth, pd6, True, True, st)
        deep.append({"R": P.DEEP_R, "depth_plies": depth, "value": round(v, 0), "first_rule": str(m),
                     "nodes": st.nodes, "cutoffs": st.cutoffs, "seconds": round(time.perf_counter() - t0, 2)})
    deep = pd.DataFrame(deep)
    deep[["R", "depth_plies", "seconds"]].to_csv(tab_dir / "game_deep_timing.csv", index=False)
    deep = deep.drop(columns=["seconds"])
    deep.to_csv(tab_dir / "game_deep.csv", index=False)

    # link to the data
    link = ""
    rings_path = data_dir / "_meta" / "rings.json"
    if rings_path.exists():
        import json
        rings = json.loads(rings_path.read_text())
        ev = np.mean([r["evasive"] for r in rings])
        link = f"{ev:.0%} of Person A's simulated rings are evasive (they split payments to 90-99.9% of a threshold)"
        feats = Path("data/processed/features.parquet")
        if feats.exists():
            f = pd.read_parquet(feats, columns=["below_threshold_share", "label"])
            m = f.groupby("label").below_threshold_share.mean()
            link += (f", and the `below_threshold_share` feature averages {m.get(1, float('nan')):.3f} for fraud "
                     f"accounts vs {m.get(0, float('nan')):.3f} for genuine ones")
        link += ". The game explains why: against any fixed threshold, splitting is the best response."

    # figures
    f_tree = tree_figure(replace(params, X=60_000, alert_cost=0), fig_dir / "game_tree_r1.png")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for label, grp in nodes.groupby("method", sort=False):
        ax.plot(grp.R, grp.nodes, marker="o", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("rounds R")
    ax.set_ylabel("nodes visited (log)")
    ax.set_xticks([2, 3, 4])
    ax.legend()
    ax.set_title("Game-tree nodes: minimax vs alpha-beta")
    fig.tight_layout()
    f_nodes = fig_dir / "game_nodes.png"
    fig.savefig(f_nodes, dpi=120)
    plt.close(fig)
    bars = pd.concat([fixed[["detector_rule", "uncaught", "caught"]],
                      pd.DataFrame([{"detector_rule": "minimax", "uncaught": acc_mm["uncaught"],
                                     "caught": acc_mm["caught"]}])])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(bars))
    ax.bar(x - 0.2, bars.uncaught, 0.4, label="moved uncaught", color="#C44E52")
    ax.bar(x + 0.2, bars.caught, 0.4, label="caught (frozen)", color="#4C72B0")
    ax.set_xticks(x, bars.detector_rule, rotation=30, fontsize=8)
    ax.set_ylabel("Rs")
    ax.legend()
    ax.set_title("Money moved vs caught per detector policy (fraudster plays best response)")
    fig.tight_layout()
    f_money = fig_dir / "game_money.png"
    fig.savefig(f_money, dpi=120)
    plt.close(fig)

    r3 = nodes[nodes.R == params.R].set_index("method").nodes
    saving = 1 - r3["alpha-beta + ordering"] / r3["minimax"]
    sections = [
        ("Summary", f"""{P.HONEST_FRAMING}

Data for the false-alarm rates: **{kind}**. Generated by `python -m mulehunt.adversarial`.

- Every fixed rule is undercut by splitting wherever the move set allows it: the fraudster's best response keeps each part below T. At T = 10k the first round (at least Rs 1 lakh in at most 8 parts) cannot get under, so he sends it as one part (one fee) and accepts that it is frozen.
- The minimax value of the R = {params.R} game is **Rs {value:,.0f}** for the fraudster (detector's first rule:
  T = {first[0]:,}, velocity {'on' if first[1] else 'off'}).
- Alpha-beta with move ordering visits {r3['alpha-beta + ordering']:,} nodes instead of {r3['minimax']:,}
  ({saving:.1%} fewer) and returns the same value and move."""),
        ("The game", f"""
- **Players:** detector (MIN) and fraudster (MAX); zero-sum.
- **Setup (B-D7):** move X = Rs {params.X:,.0f} over R = {params.R} rounds (days).
- **Detector move (B-D8):** threshold T in {params.thresholds} x velocity rule off/on. With velocity on,
  every part after the first {params.velocity_k} sent in a round is flagged.
- **Fraudster move (B-D9):** fraction f in {params.fractions} of the remaining money (forced to 1.0 in the last
  round) x n in {params.parts} equal parts, each through a different tier-1 mule.
- **Catching:** a part is caught if it is >= T, or if velocity is on and it is beyond the first {params.velocity_k}.
  Caught money is frozen.
- **Payoff (B-D10, B-D11):** per round, `moved - caught - n x {params.mule_fee:,} - {params.commission:.0%} x
  moved_uncaught + lambda x FA(T, v)`, with lambda = Rs {params.alert_cost} per false alert. Money not moved by
  the last round counts 0.
- **Why the FA term:** without it, the strictest rule (10k + velocity) always wins and the game is trivial.
  It does not depend on the fraudster's move, so his best response is unchanged.
- **Depth-limited evaluation (B-D12):** `remaining x (1 - commission) - fee`, optimistic for the fraudster.
- **Move ordering:** fraudster moves by immediate payoff, descending; detector rules by the fraudster's best
  immediate payoff against them, ascending (the most restrictive rule first). This is a slight variant of
  the plan's "money caught" key; it orders the same way when false-alarm costs are equal."""),
        ("Measured false alarms", ["Mean alerts per day on genuine (`scenario == normal`) transactions:", fa_tab]),
        ("1. A fixed threshold is beaten", [fixed, link, f_money]),
        ("2. The minimax policy", [f"Play-out with both sides playing minimax (value Rs {value:,.0f}):", mm,
                                   f"Accounting: uncaught {acc_mm['uncaught']:,.0f} + caught {acc_mm['caught']:,.0f}"
                                   f" + left over {acc_mm['left_over']:,.0f} = {acc_mm['total']:,.0f} (= X)."]),
        ("3. Node counts", [nodes, f_nodes, f_tree]),
        ("4. Lambda sweep", ["How the detector's minimax rule changes with the false-alarm price:", sweep]),
        ("5. Deep game", [f"R = {P.DEEP_R} with depth limits and the evaluation function (alpha-beta + ordering). "
                          "Timings are in `b_tables/game_deep_timing.csv`.", deep]),
        ("Viva caveat", """
Real detectors commit to a policy first and fraudsters adapt, which is exactly this move order;
minimax gives the detector its worst-case-safe policy. It is a simplified model: discrete
moves, equal parts, no learning, a fixed false-alarm price, and catching is all-or-nothing
per part."""),
        ("Parameters used", P.params_table(["GAME_X", "GAME_R", "DEEP_R", "DEEP_DEPTHS", "GAME_THRESHOLDS",
                                            "VELOCITY_K", "FRACTIONS", "PARTS", "MULE_FEE", "COMMISSION",
                                            "ALERT_COST", "ALERT_COST_SWEEP"])),
    ]
    write_markdown_report(out, "Fraudster vs detector: minimax and alpha-beta", sections)
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Person B evasion game study")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=str(REPORT))
    a = ap.parse_args(argv)
    run_game_study(a.data, a.out)


if __name__ == "__main__":
    main()
