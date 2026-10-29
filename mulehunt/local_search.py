"""Local search (Unit II): hill climbing, simulated annealing, local beam and a genetic
algorithm, applied to feature selection (B-D13) and cost-sensitive threshold tuning (B-D15).

    python -m mulehunt.local_search

Budgets count **unique** fitness evaluations (a memoising cache). Everything is tuned
on the validation split; each study reports one number on the test split, once.
Writes reports/b_04_local_search.md.
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mulehunt import b_params as P

REPORT = Path("reports/b_04_local_search.md")
FIGURES = Path("reports/b_figures")
TABLES = Path("reports/b_tables")
FEATURES = Path("data/processed/features.parquet")
MAX_STALL = 50  # cache hits in a row before an algorithm gives up (all neighbours already seen)


# ---------------------------------------------------------------- framework

class Problem:
    """Maximise `fitness`. Subclasses define the representation and operators."""

    def random_solution(self, rng):
        raise NotImplementedError

    def random_neighbour(self, sol, rng):
        raise NotImplementedError

    def neighbours(self, sol) -> list:
        raise NotImplementedError

    def fitness(self, sol) -> float:
        raise NotImplementedError

    def crossover(self, a, b, rng):
        raise NotImplementedError

    def mutate(self, sol, rng):
        raise NotImplementedError


class BudgetExhausted(Exception):
    pass


class Cache:
    """Memoised fitness; counts unique evaluations and records best-so-far after each."""

    def __init__(self, problem: Problem, budget: int):
        self.problem, self.budget = problem, budget
        self.values: dict = {}
        self.history: list[tuple[int, float]] = []
        self.best, self.best_fitness = None, -math.inf
        self.stall = 0

    @property
    def evals(self) -> int:
        return len(self.values)

    def __call__(self, sol) -> float:
        if sol in self.values:
            self.stall += 1
            if self.stall > MAX_STALL * max(self.budget, 1):
                raise BudgetExhausted
            return self.values[sol]
        if self.evals >= self.budget:
            raise BudgetExhausted
        self.stall = 0
        f = float(self.problem.fitness(sol))
        self.values[sol] = f
        if f > self.best_fitness:
            self.best, self.best_fitness = sol, f
        self.history.append((self.evals, self.best_fitness))
        return f


@dataclass
class RunResult:
    best: object
    best_fitness: float
    history: list
    evals: int
    extra: dict | None = None


def _result(cache, extra=None):
    return RunResult(cache.best, cache.best_fitness, cache.history, cache.evals, extra)


def hill_climbing(problem, budget, rng, restarts=0) -> RunResult:
    """Steepest ascent; on a local optimum, restart from a random solution (`restarts`
    times, or until the budget is used when restarts is None)."""
    cache = Cache(problem, budget)
    done = 0
    try:
        while True:
            cur = problem.random_solution(rng)
            f = cache(cur)
            while True:
                best_n, best_f = None, f
                for n in problem.neighbours(cur):
                    fn = cache(n)
                    if fn > best_f:
                        best_n, best_f = n, fn
                if best_n is None:
                    break
                cur, f = best_n, best_f
            if restarts is not None and done >= restarts:
                break
            done += 1
    except BudgetExhausted:
        pass
    return _result(cache, {"restarts": done})


def simulated_annealing(problem, budget, rng, t0=P.SA_T0, alpha=P.SA_ALPHA) -> RunResult:
    cache = Cache(problem, budget)
    trace = []  # (step, temperature, accepted_worse)
    try:
        cur = problem.random_solution(rng)
        f = cache(cur)
        T, step, worse = t0, 0, 0
        while True:
            n = problem.random_neighbour(cur, rng)
            fn = cache(n)
            step += 1
            delta = fn - f
            if delta > 0 or (T > 0 and rng.random() < math.exp(delta / T)):
                worse += delta < 0
                cur, f = n, fn
            trace.append((step, T, worse))
            T *= alpha
    except BudgetExhausted:
        pass
    return _result(cache, {"trace": trace})


def local_beam(problem, budget, rng, k=P.BEAM_K) -> RunResult:
    """Stochastic local beam: each of the k states draws |neighbourhood| / k random
    successors, so one beam step costs about one hill-climbing step; the best k of the
    pool survive."""
    cache = Cache(problem, budget)
    try:
        beam = [problem.random_solution(rng) for _ in range(k)]
        for s in beam:
            cache(s)
        per_state = max(1, math.ceil(len(problem.neighbours(beam[0])) / k))
        stale = 0
        while True:
            pool = set(beam)
            for s in beam:
                pool.update(problem.random_neighbour(s, rng) for _ in range(per_state))
            new = sorted(pool, key=lambda s: (-cache(s), str(s)))[:k]
            stale = stale + 1 if set(new) == set(beam) else 0
            beam = new
            if stale >= 3 * k:  # the beam has stopped improving: restart it
                beam, stale = [problem.random_solution(rng) for _ in range(k)], 0
                for s in beam:
                    cache(s)
    except BudgetExhausted:
        pass
    return _result(cache)


def genetic_algorithm(problem, budget, rng, pop=P.GA_POP, tournament=P.GA_TOURNAMENT, cx_rate=P.GA_CX_RATE,
                      mut_rate=P.GA_MUT_RATE) -> RunResult:
    cache = Cache(problem, budget)

    def pick(population):
        idx = rng.choice(len(population), size=tournament, replace=False)
        return max((population[i] for i in idx), key=cache)

    try:
        population = [problem.random_solution(rng) for _ in range(pop)]
        for s in population:
            cache(s)
        while True:
            elite = max(population, key=cache)
            children = [elite]
            while len(children) < pop:
                a, b = pick(population), pick(population)
                child = problem.crossover(a, b, rng) if rng.random() < cx_rate else a
                child = problem.mutate(child, rng, mut_rate)
                cache(child)
                children.append(child)
            population = children
    except BudgetExhausted:
        pass
    return _result(cache)


ALGORITHMS = {
    "hill_climbing": lambda p, b, rng: hill_climbing(p, b, rng, restarts=None),
    "simulated_annealing": lambda p, b, rng: simulated_annealing(p, b, rng),
    "local_beam": lambda p, b, rng: local_beam(p, b, rng),
    "genetic_algorithm": lambda p, b, rng: genetic_algorithm(p, b, rng),
}


# ---------------------------------------------------------------- bitstring problems

class BitProblem(Problem):
    def __init__(self, n_bits: int):
        self.n = n_bits

    def random_solution(self, rng):
        return tuple(int(b) for b in rng.integers(0, 2, self.n))

    def neighbours(self, sol):
        return [sol[:i] + (1 - sol[i],) + sol[i + 1:] for i in range(self.n)]

    def random_neighbour(self, sol, rng):
        i = int(rng.integers(self.n))
        return sol[:i] + (1 - sol[i],) + sol[i + 1:]

    def crossover(self, a, b, rng):
        cut = int(rng.integers(1, self.n))
        return a[:cut] + b[cut:]

    def mutate(self, sol, rng, rate=None):
        rate = 1 / self.n if rate is None else rate
        flips = rng.random(self.n) < rate
        return tuple(1 - x if f else x for x, f in zip(sol, flips))


class FeatureSelection(BitProblem):
    """B-D13: validation PR-AUC of the baseline LR on the selected columns, minus a
    per-feature penalty; the empty set scores -1. Reads only train and val rows."""

    def __init__(self, path=FEATURES, penalty=P.FEATURE_PENALTY):
        from mulehunt.features import FEATURE_COLUMNS
        table = pd.read_parquet(path)
        table = table[table.split.isin(["train", "val"])]  # test rows are dropped on load
        self.columns = list(FEATURE_COLUMNS)
        super().__init__(len(self.columns))
        tr, va = table[table.split == "train"], table[table.split == "val"]
        self.Xtr, self.ytr = tr[self.columns], tr.label.astype(int)
        self.Xva, self.yva = va[self.columns], va.label.astype(int)
        self.penalty = penalty

    def selected(self, sol) -> list[str]:
        return [c for c, b in zip(self.columns, sol) if b]

    def val_pr_auc(self, sol) -> float:
        from sklearn.metrics import average_precision_score
        from mulehunt.models import make_baseline_lr
        cols = self.selected(sol)
        model = make_baseline_lr().fit(self.Xtr[cols], self.ytr)
        return float(average_precision_score(self.yva, model.predict_proba(self.Xva[cols])[:, 1]))

    def fitness(self, sol) -> float:
        if not any(sol):
            return -1.0
        return self.val_pr_auc(sol) - self.penalty * sum(sol)


# ---------------------------------------------------------------- threshold tuning

class ThresholdTuning(Problem):
    """B-D15: choose t in [0, 1]; cost = C_FN * FN + C_FP * FP on validation probabilities.
    Fitness = -cost / (C_FN * positives), so it lies in [-1, 0]-ish whatever the ratio.
    Sorted probabilities make one evaluation two binary searches."""

    def __init__(self, p, y, c_fn=P.C_FN_HEADLINE, c_fp=P.C_FP, step_sd=P.THRESHOLD_STEP_SD):
        p, y = np.asarray(p, float), np.asarray(y, int)
        self.pos, self.neg = np.sort(p[y == 1]), np.sort(p[y == 0])
        self.c_fn, self.c_fp, self.sd = c_fn, c_fp, step_sd
        self.scale = c_fn * max(len(self.pos), 1)

    def counts(self, t) -> tuple[int, int]:
        fn = int(np.searchsorted(self.pos, t, "left"))  # positives with p < t
        fp = int(len(self.neg) - np.searchsorted(self.neg, t, "left"))  # negatives with p >= t
        return fn, fp

    def cost(self, t) -> float:
        fn, fp = self.counts(t)
        return self.c_fn * fn + self.c_fp * fp

    def fitness(self, t) -> float:
        return -self.cost(t) / self.scale

    def exact_optimum(self) -> tuple[float, float]:
        cands = np.unique(np.concatenate([self.pos, self.neg, [0.0, 1.0 + 1e-9]]))
        costs = [self.cost(t) for t in cands]
        i = int(np.argmin(costs))
        return float(cands[i]), float(costs[i])

    def _clip(self, t):
        return round(float(min(1.0, max(0.0, t))), 6)

    def random_solution(self, rng):
        return self._clip(rng.random())

    def random_neighbour(self, t, rng):
        return self._clip(t + rng.normal(0, self.sd))

    def neighbours(self, t):
        return sorted({self._clip(t + d) for d in (-0.1, -0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05, 0.1)}
                      - {t})

    def crossover(self, a, b, rng):  # blend (BLX-0.25)
        lo, hi = min(a, b), max(a, b)
        span = hi - lo
        return self._clip(rng.uniform(lo - 0.25 * span, hi + 0.25 * span))

    def mutate(self, t, rng, rate=None):
        return self._clip(t + rng.normal(0, self.sd)) if rng.random() < 0.5 else t


# ---------------------------------------------------------------- studies

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def evals_to_reach(history, frac=0.99) -> int:
    final = history[-1][1]
    target = final - (1 - frac) * abs(final)
    return next(e for e, b in history if b >= target)


def convergence_figure(histories: dict, budget, path, ylabel):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    grid = np.arange(1, budget + 1)
    for name, runs in histories.items():
        curves = []
        for h in runs:
            e = np.array([x for x, _ in h])
            b = np.array([y for _, y in h])
            idx = np.searchsorted(e, grid, "right") - 1
            curves.append(np.where(idx >= 0, b[np.clip(idx, 0, None)], np.nan))
        c = np.array(curves)
        ax.plot(grid, np.nanmean(c, 0), label=name)
        ax.fill_between(grid, np.nanmin(c, 0), np.nanmax(c, 0), alpha=0.2)
    ax.set_xlabel("unique fitness evaluations")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title("Best-so-far (mean, band = range over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_feature_selection_study(path=FEATURES, fig_dir=FIGURES, tab_dir=TABLES):
    prob = FeatureSelection(path)
    t0 = time.perf_counter()
    prob.fitness(tuple([1] * prob.n))
    one_fit = time.perf_counter() - t0
    rows, hist, timing = [], {}, []
    for name, alg in ALGORITHMS.items():
        for seed in P.SEEDS:
            t0 = time.perf_counter()
            r = alg(prob, P.EVAL_BUDGET, np.random.default_rng(seed))
            timing.append({"algorithm": name, "seed": seed, "seconds": round(time.perf_counter() - t0, 2)})
            hist.setdefault(name, []).append(r.history)
            rows.append({"algorithm": name, "seed": seed, "best_fitness": round(r.best_fitness, 5),
                         "val_pr_auc": round(prob.val_pr_auc(r.best), 5), "n_features": sum(r.best),
                         "evals": r.evals, "evals_to_99pct": evals_to_reach(r.history),
                         "features": " ".join(prob.selected(r.best)), "_best": r.best})
    runs = pd.DataFrame(rows)
    allf = tuple([1] * prob.n)
    baseline = pd.DataFrame([{"subset": "all 20 features", "n_features": 20,
                              "val_pr_auc": round(prob.val_pr_auc(allf), 5),
                              "fitness": round(prob.fitness(allf), 5)}])
    best_row = runs.sort_values(["best_fitness", "n_features", "algorithm", "seed"],
                                ascending=[False, True, True, True]).iloc[0]
    # the one test number: best subset, refit on train
    from sklearn.metrics import average_precision_score
    from mulehunt.models import make_baseline_lr
    table = pd.read_parquet(path)
    tr, te = table[table.split == "train"], table[table.split == "test"]
    cols = prob.selected(best_row["_best"])
    test_auc = float(average_precision_score(te.label, make_baseline_lr().fit(tr[cols], tr.label)
                                             .predict_proba(te[cols])[:, 1]))
    allc = prob.columns  # the same model on all 20 features, so the subset's test number has a reference
    test_auc_all = float(average_precision_score(te.label, make_baseline_lr().fit(tr[allc], tr.label)
                                                 .predict_proba(te[allc])[:, 1]))
    n_val_rings = None
    meta = Path("data/generated/_meta/rings.json")
    if meta.exists():
        import json
        n_val_rings = sum(r["split"] == "val" for r in json.loads(meta.read_text()))
    pd.DataFrame([{"subset": f"best subset ({sum(best_row['_best'])} features)", "test_pr_auc": round(test_auc, 4)},
                  {"subset": "all 20 features", "test_pr_auc": round(test_auc_all, 4)}]).to_csv(
        tab_dir / "local_fs_test.csv", index=False)
    summary = runs.groupby("algorithm", sort=False).agg(
        mean_best_fitness=("best_fitness", "mean"), max_best_fitness=("best_fitness", "max"),
        mean_val_pr_auc=("val_pr_auc", "mean"), mean_features=("n_features", "mean"),
        mean_evals_to_99pct=("evals_to_99pct", "mean")).round(5).reset_index()
    runs = runs.drop(columns=["_best"])
    runs.to_csv(tab_dir / "local_fs_runs.csv", index=False)
    summary.to_csv(tab_dir / "local_fs_summary.csv", index=False)
    timing.insert(0, {"algorithm": "one baseline LR fit (all features)", "seed": "-", "seconds": round(one_fit, 3)})
    pd.DataFrame(timing).to_csv(tab_dir / "local_fs_timing.csv", index=False)
    fig = convergence_figure(hist, P.EVAL_BUDGET, fig_dir / "local_fs_convergence.png", "fitness (val PR-AUC - penalty)")
    ablation = Path("reports/a_tables/ablation.csv")
    ab = pd.read_csv(ablation) if ablation.exists() else None
    return {"runs": runs, "summary": summary, "baseline": baseline, "best": best_row, "test_pr_auc": test_auc,
            "test_pr_auc_all": test_auc_all, "n_val_rings": n_val_rings, "figure": fig, "ablation": ab, "one_fit_s": one_fit}


def _rf_probs(path=FEATURES):
    from mulehunt.models import load_model
    bundle = load_model("random_forest")
    table = pd.read_parquet(path)
    out = {}
    for split in ("val", "test"):
        part = table[table.split == split]
        out[split] = (bundle["pipeline"].predict_proba(part[bundle["feature_columns"]])[:, 1], part.label.to_numpy())
    return out, float(bundle["threshold"])


def run_threshold_study(path=FEATURES, fig_dir=FIGURES, tab_dir=TABLES):
    probs, f1_threshold = _rf_probs(path)
    pv, yv = probs["val"]
    rows, sa_trace = [], None
    for c_fn in P.C_FN_SWEEP:
        prob = ThresholdTuning(pv, yv, c_fn=c_fn)
        t_opt, c_opt = prob.exact_optimum()
        rows.append({"C_FN": c_fn, "algorithm": "exact scan", "seed": "-", "threshold": round(t_opt, 4),
                     "val_cost": c_opt, "gap_vs_exact": 0.0})
        for name, alg in ALGORITHMS.items():
            for seed in P.SEEDS:
                r = alg(prob, P.EVAL_BUDGET, np.random.default_rng(seed))
                cost = prob.cost(r.best)
                rows.append({"C_FN": c_fn, "algorithm": name, "seed": seed, "threshold": round(r.best, 4),
                             "val_cost": cost, "gap_vs_exact": round((cost - c_opt) / max(c_opt, 1), 4)})
                if name == "simulated_annealing" and c_fn == P.C_FN_HEADLINE and seed == P.SEEDS[0]:
                    sa_trace = r.extra["trace"]
    runs = pd.DataFrame(rows)
    runs.to_csv(tab_dir / "local_threshold_runs.csv", index=False)
    summary = runs[runs.algorithm != "exact scan"].groupby(["C_FN", "algorithm"], sort=False).agg(
        mean_threshold=("threshold", "mean"), mean_val_cost=("val_cost", "mean"),
        worst_gap_vs_exact=("gap_vs_exact", "max")).round(4).reset_index()
    exact = runs[runs.algorithm == "exact scan"][["C_FN", "threshold", "val_cost"]].rename(
        columns={"threshold": "exact_threshold", "val_cost": "exact_val_cost"})
    summary = summary.merge(exact, on="C_FN")
    summary.to_csv(tab_dir / "local_threshold_summary.csv", index=False)

    # SA temperature / accepted-worse curve
    plt = _plt()
    tr = np.array(sa_trace)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(tr[:, 0], tr[:, 1], color="#C44E52", label="temperature")
    ax.set_xlabel("SA step")
    ax.set_ylabel("temperature", color="#C44E52")
    ax2 = ax.twinx()
    ax2.plot(tr[:, 0], tr[:, 2], color="#4C72B0", label="accepted worse moves (cumulative)")
    ax2.set_ylabel("accepted worse moves", color="#4C72B0")
    ax.set_title(f"Simulated annealing on the threshold (C_FN = {P.C_FN_HEADLINE}, seed {P.SEEDS[0]})")
    fig.tight_layout()
    f_sa = fig_dir / "local_sa_threshold.png"
    fig.savefig(f_sa, dpi=120)
    plt.close(fig)

    # the one test evaluation at the headline ratio
    head = ThresholdTuning(pv, yv, c_fn=P.C_FN_HEADLINE)
    t_head = head.exact_optimum()[0]
    sa_head = runs[(runs.C_FN == P.C_FN_HEADLINE) & (runs.algorithm == "simulated_annealing")]
    t_sa = float(sa_head.sort_values(["val_cost", "seed"]).iloc[0].threshold)
    pt, yt = probs["test"]
    test_prob = ThresholdTuning(pt, yt, c_fn=P.C_FN_HEADLINE)
    from mulehunt.models import evaluate
    test_rows = []
    for label, t in (("SA threshold (headline)", t_sa), ("exact val optimum", t_head),
                     ("Person A's F1 threshold", f1_threshold)):
        m = evaluate(yt, pt, t)
        fn, fp = test_prob.counts(t)
        test_rows.append({"threshold_source": label, "threshold": round(t, 4), "test_cost": test_prob.cost(t),
                          "FN": fn, "FP": fp, "genuine_flagged_share": round(fp / max(len(test_prob.neg), 1), 4),
                          "precision": round(m["precision"], 4), "recall": round(m["recall"], 4)})
    test = pd.DataFrame(test_rows)
    test.to_csv(tab_dir / "local_threshold_test.csv", index=False)
    return {"runs": runs, "summary": summary, "test": test, "figure": f_sa, "f1_threshold": f1_threshold,
            "n_genuine_test": len(test_prob.neg)}


def _fs_verdict(fs) -> str:
    if fs["test_pr_auc"] >= fs["test_pr_auc_all"]:
        return "The validation gain carried over to test."
    rings = f" from only {fs['n_val_rings']} validation rings" if fs["n_val_rings"] else ""
    return ("**The validation gain did not carry over to test:** the subset is selected on validation" + rings
            + ", so part of its advantage there is fitting that split. A normal result; the smaller subset "
            "is simpler but not better.")


def run_local_search_study(path=FEATURES, out=REPORT, fig_dir=FIGURES, tab_dir=TABLES) -> None:
    from mulehunt.report import write_markdown_report
    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    fs = run_feature_selection_study(path, fig_dir, tab_dir)
    th = run_threshold_study(path, fig_dir, tab_dir)
    b = fs["best"]
    head = th["test"].iloc[0]
    a_row = th["test"].iloc[2]
    sections = [
        ("Summary", f"""{P.HONEST_FRAMING}

Generated by `python -m mulehunt.local_search`. Tuning uses the validation split only; the test split is
used once, after tuning, with a reference model beside each tuned result.

- **Feature selection:** the best subset ({b.algorithm}, seed {b.seed}) keeps {b.n_features} of 20 features with
  validation PR-AUC {b.val_pr_auc:.4f} (all 20 features: {fs['baseline'].val_pr_auc.iloc[0]:.4f}). Refit on train,
  its test PR-AUC is **{fs['test_pr_auc']:.4f}**, against **{fs['test_pr_auc_all']:.4f}** for the same model on all 20
  features. {_fs_verdict(fs)}
- **Threshold tuning (C_FN = {P.C_FN_HEADLINE}):** simulated annealing picks t = {head.threshold:.4f}. On test it reaches
  recall {head.recall:.3f} by flagging **{head.FP} genuine accounts** ({head.genuine_flagged_share:.1%} of
  {th['n_genuine_test']:,}), for cost {head.test_cost:,.0f}. Person A's F1 threshold {a_row.threshold:.3f} flags
  {a_row.FP} genuine accounts at recall {a_row.recall:.3f}, cost {a_row.test_cost:,.0f}. The cost comparison favours the
  tuned threshold by construction (it minimises that cost); whether {head.genuine_flagged_share:.1%} false alarms is acceptable is a policy choice.
  Person A's chapter uses a 1% false-alarm limit, which the tuned threshold exceeds."""),
        ("Algorithms", f"""
All four share one interface (`Problem`: random solution, neighbours, fitness, crossover, mutation)
and one budget of `EVAL_BUDGET = {P.EVAL_BUDGET}` **unique** fitness evaluations, counted by a memoising cache.

- **Hill climbing:** steepest ascent over all neighbours; on a local optimum it restarts from a random
  solution until the budget is spent (random-restart hill climbing).
- **Simulated annealing:** random neighbour; a worse move is accepted with probability `exp(delta / T)`;
  `T` starts at {P.SA_T0} and is multiplied by {P.SA_ALPHA} per step.
- **Local beam (stochastic):** k = {P.BEAM_K} states; each draws |neighbourhood| / k random successors, the
  pool is ranked and the best k kept; restarts when the beam stops improving. The full-neighbourhood
  version costs k x 20 evaluations per step and cannot climb within the budget.
- **Genetic algorithm:** population {P.GA_POP}, tournament size {P.GA_TOURNAMENT}, crossover rate {P.GA_CX_RATE}
  (one-point for bitstrings, blend BLX-0.25 for thresholds), mutation {P.GA_MUT_RATE:.2f} per bit (Gaussian for
  thresholds), one elite kept."""),
        ("Feature selection", [f"""
Solution: a 20-bit string in `FEATURE_COLUMNS` order. Fitness (B-D13): validation PR-AUC of
`mulehunt.models.make_baseline_lr()` fitted on train with the selected columns, minus
{P.FEATURE_PENALTY} per feature; the empty set scores -1. Fit and run times are in `b_tables/local_fs_timing.csv`.""",
                               "Per algorithm (over seeds):", fs["summary"], "Every run:", fs["runs"],
                               "Baseline:", fs["baseline"], fs["figure"]] +
         (["Person A's per-group ablation (random forest), for comparison:", fs["ablation"]]
          if fs["ablation"] is not None else []) +
         [f"Test split: best subset `{b.features}` refit on train gives test PR-AUC **{fs['test_pr_auc']:.4f}**; the same "
          f"model on all 20 features gives **{fs['test_pr_auc_all']:.4f}**. {_fs_verdict(fs)}"]),
        ("Threshold tuning", [f"""
Solution: a threshold t in [0, 1] on the random forest's validation probabilities. Cost (B-D15) =
`C_FN x FN + C_FP x FP` with C_FP = {P.C_FP}; fitness = -cost / (C_FN x positives). Neighbour:
`clip(t + N(0, {P.THRESHOLD_STEP_SD}))`.

**Why a 1-D problem is still a fair demonstration:** the exact answer can be found by scanning every
distinct probability as a candidate threshold, so each local-search method can be *checked*
against the true optimum (`gap_vs_exact` = relative extra cost).""", th["summary"], th["figure"],
                              "Test split, once (headline ratio). `genuine_flagged_share` = FP / genuine test "
                              "accounts; read it next to the cost, because the cost is what the tuned threshold "
                              "was built to minimise:", th["test"]]),
        ("Parameters used", P.params_table(["FEATURE_PENALTY", "EVAL_BUDGET", "SEEDS", "BEAM_K", "GA_POP",
                                            "GA_TOURNAMENT", "GA_CX_RATE", "GA_MUT_RATE", "SA_T0", "SA_ALPHA",
                                            "C_FP", "C_FN_SWEEP", "C_FN_HEADLINE", "THRESHOLD_STEP_SD"])),
    ]
    write_markdown_report(out, "Local search: feature selection and threshold tuning", sections)
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Person B local search study")
    ap.add_argument("--features", default=str(FEATURES))
    ap.add_argument("--out", default=str(REPORT))
    a = ap.parse_args(argv)
    run_local_search_study(a.features, a.out)


if __name__ == "__main__":
    main()
