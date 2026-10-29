import numpy as np
import pandas as pd
import pytest

from mulehunt.features import FEATURE_COLUMNS
from mulehunt.local_search import (ALGORITHMS, BitProblem, Cache, FeatureSelection, ThresholdTuning,
                                   genetic_algorithm, hill_climbing, local_beam, simulated_annealing)


class OneMax(BitProblem):
    def fitness(self, sol):
        return sum(sol)


class Trap(BitProblem):
    """Deceptive: fitness rises with the number of ones (local optimum = all ones, n - 1),
    but the global optimum is all zeros (n)."""

    def fitness(self, sol):
        u = sum(sol)
        return self.n if u == 0 else u - 1


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_onemax_reached(name):
    p = OneMax(20)
    best = max(ALGORITHMS[name](p, 200, np.random.default_rng(s)).best_fitness for s in (0, 1, 2))
    assert best == 20


def test_hill_climbing_stuck_on_trap_sa_ga_escape():
    p = Trap(6)
    hc = hill_climbing(p, 200, np.random.default_rng(0), restarts=0)
    assert hc.best == (1,) * 6 and hc.best_fitness == 5
    sa = simulated_annealing(p, 200, np.random.default_rng(0), t0=3.0, alpha=0.995)
    ga = genetic_algorithm(p, 200, np.random.default_rng(0), mut_rate=0.3)
    assert sa.best_fitness == 6 or ga.best_fitness == 6


@pytest.mark.parametrize("alg", [hill_climbing, simulated_annealing, local_beam, genetic_algorithm])
def test_budget_respected(alg):
    r = alg(OneMax(30), 37, np.random.default_rng(0))
    assert r.evals <= 37 and len(r.history) == r.evals
    assert all(b1 <= b2 for (_, b1), (_, b2) in zip(r.history, r.history[1:]))


def test_cache_counts_unique():
    calls = []

    class Spy(OneMax):
        def fitness(self, sol):
            calls.append(sol)
            return super().fitness(sol)

    c = Cache(Spy(4), 10)
    for s in [(0, 0, 0, 1), (0, 0, 0, 1), (1, 1, 1, 1), (0, 0, 0, 1)]:
        c(s)
    assert c.evals == 2 and len(calls) == 2 and c.best == (1, 1, 1, 1)


@pytest.fixture
def tiny_parquet(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for split, n in (("train", 300), ("val", 200)):  # no test rows at all
        y = (rng.random(n) < 0.2).astype(int)
        X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
        X[:, 0] += 2 * y
        df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        df.insert(0, "account_id", [f"ACC{split[0]}{i:05d}" for i in range(n)])
        df["label"], df["split"] = y, split
        rows.append(df)
    path = tmp_path / "features.parquet"
    pd.concat(rows).to_parquet(path)
    return path


def test_feature_selection_runs_without_test_rows(tiny_parquet):
    p = FeatureSelection(tiny_parquet)
    assert p.fitness((0,) * 20) == -1
    only_signal = (1,) + (0,) * 19
    assert p.fitness(only_signal) > p.fitness((0,) + (1,) * 19)
    r = hill_climbing(p, 30, np.random.default_rng(0), restarts=None)
    assert r.evals <= 30 and r.best_fitness > 0


def _threshold_problem(c_fn=50):
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    p = np.clip(rng.beta(2, 8, 3000) + 0.5 * y * rng.random(3000), 0, 1)
    return ThresholdTuning(p, y, c_fn=c_fn)


def test_threshold_exact_is_lower_bound_and_sa_close():
    prob = _threshold_problem()
    t_opt, c_opt = prob.exact_optimum()
    for name, alg in ALGORITHMS.items():
        r = alg(prob, 200, np.random.default_rng(0))
        assert prob.cost(r.best) >= c_opt
    sa = min(prob.cost(simulated_annealing(prob, 200, np.random.default_rng(s)).best) for s in (0, 1, 2))
    assert sa <= 1.01 * c_opt


def test_threshold_cost_monotone_in_fn_fp():
    prob = _threshold_problem()
    fn_lo, fp_hi = prob.counts(0.1)
    fn_hi, fp_lo = prob.counts(0.9)
    assert fn_lo <= fn_hi and fp_hi >= fp_lo
    a, b = ThresholdTuning([0.2, 0.8], [1, 0], c_fn=10), ThresholdTuning([0.2, 0.8], [1, 0], c_fn=20)
    assert b.cost(0.5) > a.cost(0.5)  # one FN, one FP: a higher C_FN costs more
