# Chapter 6. Adversarial search and local search

> Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark.

Full reports: [`b_03_adversarial.md`](../b_03_adversarial.md) (`python -m mulehunt.adversarial`) and [`b_04_local_search.md`](../b_04_local_search.md) (`python -m mulehunt.local_search`). Numbers come from `reports/b_tables/game_*.csv` and `local_*.csv`.

## 6.1 The fraudster-vs-detector game

The fraudster must move **X = ₹2,00,000** over **R = 3** days (B-D7). Each day:

1. The **detector (MIN)** commits to a rule: threshold `T ∈ {₹10k, ₹25k, ₹50k}` (the same thresholds Person A's evasive rings split under) × velocity rule off/on, which flags every part after the third in a day (B-D8).
2. The **fraudster (MAX)** sees the rule and picks `(f, n)`: move `f ∈ {0.5, 1}` of what remains (all of it on the last day) as `n ∈ {1, 2, 4, 8}` equal parts, one mule each (B-D9).

A part is caught if it is ≥ T, or if velocity is on and it is beyond the third. Payoff to the fraudster per day: `uncaught − n × ₹1,500 − 5% × uncaught + λ × FA(T, v)` (B-D10, B-D11). `FA` is the **measured** daily number of genuine transactions the rule would flag, from Person A's data:

| Rule | 10k | 10k + velocity | 25k | 25k + velocity | 50k | 50k + velocity |
|---|---|---|---|---|---|---|
| Alerts/day | 370.1 | 442.6 | 32.2 | 104.7 | 3.7 | 76.2 |

With λ = ₹100 per alert, the strictest rule costs ₹44,261 a day in false alarms. Without that cost the detector would always choose it and the game would be trivial.

## 6.2 Results

**A fixed threshold is undercut by splitting** (`game_fixed_policies.csv`). Against ₹50k the fraudster sends 4, 2, 2 parts (each ≤ ₹50k) and nothing is caught; against ₹25k he sends 8, 4, 4 parts and nothing is caught. At ₹10k the first day cannot get under (at least ₹1 lakh in at most 8 parts), so he sends it as one part, loses it, and splits the rest. This is exactly the behaviour Person A built into the 30% of rings that are evasive.

**Minimax.** The game value is **₹1,14,272** for the fraudster. The detector's minimax rule is ₹25k + velocity on every day; the fraudster replies 8, 4, 4 parts, losing ₹87,500 to velocity and moving ₹1,12,500. Accounting closes in every play-out: uncaught + caught + left over = X (a test enforces it).

**λ sweep** (`game_lambda_sweep.csv`): at λ = 0 and 25 the detector uses ₹10k (+ velocity after day 1), at λ = 100 it uses ₹25k + velocity, and at λ = 400 it relaxes to ₹50k without velocity after day 1. The price of a false alarm decides how strict a bank can afford to be.

**Alpha-beta** returns the same value and move as plain minimax and visits far fewer nodes (`game_nodes.csv`):

| R | Minimax | Alpha-beta | Alpha-beta + ordering |
|---|---|---|---|
| 2 | 775 | 287 | 129 |
| 3 | 18,631 | 5,675 | 691 (96.3% fewer) |
| 4 | 447,175 | 48,996 | 5,606 |

Ordering puts the fraudster's best immediate move first and the detector's most restrictive rule first, so cutoffs happen early. For R = 6 (3.7 × 10⁹ leaves), depth-limited search with the optimistic evaluation `remaining × 0.95 − ₹1,500` (B-D12) gives values ₹1,20,181 / ₹1,19,054 / ₹1,19,428 at 4 / 6 / 8 plies, with at most 8,040 nodes.

**Viva caveat.** Real detectors do commit to a policy first, which is this move order, and minimax gives the detector its worst-case-safe policy. But the model is simplified: discrete moves, equal parts, no learning, a fixed alert price, and all-or-nothing catching.

## 6.3 Local search

One framework (`Problem`, a memoising cache that counts **unique** evaluations, budget 200 per run, seeds 0-2) runs four algorithms: random-restart steepest-ascent hill climbing, simulated annealing (T₀ = 0.05, ×0.97 per step), stochastic local beam (k = 4) and a genetic algorithm (population 20, tournament 3, crossover 0.9, mutation 1/20 per bit). On a deceptive "trap" function, plain hill climbing gets stuck on the local optimum while SA and the GA escape (a unit test).

**Feature selection** (B-D13): a 20-bit mask; fitness = validation PR-AUC of the baseline logistic regression on the selected features − 0.002 per feature. One fit takes 0.07 s.

| Algorithm | Mean best fitness | Mean val PR-AUC | Mean features | Evals to 99% of best |
|---|---|---|---|---|
| Hill climbing | 0.7519 | 0.7739 | 11.0 | 106 |
| Simulated annealing | **0.7562** | **0.7782** | 11.0 | **79** |
| Local beam | 0.7534 | 0.7760 | 11.3 | 118 |
| Genetic algorithm | 0.7518 | 0.7765 | 12.3 | 136 |

All 20 features give val PR-AUC 0.7560; the best subset keeps 11 (`total_inflow, total_outflow, pass_through_ratio, in_degree, out_degree, new_counterparty_share, fan_out_max_24h, neighbor_suspicious_share, txn_count, night_txn_share, below_threshold_share`) and scores 0.7784. All 12 runs keep the same core six (`pass_through_ratio, in_degree, out_degree, new_counterparty_share, fan_out_max_24h, txn_count`); `device_count` and `counterparty_state_mismatch_share` are never chosen. Four of the core six are network features, which agrees with Person A's ablation: adding the network group lifts the random forest's test PR-AUC from 0.835 to 0.921. **Test PR-AUC of that subset, refit on train, reported once: 0.8159.**

**Threshold tuning** (B-D15): choose the random forest's flag threshold to minimise `C_FN × FN + FP` on validation. The exact optimum is found by scanning every distinct probability, so each local-search result can be *checked*; that is why a 1-D problem is still a fair demonstration. Every algorithm lands within 1.2% of the exact cost at all three ratios. At the headline C_FN = 50, SA picks t = 0.123. On test (once):

| Threshold | Test cost | FN | FP | Precision | Recall |
|---|---|---|---|---|---|
| SA, C_FN = 50 (0.123) | **617** | 8 | 217 | 0.427 | 0.953 |
| Person A's F1 threshold (0.484) | 2,105 | 42 | 5 | 0.962 | 0.753 |

If a missed mule costs 50 times a false alarm, the bank should flag far more accounts: recall rises from 0.75 to 0.95 at the price of 212 more reviews.
