# Viva notes (Person B)

**Why a time-respecting state `(account, arrival_time)`?** Money can only move *after* it arrives. In a static graph, a search would follow payments made before the stolen money landed, which cannot be carrying it. Time also makes every trail strictly increasing, so tree search can't loop.

**Why is h1 admissible, and why is it weak?** `h1 = m · d(n)`: time only removes paths, so at least `d(n)` hops remain, each costing at least `m`. But the riskiest account scores 0.99985, so `m ≈ 1.5e-4` and h1 ≈ 0; A\*(h1) behaves like UCS. The layered h2 is tighter in theory but also near zero 2-3 hops from a cash-out, so measured expansions are identical to UCS.

**Then what does A\* add?** A different trail: the most suspicious one (lowest summed `−log(risk)`), not the shortest. It differs from BFS's trail on 36% of starts.

**Why tree-search DFS?** Time strictly increases along a trail, so DFS cannot loop and needs no visited set; memory stays at the stack (~8 states vs ~28 for BFS). A depth-aware visited set is optional (`graph_search=True`).

**Isn't B-D1 (known cash-outs) cheating?** It's the one documented ground-truth read: it models a bank that has seen the cash withdrawal. The search itself never reads tiers or scenarios, and a test enforces that. The weakness is real: a bank knows only some cash-outs.

**Why does a fixed threshold lose?** The fraudster sees the rule and splits under it: against ₹50k, parts of ₹50k or less and nothing is caught. That's why 30% of the simulated rings are evasive.

**Why include a false-alarm cost?** Without it, the strictest rule (₹10k + velocity) always wins. It would also flag about 443 genuine payments a day; at ₹100 per alert, minimax moves to ₹25k + velocity.

**Why use SA on a 1-D threshold?** Because the exact optimum can be computed by scanning every probability, every local-search answer can be *checked*: all land within 1.2% of it. The method then carries over to problems without an exact answer, like the 2²⁰ feature subsets.

**Why Naive Bayes on character n-grams?** Hinglish has no fixed spelling ("aapka/apka/apke"); char n-grams share pieces, so the variants support each other. NB has few parameters, which suits a few hundred messages; independence is false but mostly affects calibration, not ranking.

**Why is accuracy misleading?** Only 1.2% of accounts are fraud, so "nobody is fraud" scores 98.8% accuracy. We use PR-AUC, recall and cost (C_FN · FN + FP) instead.

**Why is the simulator credible?** Its parameters are sourced from NPCI/RBI (daily volumes, P2P/P2M mix, amount distribution) and tagged sourced/assumed in `config.yaml`. Its patterns (fan-in, fan-out, scatter-gather, cycles) are checked against the IBM AML benchmark. It includes hard cases: sleeper mules, evasive splitting, benign money collectors, shared shop phones. It also makes the data reproducible byte for byte. It is still simulated, and we say so everywhere.

## Person A's modules, in my words

- **Simulator (`simulator/`)**: generates 50,377 accounts of five types with realistic daily UPI behaviour over 90 days, then injects 75 rings in five scam scenarios. Each ring is a fraudster, 2-4 tier-1, 1-3 tier-2 and 0-2 tier-3 mules, and a cash-out. Hops have a lognormal delay (median 2 h, capped at 48 h), 30% of mules are patient, 50% are sleepers, and 30% of rings split payments under thresholds. 1.5% label noise on `is_fraud`. Validated against NPCI targets and IBM AML patterns.
- **Features (`mulehunt/features.py`)**: 20 per-account features in four groups (flow, network, behavioural, device/location), each computed over a 30-day window per split (train days 31-60, val 46-75, test 61-90). Splits are account-disjoint and keep rings together.
- **Models (`mulehunt/models.py`)**: decision tree (depth curve and cost-complexity pruning), KNN, Naive Bayes, logistic regression, bagging, random forest, AdaBoost and gradient boosting, all with `class_weight="balanced"` or resampling. The random forest is best: test PR-AUC 0.915, precision 0.962, recall 0.753. Its weak spot is `fake_kyc`, at recall 0.37. It writes the `risk_scores.csv` my search uses as edge cost, with out-of-fold scores for train.
- **Clustering (`mulehunt/clustering.py`)**: k-means and bisecting k-means on flow/network features with silhouette, purity and entropy. It finds clusters with high mule lift, compared against a shuffled baseline.
