# MuleHunt: Two-Person Work Plan

Oct 27, 2026 · @Interns

## The split

Each person owns separate syllabus units and separate folders, and you connect only through a few fixed files agreed on day 1. The plan assumes about 6 weeks.

|  | Person A: Data & Learning | Person B: Search & Systems |
| --- | --- | --- |
| Owns | Simulator, dataset validation, feature engineering, all fraud models, clustering | Scam-message dataset and classifier, transaction graph, all search algorithms, minimax game, local search, dashboard |
| Syllabus units | III, IV, V | I, II (plus Naive Bayes for text) |
| Folders | `simulator/`, `mulehunt/features.py`, `models.py`, `clustering.py` | `mulehunt/graph.py`, `search.py`, `adversarial.py`, `local_search.py`, `text/`, `app/` |
| Notebooks | 01 simulator, 05 classifiers, 06 clustering | 02 search, 03 adversarial, 04 local search, 07 scam text |
| Report chapters | Data and validation, supervised models, clustering | Agent design, search, adversarial and local search, dashboard |

The load balances out: A has the hardest single task (the simulator), and B has more separate pieces.

## How you avoid waiting on each other

The only real dependency is that B's graph and search need A's data. Three agreed file formats plus a toy dataset on day 1 remove it.

| File | Columns | Produced by | Ready |
| --- | --- | --- | --- |
| `data/generated/accounts.csv` | account\_id, account\_type, bank, state, created\_at, role, is\_fraud | A | Toy: day 1. Real: end of week 2 |
| `data/generated/transactions.csv` | txn\_id, timestamp, sender\_id, receiver\_id, amount, channel (P2P/P2M), device\_id, scenario, is\_fraud | A | Toy: day 1. Real: end of week 2 |
| `data/processed/features.parquet` | account\_id, feature columns, label, split (train/val/test) | A | Week 3 |
| `models/risk_scores.csv` | account\_id, risk\_prob | A | Week 4 |

On day 1, A writes `make_toy_data.py` (about 30 minutes): roughly 500 accounts with 3 hand-made mule chains in exactly these formats. B builds everything against the toy data from day 2, and the code works unchanged when the real data arrives.

- **Until A's model exists:** B's A\* uses a placeholder rule score (for example, high pass-through ratio means high risk), then switches to `risk_scores.csv` in week 4.
- **Local search:** B's feature selection needs only `features.parquet` and a fast logistic regression as the fitness function, not A's final models.

## Week-by-week plan

Each week ends with one handoff from A to B; B never stops while waiting for it.

| Week | Person A | Person B | Handoff |
| --- | --- | --- | --- |
| 1 | Toy data. Normal user behaviour in the simulator (salaries, rent, merchant payments, festival spikes). Collect NPCI/RBI calibration numbers into `config.yaml` | Launch the scam-message Google Form, since collection takes time. Build the time-respecting transaction graph. BFS and DFS on toy data | A to B: toy data (day 1) |
| 2 | All 5 scam scenarios and mule patterns (fan-out, fan-in, chains, cash-out). Noise: sleeper mules, genuine outliers, label noise | DLS and IDDFS. Comparison table (nodes expanded, time, memory). PEAS and environment description | A to B: first real dataset (end of week) |
| 3 | Validation against NPCI/RBI stats and IBM AML. Feature engineering, time-based split, `features.parquet` | A\* with risk heuristic, admissibility proof, comparison with uniform-cost search. Start the minimax game design | A to B: `features.parquet` |
| 4 | Decision tree with overfitting curves and pruning. KNN, Naive Bayes, logistic regression | Minimax with alpha-beta, node-count comparison. Hill climbing, simulated annealing, beam search and GA for feature selection and threshold tuning | A to B: `risk_scores.csv`, so A\* drops the placeholder |
| 5 | Bagging, boosting, random forest leaderboard (precision, recall, F1, PR-AUC). Clustering: K-means, bisecting K-means, elbow, silhouette, proximity matrix, purity and ARI against true roles | Label scam messages, train Naive Bayes on character n-grams. Streamlit app with all 4 pages, using saved models | A to B: saved models (`.joblib`) for the dashboard |
| 6 | Write own report chapters. Review B's code | Write own report chapters, README, record backup demo video. Review A's code | Merge report, rehearse viva |

## Working rules

- **Git:** one repo. Each person works only in their own folders, so merge conflicts don't happen. Use a branch per feature and a quick pull request reviewed by the other person, which doubles as viva prep.
- **Shared files:** only `CONTRACTS.md`, `requirements.txt` and `README.md`. Change these only after telling the other person.
- **Weekly 20-minute sync** (for example, Sunday): check that week's handoff, flag anything blocked, show one result each.
- **If A's real data is late:** B carries on with the toy data. Only the final numbers wait.
- **Viva insurance:** in week 6, each person explains the other's module out loud. Examiners often ask about the part you didn't build.

## Day 1 checklist

- [ ] Both: create the repo and folder structure from the implementation plan
- [ ] Both: write `CONTRACTS.md` with the four file formats above
- [ ] A: commit `make_toy_data.py` and the toy CSVs
- [ ] B: publish the scam-message Google Form and share it with friends and family
- [ ] Both: set up `requirements.txt` (pandas, numpy, networkx, scikit-learn, imbalanced-learn, matplotlib, seaborn, streamlit, pyvis)

Open question: the final submission date. The 6-week schedule shifts to fit it.
