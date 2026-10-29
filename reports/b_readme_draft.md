# MuleHunt

Detecting and tracing UPI money-mule networks: a course project for *IS353IA Artificial Intelligence and Machine Learning* (RVCE, Semester V).

> Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark. The scam-message dataset is the one fully real dataset.

## Repo map

| Part | Person | Where |
|---|---|---|
| Simulator, validation | A (Nithin Eashwar) | `simulator/`, `reports/a_01_simulator.md` |
| Features, 8 classifiers, risk scores | A | `mulehunt/features.py`, `models.py`, `models/`, `reports/a_05_classifiers.md` |
| Clustering | A | `mulehunt/clustering.py`, `reports/a_06_clustering.md` |
| Time-respecting graph, BFS/DFS/DLS/IDDFS/UCS/A\* | B | `mulehunt/graph.py`, `search.py`, `trace.py`, `reports/b_02_search.md` |
| Minimax game with alpha-beta | B | `mulehunt/adversarial.py`, `reports/b_03_adversarial.md` |
| Local search (HC, SA, beam, GA) | B | `mulehunt/local_search.py`, `reports/b_04_local_search.md` |
| Scam messages + Naive Bayes | B | `text/`, `data/raw/scam_messages/`, `reports/b_07_scam_text.md` |
| Dashboard | B | `app/` |
| Report chapters | A, B | `reports/a_chapters/`, `reports/b_chapters/` |

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data (nothing large is in git)

```bash
python -m simulator.generate --config simulator/config.yaml --out data/generated
python -m mulehunt.features
```

Compare SHA-256 with `data/generated/CHECKSUMS.txt`. Without the real files, everything falls back to the toy data in `data/generated/toy/`.

## One command per part

```bash
python -m mulehunt.pipeline            # Person A: generate -> validate -> features -> models -> clustering
python -m mulehunt.b_pipeline          # Person B: trace -> adversarial -> local search -> text -> app precompute
streamlit run app/streamlit_app.py     # the dashboard
```

## Headline results

| Result | Value | Source |
|---|---|---|
| Best classifier (random forest), test | PR-AUC 0.915, precision 0.962, recall 0.753 | `reports/a_tables/leaderboard.csv` |
| Tracing: rings traced to their cash-out | 75 / 75 (99.5% of 2,278 victim-payment starts) | `reports/b_tables/search_comparison.csv` |
| Search vs classifier: mules recovered | trails 455 / 455, classifier 309 / 455; 146 found only by trails, 1 genuine account touched | `reports/b_tables/search_vs_classifier.csv` |
| `fake_kyc` mules recovered | trails 92 / 92 vs classifier 32 / 92 | same |
| Alpha-beta node savings (R = 3) | 691 vs 18,631 nodes (96.3% fewer), same value | `reports/b_tables/game_nodes.csv` |
| Best feature subset (local search) | 11 of 20 features, val PR-AUC 0.778 vs 0.756; test PR-AUC 0.816 | `reports/b_tables/local_fs_*.csv` |
| Cost-tuned threshold (C_FN = 50), test | cost 617 vs 2,105 at the F1 threshold; recall 0.95 vs 0.75 | `reports/b_tables/local_threshold_test.csv` |
| Scam-message NB macro-F1 | pending real data | `reports/b_07_scam_text.md` |

## Syllabus mapping

(Unit numbers for III-V are our reading of the syllabus; check them against the official one.)

| Unit | Topic | Module | Report |
|---|---|---|---|
| I | Agents, PEAS, environments | the system as a whole | `reports/b_chapters/04_agent_design.md` |
| II | Uninformed search (BFS, DFS, DLS, IDDFS) | `mulehunt/search.py` | `b_02_search.md`, chapter 5 |
| II | Informed search (UCS, A\*, admissible heuristics) | `mulehunt/search.py` | `b_02_search.md`, chapter 5 |
| II | Adversarial search (minimax, alpha-beta) | `mulehunt/adversarial.py` | `b_03_adversarial.md`, chapter 6 |
| II | Local search (HC, SA, beam, GA) | `mulehunt/local_search.py` | `b_04_local_search.md`, chapter 6 |
| III | Decision trees, Naive Bayes, KNN, LR | `mulehunt/models.py`, `text/classifier.py` | `a_05_classifiers.md`, chapters 2 and 7 |
| IV | Ensembles (bagging, RF, AdaBoost, GB) | `mulehunt/models.py` | `a_05_classifiers.md`, chapter 2 |
| V | Clustering | `mulehunt/clustering.py` | `a_06_clustering.md`, chapter 3 |

## Ethics and limitations

- Transactions are simulated; results show what the methods *can* do on realistic structure, not measured performance on a bank's data.
- No Kaggle "UPI" or PaySim data, and no real person's UPI statement, is used.
- Scam messages are collected with consent, scrubbed automatically and reviewed by hand; the raw export never enters git.
- Search assumes the bank knows the cash-out accounts (B-D1).
- Every flag is a prompt for human review, never an automatic freeze.

## Team

- **Person A: Nithin Eashwar**: simulator, data, features, classifiers, clustering.
- **Person B**: search, adversarial and local search, scam-message classifier, dashboard.
