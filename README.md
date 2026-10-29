# MuleHunt

Detecting and tracing UPI money-mule networks: a course project for *IS353IA Artificial Intelligence and Machine Learning* (RVCE, Semester V).

> Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark. The scam-message corpus combines the real UCI SMS Spam Collection with Indian messages we wrote from cited reports and from templates; no message was collected from a person.

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

Run every command from the repo root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
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
| Clustering | K-means K = 20: the top cluster has 16.2× the mule rate; 77% of mules land in clusters with ≥ 5× lift | `reports/a_06_clustering.md` |
| Tracing (best case\*): rings traced to their cash-out | 75 / 75 (99.5% of 2,278 victim-payment starts) | `reports/b_tables/search_comparison.csv` |
| Search vs classifier (best case\*): mules recovered | trails 455 / 455, classifier 309 / 455; 146 found only by trails, 1 genuine account touched | `reports/b_tables/search_vs_classifier.csv` |
| `fake_kyc` mules recovered (best case\*) | trails 92 / 92 vs classifier 32 / 92 | same |
| Alpha-beta node savings (R = 3) | 691 vs 18,631 nodes (96.3% fewer), same value | `reports/b_tables/game_nodes.csv` |
| Feature selection (local search) | 11 of 20 features beat all 20 on validation (PR-AUC 0.778 vs 0.756) but not on test (0.816 vs 0.832): the validation gain did not carry over | `reports/b_tables/local_fs_test.csv` |
| Cost-tuned threshold (C_FN = 50), test | recall 0.95 vs 0.75 at the F1 threshold, by flagging 217 genuine accounts (1.7%) instead of 5; cost 617 vs 2,105 (the cost it was tuned to minimise) | `reports/b_tables/local_threshold_test.csv` |
| Scam-message classifier, transfer test | catches 54 / 56 cited Indian scam messages it never saw; 0.4% false alarms on genuine UK/Singapore SMS (untested on genuine Indian bank SMS); NB macro-F1 0.976 (words + chars) | `reports/b_tables/text_transfer_*.csv` |

\* **Best case:** search is told exactly which accounts are cash-outs (assumption B-D1), and in the simulator cash-out accounts forward nothing. These tracing numbers are partly guaranteed by that assumption; with a partial goal set they would be lower. See `reports/b_chapters/05_search.md` §5.4.

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
- Scam messages: UCI SMS Spam Collection (Almeida & Hidalgo 2011, CC BY 4.0) plus messages we wrote, each labelled by source; reported messages cite their source URL. All are anonymised.
- Search assumes the bank knows the cash-out accounts (B-D1).
- Every flag is a prompt for human review, never an automatic freeze.

## Team

- **Person A: Nithin Eashwar**: simulator, data, features, classifiers, clustering.
- **Person B**: search, adversarial and local search, scam-message classifier, dashboard.
