# MuleHunt — Person A Handoff (Data & Learning)

You are a coding agent. You have not seen the conversation that produced this project. This document is everything you need to build **Person A's half** of MuleHunt in an empty GitHub repo. The human you work with is **Person A**. Person B (a teammate) builds the search/graph/game/dashboard half in parallel in the same repo.

**Legend**
- **NOT DECIDED** — nobody decided this. Do not silently invent it. Either ask Person A, or pick a value, write it into `config.yaml` / `CONTRACTS.md`, and tell Person A which value you chose. Where I give a **PROPOSAL**, it is my suggestion only and has no standing.
- Anything under a heading that says "shared contract" must not change without telling Person B (via Person A). Column names in section 3 are fixed.

Reference artifacts (for the human; you may not be able to open them):
- Work plan doc: https://claude.ai/artifact/U3VLNeQCRGGSZYUv1jhap2
- Pitch deck: https://claude.ai/artifact/VFyHK38DE3DGPpmfsgy7Hq
- A synopsis `.docx` (MuleHunt_Synopsis.docx) was also produced; its content is summarised in section 1.

---

## 1. Project summary

**MuleHunt: AI-Driven Detection and Tracing of UPI Money-Mule Networks in India.**

**Problem.** UPI fraud victims in India (fake KYC calls, "digital arrest", job/task scams, loan-app extortion, fake investment platforms) lose money that vanishes within minutes through chains of "mule" accounts. Current bank/NPCI checks score single transactions against fixed thresholds, so they (a) miss the chain structure and (b) are evaded by fraudsters who split payments under the threshold. MuleHunt traces money through a time-respecting transaction graph (search), models fraudster-vs-detector as a game (minimax), classifies fraudulent accounts/transactions (supervised ML on imbalanced data), and finds mule-like behaviour groups without labels (clustering).

Scale facts used in the pitch (source: Lok Sabha Unstarred Question 2481, 15 Dec 2025): 12.64 lakh UPI fraud incidents and ₹981 crore in FY2024-25.

**Course.** *Artificial Intelligence and Machine Learning*, course code **IS353IA**, Semester V, Professional Core Course (Theory and Lab), common to CS, CD, CY & IS. Credits L:T:P = 3:0:1. Total hours 45L + 30P. CIE 100 + 50 marks, SEE 100 + 50 marks, SEE duration 3 + 3 hours.

**Project requirements (from the user).** Must address a real Indian scenario/issue; must involve a dataset plus model training; mapping to the syllabus is a bonus; should be creative and resume-worthy; team of two, split modularly so each works independently.

**How it is graded.** NOT DECIDED / unknown. No rubric, marks split, or examiner checklist was shared. A synopsis (Title, Abstract, Problem Statement, Objectives, Proposed Methodology, Expected Results, Conclusion) and a pitch to the professor were prepared; a viva is expected (the plan includes "viva prep"). The professor's stance on simulated data: NOT DECIDED / unknown (the user asked earlier whether faculty require real data; no answer was recorded).

**Syllabus (full, as given).**
- **Unit I (9 hrs):** What is AI?; Intelligent agents (agents and environment, rationality, nature of environments, structure of agents); Problem solving & uninformed search (problem-solving agents, BFS, DFS, depth-limited, iterative deepening DFS).
- **Unit II (9 hrs):** Informed search (A*, heuristic functions); Beyond classical search (local search and optimization problems, hill climbing, simulated annealing, local beam search, genetic algorithms); Adversarial search (games, optimal decisions in games, alpha-beta pruning).
- **Unit III (9 hrs):** Supervised learning (basic concepts, general framework for classification); Decision tree classifier (basic algorithm to build a tree, methods for expressing attribute test conditions, measures for selecting an attribute test condition, algorithm for decision tree induction, characteristics of decision tree classifiers); Model overfitting (reasons for model overfitting); Model selection (using a validation set, incorporating model complexity, estimating statistical bounds, model selection for decision trees, model evaluation).
- **Unit IV (9 hrs):** Nearest neighbor classifiers (characteristics); Naive Bayes classifier (basics of probability theory, Naive Bayes assumption); Logistic regression (as a generalized linear model, learning model parameters, characteristics); Ensemble methods (constructing ensemble classifiers, bagging, boosting, random forests).
- **Unit V (9 hrs):** Unsupervised learning (overview, what is cluster analysis, different types of clusterings, different types of clusters); K-means (basic algorithm, additional issues, bisecting K-means, K-means and different types of clusters, strengths and weaknesses, K-means as an optimization problem); Cluster evaluation (overview, unsupervised evaluation using cohesion and separation, using the proximity matrix, determining the correct number of clusters, supervised measures of cluster validity, assessing the significance of cluster validity measures, choosing a cluster validity measure).

**Who owns what.**

| | Person A (you build this) | Person B (do NOT build) |
|---|---|---|
| Owns | Simulator, dataset validation, feature engineering, all fraud models, clustering | Scam-message dataset + Naive Bayes text classifier, transaction graph, BFS/DFS/DLS/IDDFS/A*, minimax + alpha-beta, local search (hill climbing, SA, beam, GA), Streamlit dashboard |
| Syllabus units | III, IV, V | I, II (+ Naive Bayes for text) |
| Report chapters | Data and validation; supervised models; clustering | Agent design; search; adversarial and local search; dashboard |

Shared: limitations/ethics, final report merge, README, viva rehearsal.

**Data-source history (why a simulator).** No public transaction-level Indian UPI fraud dataset exists (banks/NPCI cannot release it). The original idea was to adapt PaySim; the final decision is: **a custom UPI transaction simulator, calibrated to NPCI/RBI published statistics and validated against the IBM AML benchmark.** The honest framing to use everywhere: *"Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark."*

---

## 2. Repo structure

From the implementation plan, plus `CONTRACTS.md` / `requirements.txt` from the work plan.

```
mulehunt/
├── CONTRACTS.md              # the 4 shared file formats (section 3); changes need Person B's knowledge
├── README.md                 # problem summary, architecture, how to run, results table, screenshots  [shared]
├── requirements.txt          # pinned deps  [shared]
├── data/
│   ├── raw/                  # IBM AML sample, NPCI/RBI stats, scam messages (B)
│   ├── generated/            # simulator output: accounts.csv, transactions.csv
│   └── processed/            # features.parquet (+ train/val/test splits)
├── simulator/                # PERSON A
│   ├── config.yaml           # ALL calibration numbers, each with its source written beside it
│   ├── accounts.py           # normal users, merchants, mules
│   ├── behaviors.py          # salary, rent, daily spend, festivals
│   ├── scams.py              # the five fraud scenarios
│   ├── generate.py           # entry point: writes data/generated/*.csv
│   └── make_toy_data.py      # day-1 toy dataset (~500 accounts, 3 hand-made mule chains). Location NOT DECIDED; put it here
├── mulehunt/
│   ├── features.py           # PERSON A: per-account features -> features.parquet
│   ├── models.py             # PERSON A: all supervised models + evaluation + saving
│   ├── clustering.py         # PERSON A: K-means, bisecting K-means, validity measures
│   ├── graph.py              # B: build the time-respecting transaction graph
│   ├── search.py             # B: BFS, DFS, DLS, IDDFS, A*
│   ├── adversarial.py        # B: minimax + alpha-beta
│   └── local_search.py       # B: hill climbing, SA, local beam, GA
├── text/                     # B: scam-message classifier
├── notebooks/                # 01_simulator ... 07_*; A owns 01, 05, 06 (see section 9)
├── app/streamlit_app.py      # B: dashboard
├── models/                   # saved .joblib models + risk_scores.csv (A produces)
├── reports/                  # figures, tables, report chapters
└── README.md
```

**Git rules.** One repo. Each person edits only their own folders (no merge conflicts). Branch per feature, small PR reviewed by the other person. Shared files (`CONTRACTS.md`, `requirements.txt`, `README.md`) change only after telling the other person. Repo name/URL: NOT DECIDED.

**Naming conflict to resolve.** The implementation plan numbered notebooks `01_simulator ... 07_clustering`; the later work plan assigns A: 01 simulator, 05 classifiers, 06 clustering and B: 02 search, 03 adversarial, 04 local search, 07 scam text. **Use the work-plan numbering.** Exact filenames beyond the number: NOT DECIDED.

---

## 3. Data contracts (shared contract)

Written on day 1 into `CONTRACTS.md`. Column **names and order below are fixed**. Types, allowed values and ID formats were not decided except where stated.

### 3.1 `data/generated/accounts.csv`
Columns: `account_id, account_type, bank, state, created_at, role, is_fraud`

| Column | Decided | NOT DECIDED / proposal |
|---|---|---|
| `account_id` | unique per account | Format NOT DECIDED. PROPOSAL: string `ACC` + zero-padded integer, e.g. `ACC000123` |
| `account_type` | from the account kinds in section 4.1 (salaried, student, gig worker, small merchant, large merchant, plus whatever types the fraud actors need) | Exact string values NOT DECIDED. PROPOSAL: `salaried, student, gig_worker, small_merchant, large_merchant` |
| `bank` | an Indian bank | Value list NOT DECIDED. A Kaggle synthetic UPI dataset used SBI, HDFC, ICICI, Axis, PNB, Kotak, IndusInd, Yes Bank with market-share-based probabilities; reuse as a starting point |
| `state` | an Indian state | Value list NOT DECIDED. Same Kaggle dataset used Maharashtra, Karnataka, Delhi, Tamil Nadu, West Bengal, Gujarat, Rajasthan, Uttar Pradesh, Andhra Pradesh, Telangana |
| `created_at` | account creation (needed for "account age" features) | Format NOT DECIDED. PROPOSAL: ISO-8601 date-time |
| `role` | **ground truth role** used for clustering validation. Roles named in the plan: normal user, merchant, fraudster, victim, tier-1 mule, tier-2 mule, tier-3 mule, cash-out account | Exact strings NOT DECIDED. PROPOSAL: `normal, merchant, victim, fraudster, mule_t1, mule_t2, mule_t3, cashout` |
| `is_fraud` | binary label | Which roles count as 1: NOT DECIDED. PROPOSAL: fraudster, mule_t1/t2/t3, cashout = 1; normal, merchant, victim = 0. Whether label noise is applied here (1–2%) while `role` stays clean: NOT DECIDED, but `role` must stay clean because it is the ground truth for ARI/purity |

### 3.2 `data/generated/transactions.csv`
Columns: `txn_id, timestamp, sender_id, receiver_id, amount, channel, device_id, scenario, is_fraud`

| Column | Decided | NOT DECIDED / proposal |
|---|---|---|
| `txn_id` | unique | Format NOT DECIDED. PROPOSAL: `TXN` + zero-padded integer |
| `timestamp` | when it happened | Format/timezone NOT DECIDED. PROPOSAL: ISO-8601 in IST |
| `sender_id`, `receiver_id` | `account_id` values from accounts.csv | — |
| `amount` | rupees | PROPOSAL: float, INR, 2 decimals |
| `channel` | **`P2P` or `P2M`** (decided) | — |
| `device_id` | device used by the sender | Format NOT DECIDED. Needed for "device changes" feature |
| `scenario` | which scam scenario (or normal) | Values NOT DECIDED. PROPOSAL: `normal, fake_kyc, digital_arrest, job_task, loan_app, fake_investment` |
| `is_fraud` | transaction-level label | Exact rule NOT DECIDED. PROPOSAL: 1 for every transaction in a scam flow (victim→mule, mule→mule, mule→cash-out) |

**Gap in the contract.** The earlier implementation plan listed a `location` column and a "state mismatches" feature, but the contract has no location column. Options: derive location from the sender's `state` in accounts.csv, or add a column. NOT DECIDED — adding a column is a contract change (tell Person B).

### 3.3 `data/processed/features.parquet`
Layout: `account_id`, then feature columns, then `label` and `split` (values `train`/`val`/`test`). One row per account. Feature column list: section 6. `label` is derived from `is_fraud`. Writing parquet needs `pyarrow` (not on the original dependency list; add it to `requirements.txt`).

### 3.4 `models/risk_scores.csv`
Columns: `account_id, risk_prob`. Person B's A* turns this into edge cost = **−log(risk probability)** (minimising the sum maximises combined suspicion along a path). Implication: `risk_prob` must be strictly greater than 0 for every account or the cost is infinite. Clip it (PROPOSAL: clip to [1e-6, 1−1e-6]). Which model produces it: NOT DECIDED (expected: best ensemble). Whether scores are out-of-fold or in-sample: NOT DECIDED (PROPOSAL: out-of-fold for train accounts so tracing isn't fed memorised labels). Must be ready **end of Week 4**; until then B uses a placeholder rule score (e.g. high pass-through ratio = high risk).

### 3.5 Saved models for the dashboard
Decided: models are saved as `.joblib` (end of Week 5) and the dashboard "loads saved models and precomputed features so the demo is fast." The dashboard's transaction-check page shows a **fraud probability with the top contributing features**. File names, the interface (e.g. `predict_proba`), and how to get "top contributing features" are NOT DECIDED. PROPOSAL: save each fitted sklearn `Pipeline` (scaler + model) to `models/<name>.joblib`, e.g. `random_forest.joblib`, `gradient_boosting.joblib`, `logistic_regression.joblib`, `decision_tree.joblib`; plus `models/feature_columns.json` (ordered feature names); contributions via `feature_importances_` (trees) or coef × standardised value (logistic regression). The scam-message Naive Bayes model is Person B's.

### 3.6 Day-1 toy data (unblocks Person B)
Deliverable on day 1 (~30 minutes): `make_toy_data.py` producing about **500 accounts with 3 hand-made mule chains**, in exactly the formats above. Person B builds everything against it from day 2; the real dataset must be a drop-in replacement.

---

## 4. Simulator design

Goal: realistic Indian UPI accounts and transactions with perfect ground truth (roles) for classification and clustering, plus enough noise that models do **not** score a suspicious ~100%.

### 4.1 Normal behaviour
- **Population:** about **20,000–50,000 accounts** in the real dataset (toy: ~500). Mix per type: NOT DECIDED.
- **Account types:** salaried users, students, small merchants (kirana stores, chai stalls), large merchants, gig workers. Each gets a bank, state, account age and a device.
- **Time span:** about **90 days**. Start date NOT DECIDED (choose so a festival such as Diwali falls inside the window, since a festival spike is required).
- **Behaviours to generate:**
  - salary credits near the start of the month;
  - rent payments;
  - bills and recharges;
  - many small P2M payments (merchant payments);
  - P2P transfers to family;
  - weekend and night patterns;
  - a festival spending spike.
- **Distribution shapes and parameters** (amount distributions, transactions per day per type, hour-of-day curve, P2P/P2M mix, festival uplift size): NOT DECIDED. They must come from calibration sources in section 5. Reference for design ideas: a public Kaggle synthetic dataset ("UPI Transactions 2024 Dataset") uses category-specific amount ranges (Food, Grocery, Fuel, Entertainment, Shopping, Healthcare, Education, Transport, Utilities, Other), a weekend spending boost for Food/Entertainment/Shopping, and market-share-based bank probabilities.

### 4.2 Roles
Fraudster, victims, tier-1 / tier-2 / tier-3 mules, cash-out accounts (plus normal users and merchants). The user's wording once also used "controller"; the plan's term is **fraudster**. Role generation counts (how many fraudsters, mules per fraudster, victims per scenario): NOT DECIDED.

### 4.3 The five scam scenarios (from the plan)
Each creates a **victim payment followed by a mule chain**. Amounts, timing gaps and counts are NOT DECIDED unless stated.

| # | Scenario | Story / what it produces |
|---|---|---|
| 1 | **Fake KYC / bank impersonation** | One victim makes **one large payment**, then **fast fan-out** to many accounts |
| 2 | **"Digital arrest"** | **Repeated large payments from one victim over days** |
| 3 | **Job/task scams** | **Many victims, small payments gathered into one account (fan-in)** |
| 4 | **Loan-app extortion** | **Repeated small payments from many victims** |
| 5 | **Fake investment platform** | **Small early "returns" paid back to build trust, then one large loss** |

Evasion behaviour (feeds Person B's minimax game): fraudsters **split amounts to stay under detection thresholds**. The game's discrete threshold options are ₹10k / ₹25k / ₹50k plus a velocity rule; "more mules cost more" (extra mules = extra cost to the fraudster). So include a share of fraud flows whose per-transaction amounts sit just under those thresholds. What share: NOT DECIDED.

### 4.4 Mule patterns
Implement: **fan-out, fan-in, chains, scatter-gather, occasional cycles**, ending in **cash-out** (ATM cash-out or crypto, per the plan). These match patterns in the IBM AML data (its documentation lists 8 laundering patterns from the AMLSim simulator by Suzumura & Kanezashi). How each is parameterised (chain depth, fan width, hop delay): NOT DECIDED. Constraint from Person B's design: money moves **forward in time**, and B only counts a hop B→C if it follows A→B within a bounded window (the plan says "say 48 hours"). Generate hops so most chains have realistic fast hops (mules receive and forward quickly — dwell time and pass-through ratio are expected strongest features).

### 4.5 Noise (mandatory)
- **Sleeper mules:** behave normally for weeks before activating.
- **Genuine outliers:** normal users with unusual spikes (a wedding, a hospital bill, buying a phone).
- **Label noise:** about **1–2%**.
- Fraudsters splitting amounts to stay under thresholds (above).

### 4.6 `config.yaml`
All calibration numbers live in one file, **each with its source written beside it**, so the report can cite where every assumption came from. The plan names these parameter groups: ticket-size distribution, P2P vs P2M share, hour-of-day pattern, fraud share, festival uplift, scenario mix, bank/state mix. **No numeric values were ever decided.** Fill from section 5 sources; record the source next to each key; where no source exists, mark it `assumed` in the file.

### 4.7 Target size and fraud ratio
- Accounts: ~20,000–50,000. Transactions: NOT DECIDED.
- Fraud ratio: **NOT DECIDED**. The plan's only constraint: fraud is **under 1% of data**, which is why accuracy is meaningless (section 7). For context only (other datasets, not our targets): the Kaggle synthetic UPI dataset uses a 0.2% base rate; the IBM AML HI-Small subsample used in one paper is 0.27% laundering; SAML-D is 0.1039% suspicious.
- Keep enough fraud accounts that each of the five scenarios appears in train, val and test (needed for the per-scenario breakdown).

---

## 5. Calibration and validation

### 5.1 Sources gathered
**India UPI fraud (for scale and fraud share).** Ministry of Finance, Lok Sabha Unstarred Question No. 2481, answered 15 Dec 2025 ("UPI Domestic Payment Frauds"). PDF: https://eparlib.sansad.in/bitstream/123456789/3018042/1/AU2481_5bIzKk.pdf

| Financial year | Incidents (lakh) | Amount involved (₹ crore) |
|---|---|---|
| 2021-22 | 4.07 | 242 |
| 2022-23 | 7.25 | 573 |
| 2023-24 | 13.42 | 1,087 |
| 2024-25 | 12.64 | 981 |
| 2025-26 (till Nov 2025) | 10.64 | 805 |

Implied average loss per incident (FY2024-25): ₹981 crore ÷ 12.64 lakh ≈ ₹7,761 — useful for sanity-checking victim-payment amounts (derived by me, verify).

**Other India figures encountered (secondary sources; verify before citing).**
- Bank-reported cyber frauds of ₹1 lakh and above (Lok Sabha): FY2023-24 29,082 cases / ₹177.05 crore; FY2024-25 (Apr–Dec 2024) 13,384 cases / ₹107.21 crore. https://www.angelone.in/news/market-updates/india-reports-over-rs-107-crore-in-cyber-fraud-losses-first-three-quarters-fy25
- **Conflict:** a Medianama summary of the RBI FY25 annual report gives the *same count*, 29,082 frauds for FY24, but **₹1,457 crore** and labels them card/internet. The two cannot both be right for the same series. Do not cite either without checking the primary source. https://www.medianama.com/2025/05/223-upi-84-india-fy25-retail-payment-volume-rbi/
- National Cyber Crime Reporting Portal: financial-fraud complaints reached 24.03 lakh in 2025 (reported by a secondary exam-prep site; unverified).
- RBI July 2024 guidelines require banks to run Early Warning Systems for suspicious transactions.
- NPCI provides banks an AI/ML-based fraud-monitoring solution; MHA runs the national cybercrime portal and helpline 1930.

**NPCI monthly UPI statistics** (volume, value, P2P vs P2M share, ticket size): required by the plan as the main calibration source for `config.yaml`, but **no numbers were collected in the conversation**. You must fetch them from NPCI's UPI statistics page (verify URL) and record them with a date. One datapoint seen in passing was about UPI's share of all digital payment volume (82%) — that is NOT the P2M share; don't use it as such.

**IBM AML benchmark (structure validation).** IBM "Transactions for Anti Money Laundering (AML)": https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml · https://github.com/IBM/AML-Data (data licence CDLA-Sharing-1.0). Synthetic, produced by a multi-agent virtual world of banks, individuals and companies; the generator tracks illicit funds through arbitrarily many transactions, so laundering is labelled many hops from the source; lists laundering transactions following 8 AMLSim patterns (not all laundering follows them). Variants: HI-Small (515,088 accounts/nodes, 5,078,345 edges), LI-Small (705,907 / 6,924,055), HI-Medium, LI-Medium, HI-Large (~180M txns), LI-Large. Use **HI-Small** (laptop-friendly). It is not Indian: use it only to check pattern structure, not as training data (amounts/currencies would need remapping if ever reused). Other references: SAML-D (9,504,852 txns, 28 typologies: 11 normal + 17 suspicious, 0.1039% suspicious) https://arxiv.org/pdf/2404.14746 ; IBM paper: https://dl.acm.org/doi/abs/10.5555/3666122.3667422.

### 5.2 Datasets NOT to rely on
- **Kaggle synthetic "UPI" datasets** (e.g. kalpitlabs "Pattern-Based UPI Transaction Risk", skullagos5246 "UPI Transactions 2024 Dataset", devildyno, maulikgajera, iamravi11, omsshete) have Indian-flavoured fields but are flat tables with **no account-to-account network** and, in many of them, a fraud flag assigned near-randomly so models can learn nothing real. Use for design ideas only.
- **PaySim:** its fraud labels follow a very simple pattern (models reach ~100% accuracy, which looks trivial to an examiner) and it has almost no multi-hop chains.

### 5.3 Validation checks the generated data must pass
Deliver a **validation report** (it becomes the "Data Validation" chapter, the main defence against "your data is fake"):
1. **Ticket-size distribution** matches NPCI/RBI-calibrated targets.
2. **P2P/P2M share** matches NPCI.
3. **Hour-of-day pattern** matches calibration.
4. **Fraud share** consistent with the chosen target (section 4.7).
5. **IBM AML pattern check:** the laundering patterns (fan-out, fan-in, chains, scatter-gather, cycles) appear in our graph.
6. Models do **not** score near-100% (sanity check; if they do, add noise / look for leakage).

Numeric pass/fail tolerances: NOT DECIDED (PROPOSAL: report the target, the generated value and the gap in a table; let Person A set tolerances).

---

## 6. Feature engineering (`mulehunt/features.py`)

**Aggregation level:** per account, over a rolling window. Window length: NOT DECIDED. Output: `data/processed/features.parquet` (section 3.3).

**Feature families and members (from the plan).**
- **Flow:** total inflow, total outflow; **pass-through ratio = (money sent out within 24 hours) ÷ (money received)**; **average dwell time** before forwarding money received.
- **Network:** in-degree, out-degree; number of new counterparties; fan-in count, fan-out count; whether neighbours were flagged.
- **Behavioural:** night transaction share; round-number amounts; amounts just below common thresholds (₹10k/₹25k/₹50k); burstiness; account age at first large receipt ("large" cut-off NOT DECIDED).
- **Device/location:** device changes; state mismatches (see contract gap in 3.2).

Pass-through ratio and dwell time are expected to be the strongest features and are easy to explain in the viva. Exact formulas for the rest (burstiness measure, "round number" rule, night hours, "just below" margin, large-receipt threshold): NOT DECIDED; document each choice in the code docstring and notebook.

**Split (time-based, to prevent leakage).** Train = **days 1–60**, validation = **days 61–75**, test = **days 76–90**. Also from the plan: *accounts must not appear in both train and test.* These two rules conflict for account-level features (a mule active across days appears in several periods). How to reconcile (e.g. assign each account to one split by its first-fraud or first-activity day and compute its features only from that split's days): **NOT DECIDED — resolve with Person A and document it.**

**Leakage rules (explicit in the plan).**
- Never apply SMOTE to validation or test data; resample training only.
- "Neighbours flagged" uses labels: compute it only from training-period labels (or from predictions), never from val/test labels.
- No feature may be derived from `role`, `is_fraud` or `scenario` of the account being scored.
- Leakage is called out as the most common reason student fraud projects get suspiciously perfect scores.

---

## 7. Models (`mulehunt/models.py`, notebook 05)

All models use the **same features and the same splits** and go into **one leaderboard**.

1. **Decision tree (Unit III):** train at increasing depths and plot **training vs validation** performance (overfitting curve); choose depth using the validation set; apply **cost-complexity pruning**; visualise the final tree as an explainable "mule rulebook". Syllabus items to show explicitly: attribute-test conditions, impurity measures for choosing a split (e.g. Gini vs entropy), model complexity, model selection, model evaluation. "Estimating statistical bounds" (e.g. confidence intervals on the metrics): NOT DECIDED how.
2. **KNN:** scaled features, tuned k.
3. **Naive Bayes:** explain the independence assumption and why it suffers with correlated flow features. (The scam-*message* Naive Bayes is Person B's.)
4. **Logistic regression:** report coefficients as risk factors; use class weighting.
5. **Bagging**, 6. **Gradient boosting**, 7. **Random forest.**

**Class imbalance:** use `imbalanced-learn` (SMOTE on training data only, per the leakage rule) and/or class weights. Which method per model: NOT DECIDED.

**Metrics:** **precision, recall, F1, PR-AUC**, and **recall at a fixed false-alarm rate** (the rate itself: NOT DECIDED). **Do not headline accuracy** — with fraud <1%, predicting "not fraud" always scores >99%; say so explicitly in the report.

**Required extra analyses:**
- **Ablation:** how much network features add over transaction-only features.
- **Per-scenario results** (one row per scam scenario).

**Expected outputs:** depth-vs-train/val curve; pruned-tree visual; leaderboard table (all models × metrics); PR curves (implied by PR-AUC); logistic-regression coefficient table; ablation table; per-scenario table. Models saved to `models/*.joblib`; `risk_scores.csv` written (section 3.4).

**Handoff to Person B:** `risk_scores.csv` end of Week 4; `.joblib` models end of Week 5. Person B's local search uses `features.parquet` with a fast logistic regression as fitness (feature selection) and the cost function "a missed fraud costs far more than a false alarm" (ratio NOT DECIDED, B's side) — so expose a fast, reproducible logistic-regression baseline in `models.py` that B can import.

---

## 8. Clustering (`mulehunt/clustering.py`, notebook 06)

1. Cluster accounts on **scaled behavioural features without using any labels**. Which feature subset: NOT DECIDED; the plan's examples are **pass-through ratio, dwell time, fan-in/fan-out behaviour** (add other behavioural features as justified). Scale before clustering.
2. **K-means** over a range of K (range NOT DECIDED); choose K with the **SSE elbow plot plus silhouette score**.
3. **Bisecting K-means** (scikit-learn `BisectingKMeans`) and compare with K-means.
4. **Proximity matrix** sorted by cluster; clean clusters appear as dark blocks. (Sample accounts if the full matrix is too big — NOT DECIDED.)
5. **Cohesion and separation** (SSE-style / silhouette) — unsupervised evaluation per the syllabus.
6. **Supervised validity** against the simulator's true `role`: **purity, entropy, adjusted Rand index (ARI)**.
7. **Significance:** compare SSE/silhouette against the same features shuffled randomly to show structure is not chance.
8. Cluster profiles (mean feature values per cluster, role composition).

**Target headline:** one or more clusters contain mostly mules even though the algorithm never saw a label. This is a hope, not a guarantee — report whatever the data shows. Also discuss K-means strengths/weaknesses, K-means and different types of clusters, and K-means as an optimization problem (it minimises SSE) — all syllabus items.

---

## 9. Notebooks and report

**Notebook 01 — simulator.** Show: config and calibration sources; generation of accounts/transactions; the five scenarios and mule patterns with example graphs; noise (sleeper mules, genuine outliers, label noise); the **validation report** (section 5.3); dataset summary (counts, fraud ratio, scenario mix). Plus `make_toy_data.py` usage.
**Notebook 05 — classifiers.** Feature definitions; leakage-safe split; decision-tree overfitting curve + pruning + tree figure; leaderboard of all models; imbalance handling; metrics with the "why not accuracy" explanation; ablation; per-scenario breakdown; LR coefficients; writing `risk_scores.csv` and saving `.joblib`.
**Notebook 06 — clustering.** Section 8 in full.
Notebook filenames beyond the number: NOT DECIDED. Each notebook should be written so Person A can defend every line in a viva.

**Person A's report chapters** (title names from the plan): (1) **Data and validation** — simulator design, calibration sources, IBM AML pattern comparison, validation results, the honest-framing sentence in section 1, limitations of simulated data; (2) **Supervised models** — features, split, leakage controls, models, metrics, ablation, per-scenario results; (3) **Clustering** — method, evaluation, role-validity results. Shared chapters (limitations, ethics, conclusion) are merged in Week 6. Ethics points already identified: privacy of the scam-message contributors (B), and that wrongly freezing a genuine user's account causes real harm, which is why **precision** matters.

---

## 10. Constraints, schedule, and things easy to miss

**Stack.** Python **3.11**. pandas, NumPy, NetworkX (B), scikit-learn (includes BisectingKMeans), imbalanced-learn, matplotlib, seaborn, Streamlit and pyvis (B). Add `pyarrow` for parquet. Runs on a normal laptop; **no GPU**. Runtime limits: NOT DECIDED (aim for the full pipeline to rerun in minutes).

**Schedule (6 weeks assumed). Start date and final submission date: NOT DECIDED** (the user was asked for the deadline; no answer recorded). Person A's column and the handoffs:

| Week | Person A | Person B (context) | Handoff |
|---|---|---|---|
| 1 | Toy data; normal behaviour in the simulator; collect NPCI/RBI numbers into `config.yaml` | Scam-message Google Form; build the time-respecting graph; BFS/DFS on toy data | A→B toy data (day 1) |
| 2 | All 5 scam scenarios + mule patterns; noise | DLS, IDDFS; comparison table; PEAS | A→B first real dataset (end of week) |
| 3 | Validation vs NPCI/RBI and IBM AML; feature engineering; time-based split; `features.parquet` | A* with risk heuristic; admissibility proof; start minimax | A→B `features.parquet` |
| 4 | Decision tree (curves, pruning); KNN, Naive Bayes, logistic regression | Minimax + alpha-beta; hill climbing, SA, beam, GA | A→B `risk_scores.csv` |
| 5 | Bagging, boosting, random forest leaderboard; clustering | Label scam messages, text NB; Streamlit app | A→B saved `.joblib` models |
| 6 | Own report chapters; review B's code | Own report chapters; README; backup demo video; review A's code | Merge report; rehearse viva |

**Day 1 checklist.** Both: create repo + folder structure; write `CONTRACTS.md` with the four formats; set up `requirements.txt` (pandas, numpy, networkx, scikit-learn, imbalanced-learn, matplotlib, seaborn, streamlit, pyvis — plus pyarrow). A: commit `make_toy_data.py` and toy CSVs. B: publish the scam-message Google Form.

**Working rules.** Weekly 20-minute sync (e.g. Sunday): check the week's handoff, flag blockers, show one result each. If A's real data is late, B continues on toy data; only final numbers wait. Viva insurance: in Week 6 each person explains the other's module aloud.

**Warnings and decisions easy to miss.**
1. **Do not train on Kaggle "UPI" CSVs or PaySim.** Reasons in 5.2.
2. **Avoid a suspicious ~100% score.** Remedies named in the plan: sleeper mules, genuine outliers, label noise, leakage checks, time-based splits.
3. **Accuracy is not a valid headline metric** (fraud <1%).
4. **Split rule conflict** (time-based vs accounts-disjoint) — unresolved (section 6).
5. **Keep `role` clean.** It is the ground truth for clustering validity; label noise belongs on `is_fraud` only (unconfirmed — ask).
6. **`risk_prob` must be > 0** (A* edge cost is −log p).
7. **Statistic conflict** for FY2023-24 (₹177.05 crore vs ₹1,457 crore) — verify against the primary source.
8. **No NPCI numbers have been collected yet.** Calibration cannot be claimed until they are.
9. **Notebook numbering** differs between the implementation plan and the work plan — use the work plan.
10. **The `location` / "state mismatch" feature** has no column in the contract.
11. **Never use real people's UPI statements** — financial data and a privacy risk (the plan only collects anonymised scam *messages*, B's side).
12. **Examiner expectations beyond the project brief** (Indian scenario, dataset + model training, syllabus mapping): not known. Plan for a viva where each member explains the other's module and where "Why trust a simulated dataset?" is the most likely challenge; the answer is section 5's calibration + validation.
13. **Anything under "Person B owns" is out of scope** for you. If the dashboard needs something from you that isn't in section 3, ask Person A rather than building B's side.
