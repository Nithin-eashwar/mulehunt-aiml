# MuleHunt Person A (Data & Learning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Person A's half of MuleHunt: a calibrated UPI transaction simulator with five scam scenarios and mule networks, a validation report, leakage-safe per-account features, a leaderboard of eight supervised fraud models, and label-free clustering checked against true roles. It also produces every handoff file Person B depends on.

**Architecture:** Two Python packages share one repo with Person B. `simulator/` turns `config.yaml` into `data/generated/accounts.csv` and `transactions.csv`. `mulehunt/features.py` turns those into `data/processed/features.parquet`. `mulehunt/models.py` and `mulehunt/clustering.py` read the parquet and write `models/*.joblib`, `models/risk_scores.csv` and figures in `reports/`. There are no notebooks: every step is a terminal command that writes its figures, tables and a Markdown summary to `reports/`, and `python -m mulehunt.pipeline` reruns everything end to end. All logic is covered by `pytest`.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn (≥1.4, for `BisectingKMeans` and `class_weight` in `HistGradientBoostingClassifier`), imbalanced-learn, matplotlib, seaborn, pyarrow, PyYAML, pytest. No Jupyter.

**Spec:** [docs/specs/MuleHunt_PersonA_Handoff.md](../specs/MuleHunt_PersonA_Handoff.md) (primary) and [docs/specs/MuleHunt_Two_Person_Work_Plan.md](../specs/MuleHunt_Two_Person_Work_Plan.md) (split and schedule). Section numbers such as "§4.3" refer to the handoff.

---

## Decisions for you to confirm (the review list)

The handoff leaves these open. The plan uses the value in the "Plan uses" column throughout. Change any row and I'll update the plan before starting. Rows marked **(B)** change the shared contract, so tell Person B.

| # | Open question (handoff §) | Plan uses | Why |
|---|---|---|---|
| D1 | ID formats (§3.1, §3.2) **(B)** | `ACC000001` (6 digits), `TXN00000001` (8 digits), `DEV000001` | Sorts as text, so it's easy to read in the graph and dashboard |
| D2 | Allowed values (§3.1, §3.2) **(B)** | `account_type`: `salaried, student, gig_worker, small_merchant, large_merchant`. `role`: `normal, merchant, victim, fraudster, mule_t1, mule_t2, mule_t3, cashout`. `scenario`: `normal, fake_kyc, digital_arrest, job_task, loan_app, fake_investment` | The handoff's proposals as written. Fraud actors use a personal `account_type` (mules are usually students or gig workers) |
| D3 | Which roles are fraud (§3.1) **(B)** | `is_fraud=1` for `fraudster, mule_t1, mule_t2, mule_t3, cashout`; victims are 0 | Victims are not complicit |
| D4 | Timestamps (§3.1, §3.2) **(B)** | ISO-8601 with IST offset, e.g. `2025-09-14T21:05:33+05:30`. `created_at` uses the same format | Unambiguous when parsed with `pd.to_datetime` |
| D5 | Simulation window (§4.1) | 90 days, **2025-08-13 → 2025-11-10**. The window is in the past, so NPCI figures for the same months exist. Dussehra (2 Oct 2025, day 51) falls in train, inside the train ring window. Diwali (20 Oct 2025, day 69) falls in validation. Verify both dates against an official 2025 holiday list | A festival spike is required and both training and evaluation should see one. A window ending in the future would look odd to an examiner |
| D6 | Size (§4.7) | 50,000 accounts and about 1.5M transactions by default (knobs in `config.yaml`). `--small` = 5,000 accounts. Toy: 500 accounts | Needed for enough rings (D7). Person B should check NetworkX copes with 1.5M edges and can develop on `--small` |
| D7 | Fraud volume (§4.2, §4.7) | **75 rings (15 per scenario)**, each with 1 fraudster, 2–4 tier-1 mules, 1–3 tier-2, 0–2 tier-3 and 1 cash-out (~8 accounts), giving about **600 fraud accounts (~1.2%)**. Rings split **50/25/25** across train/val/test, about 37/19/19 rings, so ≥ 3 per scenario per split. Fraud *transactions* stay well under 1%, which is what the "< 1%" rule is about | 40 rings left only about 6 test rings, so per-scenario recall would have been anecdotal |
| D8 | **Split conflict** (§6) | Every account belongs to exactly one split, so splits are account-disjoint. Rings **and** normal accounts are both split 50/25/25, so fraud prevalence is the same in every split. Victims follow their ring. Features use a **30-day lookback window ending on the split's last day** (train: days 31–60, val: 46–75, test: 61–90). **Each ring's whole activity sits inside its split's period and feature window:** start ∈ [first day, split end − ring duration − 3], i.e. train starts on days 31–57, val on 61–72 and test on 76–87 at the latest. Ring durations are capped at 8 days | Satisfies both rules: no account appears in two splits, and nothing used for training comes from after day 60. The start restriction stops a ring finishing before its feature window opens, which would give fraud labels to accounts with normal-looking features |
| D9 | Label noise (§4.5) | `role` stays clean. In `accounts.csv`, 1.5% of positives are flipped to 0 and the same number of random non-victim negatives are flipped to 1, which keeps prevalence the same. `transactions.csv` `is_fraud` stays clean | Flipping 1.5% of all negatives would add about 450 false frauds and swamp the 250 real ones |
| D10 | "Neighbours flagged" (§6) | Replaced by a label-free version, `neighbor_suspicious_share`: the share of an account's counterparties whose own pass-through ratio is ≥ 0.8 | Avoids label leakage completely and is easy to defend in the viva |
| D11 | Location gap (§3.2) | No contract change. `counterparty_state_mismatch_share` uses `state` from accounts.csv | Avoids a contract change for Person B |
| D12 | Feature definitions (§6) | Night = 23:00–04:59. Round amount = ≥ ₹1,000 and a multiple of ₹500. Just-below = in [90%, 100%) of ₹10k/25k/50k, the same band the evasive rings use (D14). Burstiness = (σ−μ)/(σ+μ) of the gaps between transactions (Goh–Barabási). Large receipt = ≥ ₹10,000. **Pass-through and dwell use first-in-first-out matching:** within the window, each outflow rupee is matched to the earliest unmatched inflow rupee, so no rupee is counted twice. Outflows with no inflow left to match are ignored. Pass-through = rupees matched within 24 h of their inflow ÷ total inflow. Dwell = amount-weighted mean hours from inflow to matched outflow, where unmatched rupees count the hours from inflow to window end | Each one can be explained in a sentence. FIFO stops 30 inflows "sharing" one outflow and reporting a ratio of 1.0 |
| D13 | Salary payer | Salaries come from `large_merchant` accounts acting as employers, as `P2P` transactions | The contract only allows P2P and P2M |
| D14 | Mule sourcing and evasion (§4.3, §4.5) | 30% of mules are sleepers: existing normal accounts with at least 21 days of normal history before they activate. The other 70% are newly opened 3–20 days before the ring starts. 30% of rings split payments to 90–99.9% of ₹10k/₹25k/₹50k. Hop delay is lognormal with a median of 2 h and **σ = 1.5** (about 95% under 24 h), redrawn if above 48 h | Realistic and noisy, and it keeps Person B's 48 h hop rule meaningful. At σ ≈ 1.94 exactly 90% would fall under 24 h, which would make the ≥ 90% test flaky |
| D15 | Imbalance (§7) | `class_weight="balanced"` for DT, LR, RF, bagging and gradient boosting. **AdaBoost** (depth-1 to depth-3 trees with balanced class weights) is an eighth model, since Tan–Steinbach–Kumar's "boosting" means AdaBoost. SMOTE (training only, inside an imblearn `Pipeline`) for KNN. Naive Bayes uses neither (priors are set from the data). Also an LR comparison of class weights vs SMOTE | Covers both techniques without doubling every experiment |
| D16 | Metrics extras (§7) | Recall at **1% false-positive rate**. The decision threshold maximises F1 on validation. "Statistical bounds" = 95% **ring-level (cluster) bootstrap** CI (1,000 resamples) on test PR-AUC and F1: whole rings are resampled, and normal accounts are resampled one by one | Handles the Unit III "estimating statistical bounds" item. Ring-mates are correlated, so resampling them one at a time would make the intervals look tighter than they should |
| D17 | `risk_scores.csv` (§3.4) **(B)** | From the model with the best validation PR-AUC. Train accounts get 5-fold out-of-fold scores from **`StratifiedGroupKFold` with `ring_id` as the group** (each normal account is its own group). Val and test accounts are scored by the model fitted on train. Clipped to [1e-6, 1−1e-6] | Person B's A* never sees memorised labels or log(0). Grouping stops ring-mates in other folds leaking through shared devices |
| D18 | Saved models (§3.5) **(B)** | `models/<name>.joblib`, each a full sklearn/imblearn `Pipeline` with `predict_proba`; `models/feature_columns.json`; `models/best_model.txt`; and `mulehunt.models.explain(...)` for the top features | One loading pattern for the dashboard |
| D19 | Clustering (§8) | Clustering runs on **active accounts only** (≥ 1 inflow and ≥ 1 outflow in the window). Features: the flow and network families plus `night_txn_share` and `burstiness`, with `log1p` on heavy-tailed columns and then `StandardScaler`. **K from 2 to 20.** Silhouette on a 5,000-account sample. Proximity matrix on a 2,000-account stratified sample. Shuffle baseline with 20 repeats. Besides purity, entropy and ARI, report **each cluster's mule share and lift over the base rate**, and the share of all mules that land in clusters with lift ≥ 5 | About 99% of accounts are non-fraud, so purity is high for any clustering and ARI sits near zero. Lift is the number that shows whether mules really group together |
| D20 | Validation tolerances (§5.3) | The report gives target, generated value and gap for every check. PASS if the shares are within ±3 percentage points of the target and the median ticket is within ±15%. The **IBM AML comparison is a structure check**: PASS = every IBM pattern type we implement appears in our graph. The pattern *mix* is reported side by side but not pass/fail, because our mix is set by our own config. Our "chain" has no IBM label (check the label strings in `HI-Small_Patterns.txt`), so it's reported as ours only | You can tighten these later. It would be wrong to claim a distribution match |
| D21 | Toy-data location; notebooks (§2, §9) | `simulator/make_toy_data.py`. **No notebooks for Person A** (Person A's decision, 2026-10-27). The §9 notebook contents are produced by terminal commands as `reports/a_01_simulator.md`, `reports/a_05_classifiers.md` and `reports/a_06_clustering.md`, with figures in `reports/a_figures/` (numbering kept from the work plan). Person B decides separately about notebooks 02, 03, 04 and 07 | The ML here is simple, so the terminal is enough. Markdown reports show directly on GitHub for the examiner, and nothing extra needs installing |
| D22 | Tests and imports | `tests/simulator/` and `tests/mulehunt/` (Person A owns these), and a root `pytest.ini` with `pythonpath = .` (shared file, so tell Person B) | Lets `pytest` find both packages without installing them |
| D23 | How Person B gets the data **(B)** | **Person B regenerates; nothing large goes in git.** Committed to git: the toy CSVs in `data/generated/toy/`, `data/generated/CHECKSUMS.txt` (SHA-256 of the real CSVs and parquet), `models/risk_scores.csv` and `models/*.joblib` (each < 50 MB). Not committed: the real CSVs and `features.parquet`, which Person B rebuilds with `python -m simulator.generate && python -m mulehunt.features` and checks against `CHECKSUMS.txt`. `CONTRACTS.md` says code must fall back to `data/generated/toy/` when the real files are missing | The ignored default paths meant Person B would never receive the files. Regenerating is byte-identical with pinned dependencies, and the checksums prove it. No Git LFS setup needed |
| D24 | Shortcut features | **8% of normal accounts open during the window** (days 1–80) and start slowly, so being new isn't unique to mules. **4% of normal households share one device between 2–3 family accounts**, so device sharing isn't unique to mules. Task 10 prints **single-feature PR-AUC** for every feature, and any feature above 0.9 alone is investigated | Otherwise account age or a shared device alone would nearly separate mules from normal accounts and trip the PR-AUC > 0.98 gate |

**Still needs a human:** NPCI calibration numbers (Task 3 fetches them, but you should check the values), the submission date (the schedule shifts to fit it), and the professor's view on simulated data.

---

## Global Constraints

- Python **3.11**. CPU only. The full pipeline (generate → features → models → clustering) should finish in **under 15 minutes** on a normal laptop with the default config.
- Column **names and order** in §3.1–§3.4 are fixed: `accounts.csv` = `account_id, account_type, bank, state, created_at, role, is_fraud`; `transactions.csv` = `txn_id, timestamp, sender_id, receiver_id, amount, channel, device_id, scenario, is_fraud`; `features.parquet` = `account_id, <features…>, label, split`; `risk_scores.csv` = `account_id, risk_prob`.
- `channel` ∈ {`P2P`, `P2M`}. `amount` is in INR, a float rounded to 2 decimals.
- Every random choice goes through a single `np.random.Generator` seeded from `config.yaml` `seed`. The same config must produce byte-identical CSVs.
- No feature may read `role`, `is_fraud` or `scenario` of the account being scored. SMOTE is only ever fitted on training data.
- Accuracy is never a headline metric. Report precision, recall, F1, PR-AUC and recall@1% FPR.
- Person A edits only `simulator/`, `mulehunt/features.py`, `mulehunt/models.py`, `mulehunt/clustering.py`, `mulehunt/pipeline.py`, `tests/simulator/`, `tests/mulehunt/`, `reports/a_*`, `models/`, and `data/generated|processed/`. Shared files (`CONTRACTS.md`, `requirements.txt`, `README.md`, `pytest.ini`) change only after telling Person B.
- Never use Kaggle "UPI" CSVs or PaySim as training data, and never use real people's statements.
- Always use the honest framing: *"Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark."*
- One branch and one PR per task (e.g. `a/toy-data`), reviewed by Person B.
- Git holds only the toy CSVs, `CHECKSUMS.txt`, `risk_scores.csv` and `models/*.joblib` (each < 50 MB). The real CSVs and parquet are regenerated (D23).
- A figure is cited as "NPCI" only if it comes from an NPCI page. Secondary reports are labelled as secondary, and anything without a source is `assumed`.

## Review Focus

1. **Accounts with no inflow, no outflow or no activity in their window**: features are finite (0 or the documented fill value), never NaN or inf. Tested in Task 8.
2. **`risk_prob` of exactly 0 or 1**: clipped to [1e-6, 1−1e-6], because Person B's A* takes −log p. Tested in Task 11.
3. **Toy data and real data as drop-in replacements**: both pass the same `validate_accounts`/`validate_transactions`, and timestamps survive a CSV round trip with the IST offset. Tested in Tasks 1, 2 and 6.
4. **A scenario with no positives in a split**: the simulator guarantees at least 1 ring per scenario per split (Task 5 test), and `per_scenario_table` still returns a row with `n_fraud=0` instead of crashing (Task 10 test).
5. **Time ordering of mule hops**: every forwarding hop is strictly later than the payment that funded it, and ≥ 90% are within 24 h (all within 48 h). Tested in Task 5.
6. **A ring active outside its feature window**: every fraud transaction of a ring falls inside both its split's day range and its feature window, so no labelled mule has empty features. Tested in Task 5.

---

## File Structure

```
CONTRACTS.md                 shared: the 4 formats + D1–D4, D17, D18 decisions
requirements.txt             shared: pinned deps
pytest.ini                   shared: pythonpath = .
simulator/
  __init__.py
  schema.py                  column lists, allowed values, validate_accounts/validate_transactions
  config.yaml                every calibration number with {value, source, status}
  config.py                  load_config / unsourced_keys
  make_toy_data.py           day-1 toy dataset (500 accounts, 3 chains)
  accounts.py                normal + merchant population
  behaviors.py               salary, rent, bills, P2M, family P2P, weekend/night, festival, outliers
  patterns.py                fan_out, fan_in, chain, scatter_gather, cycle, hop delays
  scams.py                   Ring, ring scheduling, the 5 scenarios, sleepers, threshold splitting
  generate.py                orchestration, label noise, ID assignment, CSV writing (CLI)
  validate.py                validation checks, IBM-AML-style pattern detection, report writer (CLI)
mulehunt/
  __init__.py
  features.py                splits, per-account features, features.parquet (CLI)
  models.py                  baseline LR for B, DT curves/pruning, model zoo, leaderboard, CIs,
                             ablation, per-scenario, risk_scores, saving, explain (CLI)
  clustering.py              sweep, K-means, bisecting, validity, proximity, shuffle baseline (CLI)
  pipeline.py                one command: generate → validate → features → models → clustering (CLI)
reports/a_01_simulator.md, a_05_classifiers.md, a_06_clustering.md   terminal-generated results
reports/a_figures/, reports/a_tables/, reports/a_validation_report.md, reports/a_chapters/
tests/simulator/…, tests/mulehunt/…
```

`patterns.py` is split from `scams.py` because the topologies (fan-out, chain…) can be tested on their own and the IBM AML comparison refers to them by name.

---

## Week 1

### Task 1: Repo skeleton, shared contract and schema validators (Day 1, with Person B)

**Files:**
- Create: `CONTRACTS.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `simulator/__init__.py`, `mulehunt/__init__.py`, `simulator/schema.py`, and empty `data/{raw,generated,processed}/.gitkeep`, `models/.gitkeep`, `reports/.gitkeep`
- Test: `tests/simulator/test_schema.py`

**Interfaces:**
- Produces: `ACCOUNT_COLUMNS: list[str]`, `TRANSACTION_COLUMNS: list[str]`, `ACCOUNT_TYPES`, `ROLES`, `FRAUD_ROLES`, `SCENARIOS`, `CHANNELS` (tuples), `validate_accounts(df: pd.DataFrame) -> list[str]`, `validate_transactions(tx: pd.DataFrame, accounts: pd.DataFrame) -> list[str]` (empty list = valid), `read_accounts(path) -> pd.DataFrame`, `read_transactions(path) -> pd.DataFrame` (both parse timestamps as tz-aware IST).

- [ ] **Step 1:** Write `CONTRACTS.md` with the four formats, the column order from Global Constraints, and decisions D1–D4, D17 and D18. Write `requirements.txt` with pinned pandas, numpy, networkx, scikit-learn>=1.4, imbalanced-learn, matplotlib, seaborn, streamlit, pyvis, pyarrow, pyyaml, pytest, jupyter, nbconvert. In `.gitignore`, ignore `data/generated/*.csv`, `data/generated/_meta/` and `data/processed/*` but **not** `data/generated/toy/` or `data/generated/CHECKSUMS.txt`. In `CONTRACTS.md`, describe the D23 route: the rebuild command, the checksum check, and that code falls back to `data/generated/toy/` when the real files are missing.
- [ ] **Step 2: Write failing tests**
  - `test_valid_frames_pass`: hand-built 2-account / 1-transaction frames → both validators return `[]`.
  - `test_wrong_column_order_fails`: swapping `bank`/`state` → an error mentioning `"column order"`.
  - `test_unknown_role_fails`: `role="boss"` → an error mentioning `"role"`.
  - `test_unknown_sender_fails`: `sender_id` not in accounts → an error mentioning `"sender_id"`.
  - `test_fraud_label_matches_role`: is_fraud=1 for a `normal` role is **allowed** (label noise), but `is_fraud ∉ {0,1}` fails.
  - `test_duplicate_ids_fail`, `test_channel_values`, `test_amount_positive`.
  - `test_csv_roundtrip_keeps_ist`: write → `read_transactions` → `timestamp.dt.tz` has a UTC offset of +05:30.
- [ ] **Step 3:** Run `pytest tests/simulator/test_schema.py -v`. Expected: FAIL (module missing).
- [ ] **Step 4:** Implement `simulator/schema.py` with the constants from D2/D3 and the validators (check columns/order, uniqueness, allowed values, FK integrity, amount > 0, sender ≠ receiver, timestamps tz-aware).
- [ ] **Step 5:** Run `pytest tests/simulator -v`. Expected: all PASS.
- [ ] **Step 6:** Commit `chore: repo skeleton, contracts and schema validators`. Open a PR for Person B to approve the contract.

### Task 2: Day-1 toy dataset

**Files:**
- Create: `simulator/make_toy_data.py`, outputs `data/generated/toy/accounts.csv`, `data/generated/toy/transactions.csv` (committed)
- Test: `tests/simulator/test_toy_data.py`

**Interfaces:**
- Consumes: `schema.*` (Task 1).
- Produces: `make_toy_data(seed: int = 7, n_accounts: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]`; CLI `python -m simulator.make_toy_data --out data/generated/toy`. The files are committed under `data/generated/toy/`, the fallback path in `CONTRACTS.md` (D23). Nothing is copied to the ignored default paths.

- [ ] **Step 1: Write failing tests**
  - `test_toy_passes_schema`: both validators return `[]`.
  - `test_toy_size`: `len(accounts) == 500`.
  - `test_three_chains`: the non-`normal` scenarios together hold exactly 3 rings. Each ring has ≥ 1 victim → mule_t1 → mule_t2 → cashout path where every hop's timestamp is later than the previous hop's and ≤ 48 h after it.
  - `test_toy_deterministic`: two calls with the same seed give equal frames.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement with hand-written ring templates (one fan-out, one fan-in, one plain chain, using scenarios `fake_kyc`, `job_task`, `digital_arrest`) on top of about 3,000 simple random normal P2P/P2M transactions over the D5 window. Keep it under about 150 lines; it is throwaway.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat(sim): day-1 toy dataset` and tell Person B it's ready.

### Task 3: Calibration config and loader

**Files:**
- Create: `simulator/config.yaml`, `simulator/config.py`, `data/raw/calibration_sources.md` (each source with its URL, access date and the exact table or figure)
- Test: `tests/simulator/test_config.py`

**Interfaces:**
- Produces: `load_config(path="simulator/config.yaml") -> dict` (nested dict of plain values, with `value` unwrapped); `load_raw_config(path) -> dict`; `unsourced_keys(raw: dict) -> list[str]` (dotted keys whose `status == "assumed"`).

Every leaf in `config.yaml` has the form `{value: …, source: "…", status: sourced|assumed}`. Required top-level groups (§4.6): `seed`, `window` (start 2025-08-13, days 90), `splits` (train [1,60], val [61,75], test [76,90]; split shares 0.50/0.25/0.25 for both rings and normal accounts; lookback_days 30; new_counterparty_history_days 30; ring_tail_days 3), `population` (n_accounts 50000, small_n_accounts 5000, type mix, new_account_share 0.08, shared_device_household_share 0.04), `bank_mix`, `state_mix`, `ticket_size` (per-behaviour lognormal median/sigma), `channel_mix` (P2P/P2M share), `hour_of_day` (24 weights), `weekend` (uplift), `festivals` (Dussehra 2025-10-02, Diwali 2025-10-20: date, uplift, duration), `behaviours` (rates per type), `outliers` (rate, amount range), `fraud` (rings_per_scenario 15, ring size ranges per D7, max_ring_days 8, sleeper_share 0.30, threshold_split_share 0.30, thresholds [10000, 25000, 50000], threshold_band [0.90, 0.999], hop_delay median_h 2, sigma 1.5, cap_h 48), `label_noise` (rate 0.015), `scenarios` (per-scenario amount and timing parameters, §4.3).

- [ ] **Step 1: Fetch calibration sources.** Fetch NPCI UPI product statistics for **Aug–Nov 2025**, the same months as the window (monthly volume and value → average ticket size). Look for a P2P vs P2M split on an NPCI page. If NPCI doesn't publish one, mark it `assumed`; a secondary report may set the value but must be cited as secondary, never as NPCI. Use RBI Payment System Indicators or the Annual Report as a cross-check, and the Lok Sabha AU2481 table (§5.1). Write the figures and URLs to `data/raw/calibration_sources.md`. Hour-of-day, festival uplift and per-type rates probably have no public source, so mark them `assumed` with a one-line rationale. Don't cite the conflicting FY24 figure (§5.1) unless the primary RBI source settles it.
- [ ] **Step 2: Write failing tests**
  - `test_every_leaf_has_source`: every leaf of `load_raw_config` has the keys `value`, `source` and `status`, and `status` ∈ {sourced, assumed}.
  - `test_mixes_sum_to_one`: `population.type_mix`, `bank_mix`, `state_mix` and `channel_mix` each sum to 1 ± 1e-9.
  - `test_hour_weights_24`: 24 non-negative weights.
  - `test_splits_cover_window`: the split windows are contiguous, cover 1–90 and don't overlap.
  - `test_unsourced_keys_lists_assumed`: a small fixture yaml with one assumed leaf → `["a.b"]`.
- [ ] **Step 3:** Run them. Expected: FAIL.
- [ ] **Step 4:** Write `config.yaml` with the fetched values and implement `config.py`.
- [ ] **Step 5:** Run them. Expected: PASS.
- [ ] **Step 6:** Commit `feat(sim): calibration config with sources`. **Person A reviews the numbers** before Task 4 uses them.

### Task 4: Normal population and behaviour

**Files:**
- Create: `simulator/accounts.py`, `simulator/behaviors.py`
- Test: `tests/simulator/test_accounts.py`, `tests/simulator/test_behaviors.py`

**Interfaces:**
- Consumes: `load_config` (Task 3), `schema` constants.
- Produces:
  - `make_population(cfg: dict, rng: np.random.Generator) -> pd.DataFrame`: the `ACCOUNT_COLUMNS` plus internal columns `primary_device: str` and `split: str`. Roles are only `normal`/`merchant` here; `is_fraud` is 0. `account_id`s are assigned later in Task 6, so this uses temporary integer ids in `account_id`.
  - `generate_normal_transactions(accounts: pd.DataFrame, cfg: dict, rng) -> pd.DataFrame`: `TRANSACTION_COLUMNS` without `txn_id`, with `scenario="normal"` and `is_fraud=0`.
  - Helpers (tested directly): `festival_multiplier(day: pd.Timestamp, cfg) -> float`, `sample_hours(n: int, cfg, rng) -> np.ndarray`, `outlier_events(accounts, cfg, rng) -> pd.DataFrame`.

- [ ] **Step 1: Write failing tests** (with a small config fixture of 2,000 accounts)
  - `test_population_mix`: the type shares are within ±2 pp of `population.type_mix`. Merchants have `role="merchant"`.
  - `test_created_at_mix`: 8% ± 1.5 pp of normal accounts open between window days 1 and 80 and have no transactions before `created_at`. The rest open 30 days to 8 years before the window.
  - `test_family_device_sharing`: 4% ± 1 pp of normal households have 2–3 accounts sending from one shared device.
  - `test_salary_near_month_start`: ≥ 80% of salaried credits fall on days 1–5 of a month, and the payer is a `large_merchant`.
  - `test_p2m_share`: the generated P2M share is within ±3 pp of `channel_mix`.
  - `test_festival_spike`: mean daily P2M volume during the festival is ≥ (1 + uplift × 0.8) × the mean of the surrounding 14 days.
  - `test_night_share_matches_hours`: the 23:00–04:59 share is within ±2 pp of the configured weights.
  - `test_outliers_are_rare_and_large`: the outlier account rate is within ±0.5 pp of config, and each outlier amount is ≥ 10× that account's median.
  - `test_all_timestamps_in_window`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Generate everything vectorised: for each behaviour, draw per-account counts with Poisson(rate × days × weekend/festival multipliers), then the days, then `sample_hours`, then lognormal amounts. P2M receivers are drawn from merchants weighted by size. Each account has 1–4 family accounts for P2P, and shared-device households (D24) use the same `primary_device`. Accounts opened mid-window start at half their type's rate for their first 14 days. Most transactions use `primary_device`; about 2% of accounts change phone once mid-window.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Run a timing check: `python -c "…"` with the default config should give about 1.5M transactions in under 90 s. If it's slower, profile before moving on.
- [ ] **Step 6:** Commit `feat(sim): normal population and behaviours`.

**Handoff check (end of Week 1):** toy data delivered, `config.yaml` reviewed, normal behaviour working.

---

## Week 2

### Task 5: Mule patterns and the five scam scenarios

**Files:**
- Create: `simulator/patterns.py`, `simulator/scams.py`
- Test: `tests/simulator/test_patterns.py`, `tests/simulator/test_scams.py`

**Interfaces:**
- Consumes: the population and normal transactions from Task 4; `cfg["fraud"]`, `cfg["scenarios"]`, `cfg["splits"]`.
- Produces:
  - `patterns.py`: `hop_delay(rng, cfg, size=None) -> np.ndarray` (hours). The functions `fan_out(src, dsts, total, t0, rng, cfg)`, `fan_in(srcs, dst, amounts, t0s, rng, cfg)`, `chain(path, amount, t0, rng, cfg)`, `scatter_gather(src, middles, dst, total, t0, rng, cfg)` and `cycle(nodes, amount, t0, rng, cfg)` each return `list[dict]` (transaction rows without txn_id). Each forwarded amount is 92–99% of what the mule received (the mule keeps a commission), and each hop is later than the payment that funded it.
  - `split_under_threshold(amount: float, thresholds: list[int], rng) -> list[float]`: parts that each sit at 90–99.9% of the chosen threshold (the last part may be smaller), summing to `amount`.
  - `scams.py`: `@dataclass Ring(ring_id: int, scenario: str, split: str, start: pd.Timestamp, fraudster: int, mules: dict[str, list[int]], cashout: int, victims: list[int], evasive: bool)`.
    `schedule_rings(cfg, rng) -> list[Ring]` (no members yet). Splits rings 50/25/25 so every scenario × split has ≥ 3 rings. Draws each ring's duration (≤ `max_ring_days`) first and then its start from [max(split start, feature-window start), split end − duration − tail] (D8). Adds `duration_days: int` to `Ring`.
    `recruit_members(rings, accounts, cfg, rng) -> pd.DataFrame` (returns the updated accounts: new mule, fraudster and cashout rows, sleepers relabelled, victims relabelled, roles set, `split` set to the ring's split for all members and victims. Mules in a ring share 1–2 devices with the fraudster).
    One function per scenario, `fake_kyc(ring, cfg, rng)`, `digital_arrest(…)`, `job_task(…)`, `loan_app(…)`, `fake_investment(…)`, each returning `list[dict]` with `scenario` set and `is_fraud=1`.
    `generate_fraud(accounts, cfg, rng) -> tuple[pd.DataFrame, pd.DataFrame, list[Ring]]`, plus light camouflage spending (a few small P2M payments) for fresh mules.

Scenario shapes (§4.3): **fake_kyc** = 1 victim, 1 large payment (₹20k–₹2L) → fast fan-out to tier-1 → chains to cash-out. **digital_arrest** = 1 victim, 3–8 large payments spread over 2–8 days → the same tier-1 mule each time → scatter-gather to cash-out. **job_task** = 15–60 victims, small payments (₹500–₹5k) → fan-in to one tier-1 → chain. **loan_app** = 10–40 victims, each making 2–5 repeated small payments over the ring's duration (≤ 8 days) → fan-in → chain. **fake_investment** = 5–20 victims, each receiving 1–3 small "returns" from the fraudster first and then making 1 large payment, all within the ring's duration → fan-out → one cycle among the tier-2 mules in 1 ring out of every 4 → cash-out. Exact ranges are in `config.yaml`.

- [ ] **Step 1: Write failing tests**
  - `test_fan_out_shape`: `fan_out(…, dsts=[1,2,3])` gives 3 rows from the same sender, the amounts sum to ≤ total, and every timestamp is > t0.
  - `test_chain_time_respecting`: in a 4-node chain, each hop is later than the previous one, amounts don't increase, and every delay is ≤ 48 h.
  - `test_cycle_returns_to_start`.
  - `test_split_under_threshold`: `split_under_threshold(60000, [10000,25000,50000], rng)` sums to 60000 ± 0.01 and every part is < its threshold.
  - `test_every_scenario_in_every_split`: `schedule_rings(default_cfg)` has ≥ 3 rings for each of the 15 scenario × split pairs, and the ring split shares are 50/25/25 ± 5 pp.
  - `test_ring_inside_feature_window` (**Review Focus 6**): after `generate_fraud`, every fraud transaction of a ring falls within its split's day range **and** its feature window (train 31–60, val 61–75, test 76–90). Train ring starts are therefore all on days ≥ 31.
  - `test_hop_timing_distribution`: over all generated fraud transactions, every forwarding hop is later than its funding payment, ≥ 90% are ≤ 24 h, and 100% are ≤ 48 h.
  - `test_sleepers_have_history`: ≥ 25% of mules have ≥ 21 days of normal transactions before their first fraud transaction.
  - `test_roles_consistent`: every `mule_t*`, `fraudster` and `cashout` account appears in fraud transactions. Victims only ever send money to mules (or receive "returns" from the fraudster in `fake_investment`).
  - `test_scenario_signatures`: job_task has a tier-1 mule with ≥ 15 distinct senders; digital_arrest has a victim with ≥ 3 fraud payments on ≥ 2 distinct days; fake_investment victims receive money before they pay.
  - `test_evasive_share`: 30% ± 10 pp of rings are `evasive`, and their victim payments fall in [90%, 99.9%] of a threshold, the same band the feature detects (D12).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `patterns.py` first, then `scams.py`.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat(sim): mule patterns and five scam scenarios`.

### Task 6: Generator, label noise and first real dataset

**Files:**
- Create: `simulator/generate.py`
- Test: `tests/simulator/test_generate.py`

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces: `generate(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[Ring]]` (final contract frames, with sequential `ACC…`/`TXN…`/`DEV…` ids assigned after sorting transactions by timestamp); `apply_label_noise(accounts, rate, rng) -> pd.DataFrame` (D9); `write_outputs(accounts, txns, rings, out_dir)`, which writes the two CSVs plus the **internal** `data/generated/_meta/account_split.csv` (`account_id, split, ring_id`, where `ring_id` is empty for accounts outside a ring and victims carry their ring's id) and `rings.json`. These are used for splitting, grouped folds and analysis, never as features. It also writes `data/generated/CHECKSUMS.txt` (SHA-256 of both CSVs). CLI: `python -m simulator.generate --config simulator/config.yaml --out data/generated [--small]`.

- [ ] **Step 1: Write failing tests** (using the `--small` config, 5,000 accounts)
  - `test_outputs_pass_schema`.
  - `test_deterministic`: two runs with the same seed give byte-identical CSVs.
  - `test_label_noise_preserves_prevalence`: the number of positives before and after is equal ± 1, the number of flipped positives is round(1.5% × positives) ± 1, `role` is unchanged, and no victim is flipped.
  - `test_fraud_share`: the transaction `is_fraud` mean is < 0.01 (the "< 1%" rule) and the account fraud share is between 0.8% and 1.8%.
  - `test_ids_sequential_and_time_sorted`.
  - `test_meta_split_covers_all_accounts`: every account has exactly one split, every fraud-role account has a `ring_id`, and all members of a ring share one split.
  - `test_checksums_match_files`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest tests/simulator -v`. Expected: all PASS.
- [ ] **Step 5:** Run the full generation `python -m simulator.generate --config simulator/config.yaml --out data/generated`. Expected: about 50k accounts, about 1.5M transactions, under 4 minutes. Print a summary (counts, fraud share, rings per scenario × split).
- [ ] **Step 6:** Commit `feat(sim): generator with label noise`, including `CHECKSUMS.txt`. **Handoff (D23):** Person B runs `python -m simulator.generate` and confirms the checksums match on their machine. If they don't match, pin the differing package versions before going further.

**Handoff check (end of Week 2):** Person B has regenerated the first real dataset and the checksums match.

---

## Week 3

### Task 7: Validation report

**Files:**
- Create: `simulator/validate.py`, outputs `reports/a_validation_report.md`, `reports/a_figures/val_*.png`
- Test: `tests/simulator/test_validate.py`

**Interfaces:**
- Consumes: the CSVs, `load_config`.
- Produces: `run_checks(accounts, txns, cfg) -> pd.DataFrame` (columns `check, target, generated, gap, passed`; covers ticket-size median and quantiles, P2P/P2M share, hour-of-day (total variation distance ≤ 0.05), fraud share, and per-scenario counts). `detect_patterns(txns, window_h=48) -> dict[str, int]` (counts of fan_out ≥ 3, fan_in ≥ 3, time-respecting chains of ≥ 3 hops, scatter_gather, cycles, restricted to fraud transactions and to all transactions separately). `compare_ibm(patterns: dict, ibm_patterns_path: str | None) -> pd.DataFrame`. `write_report(...)`. CLI: `python -m simulator.validate`.

The IBM comparison uses `HI-Small_Patterns.txt` from the IBM AML Kaggle dataset (download it manually to `data/raw/ibm_aml/`; it's small). Read the pattern label strings from the file rather than assuming them. From memory they are FAN-OUT, FAN-IN, GATHER-SCATTER, SCATTER-GATHER, CYCLE, RANDOM, BIPARTITE and STACK, with no "chain". `compare_ibm` maps our pattern names to IBM's where a match exists. **Pass/fail is presence only** (every mapped type appears in our graph). The share of each type is shown side by side for reference only, and "chain" is listed as ours with no IBM equivalent (D20). Use structure only, never amounts (§5.1). If the file is missing, the check is reported as SKIPPED, not failed.

- [ ] **Step 1: Write failing tests**
  - `test_detect_patterns_on_toy`: on the toy data, `fan_out ≥ 1`, `fan_in ≥ 1` and `chain ≥ 3` (the 3 hand-made rings).
  - `test_detect_cycle`: a synthetic 3-node cycle gives `cycle == 1`.
  - `test_chain_requires_time_order`: a path whose second hop is *earlier* than the first is not counted.
  - `test_checks_table_columns` and `test_hour_tvd_zero_for_identical`.
  - `test_ibm_missing_is_skipped`.
  - `test_ibm_mix_not_pass_fail`: a pattern mix very different from IBM's still passes as long as every mapped type is present.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement, plus figures: ticket-size histogram (log x-axis) against the target, P2P/P2M bar, hour-of-day line, pattern-share bars against IBM.
- [ ] **Step 4:** Run them. Expected: PASS. Then run the CLI on the real data and read the report. **If any check fails, adjust `config.yaml` and regenerate** before moving on (and note it in the report).
- [ ] **Step 5:** Commit `feat(sim): validation report`.

### Task 8: Features, split and features.parquet

**Files:**
- Create: `mulehunt/features.py`
- Test: `tests/mulehunt/test_features.py`

**Interfaces:**
- Consumes: the CSVs and `_meta/account_split.csv` (Task 6); `cfg["splits"]`.
- Produces:
  - `FEATURE_GROUPS: dict[str, list[str]]`:
    - `flow`: `total_inflow, total_outflow, pass_through_ratio, avg_dwell_hours`
    - `network`: `in_degree, out_degree, new_counterparty_share, fan_in_max_24h, fan_out_max_24h, neighbor_suspicious_share`
    - `behavioural`: `txn_count, night_txn_share, round_amount_share, below_threshold_share, burstiness, has_large_receipt, account_age_at_first_large_receipt_days`
    - `device_location`: `device_count, device_shared_accounts, counterparty_state_mismatch_share`
  - `FEATURE_COLUMNS: list[str]` (the groups flattened, in order). `TXN_ONLY = flow + behavioural + device_location` (for the ablation).
  - `split_windows(cfg) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]` (feature windows per D8).
  - `compute_account_features(txns, accounts, account_ids, start, end) -> pd.DataFrame` (indexed by account_id, columns = FEATURE_COLUMNS, all finite). `new_counterparty_share` uses a **fixed 30-day history**, [start − 30 days, start), for every split.
  - `fifo_match(inflows: pd.DataFrame, outflows: pd.DataFrame) -> pd.DataFrame` (one account's transactions, in time order → matched pieces with columns `amount, inflow_time, outflow_time`). Each rupee is used at most once. Pass-through and dwell are both computed from this (D12).
  - `build_feature_table(accounts, txns, splits: pd.Series, cfg) -> pd.DataFrame` (`account_id, *FEATURE_COLUMNS, label, split`; label = `is_fraud` from accounts.csv, noise included).
  - `account_scenarios(txns) -> pd.Series` (account_id → dominant non-normal scenario; for analysis only).
  - CLI `python -m mulehunt.features` writes `data/processed/features.parquet`.

Formulas follow D12. Fill values: an account with no inflow has `pass_through_ratio = 0` and `avg_dwell_hours = window hours`. Fewer than 3 transactions gives `burstiness = 0`. Without a large receipt, `account_age_at_first_large_receipt_days` is the account age at window end and `has_large_receipt = 0`.

- [ ] **Step 1: Write failing tests** (on small hand-built frames with exact expected values)
  - `test_pass_through_full`: A receives ₹10,000 at t and sends ₹9,500 at t+2 h, with the window ending at t+10 h. This gives `pass_through_ratio == 0.95` and `avg_dwell_hours == (9500·2 + 500·10)/10000 == 2.4`.
  - `test_pass_through_not_double_counted`: 30 inflows of ₹100 within one hour and one outflow of ₹100 give `pass_through_ratio == pytest.approx(1/30)`.
  - `test_fifo_order`: inflows ₹100 at t and ₹100 at t+5 h, and an outflow of ₹150 at t+6 h. The match is ₹100 from the first inflow (6 h) and ₹50 from the second (1 h).
  - `test_outflow_without_inflow_ignored`: an outflow with no earlier inflow in the window doesn't change pass-through.
  - `test_pass_through_late_outflow_ignored`: an outflow at t+30 h gives `pass_through_ratio == 0` (the rupees are matched, so dwell counts 30 h).
  - `test_fan_in_max_24h`: 5 senders within 10 h and 1 sender 3 days later give `fan_in_max_24h == 5`.
  - `test_below_threshold`: amounts [9,100, 23,000, 5,000, 49,990, 10,000] give `below_threshold_share == 0.6` (the band is [90%, 100%), so ₹10,000 itself is excluded).
  - `test_new_counterparty_history_fixed`: a counterparty last seen 31 days before `start` counts as new, and one seen 29 days before doesn't, for any split.
  - `test_round_amount`: [1,000, 1,500, 1,234, 500] give `0.5` (₹500 is below the ₹1,000 floor).
  - `test_night_share`: hours [23, 2, 12, 18] give `0.5`.
  - `test_device_shared_accounts`: two accounts sending from the same device give 1 for each.
  - `test_idle_account_is_finite`: an account with zero transactions in the window gets all finite values matching the fill rules (**Review Focus 1**).
  - `test_window_respected`: a transaction one second after `end` doesn't change any feature.
  - `test_no_label_columns_used`: shuffling `role`/`is_fraud`/`scenario` in the inputs leaves the feature matrix unchanged.
  - `test_table_layout`: the columns are exactly `["account_id", *FEATURE_COLUMNS, "label", "split"]`, each account appears once, and splits ∈ {train, val, test}.
  - `test_splits_disjoint_and_time_safe`: train rows only use transactions ≤ the end of day 60 (checked by building with a txn on day 61 added and confirming train rows don't change).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement with pandas groupby. `fifo_match` is a two-pointer walk per account (only accounts with both inflows and outflows need it, so it stays fast). Fan-in/fan-out max uses a rolling 24 h distinct count per receiver/sender.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Run the CLI on the real data. Expected: under 4 minutes, and each split's positive count is printed (about 300/150/150; at least 100 per split, otherwise raise `rings_per_scenario` and regenerate). Append the parquet's SHA-256 to `CHECKSUMS.txt`.
- [ ] **Step 6:** Commit `feat: per-account features and leakage-safe split`. **Handoff (D23):** Person B runs `python -m mulehunt.features` and checks the parquet checksum.

**Handoff check (end of Week 3):** `features.parquet` delivered and the validation report is drafted.

---

## Week 4

### Task 9: Baseline LR for Person B, and the decision tree study

**Files:**
- Create: `mulehunt/models.py` (first part)
- Test: `tests/mulehunt/test_models.py`

**Interfaces:**
- Consumes: `features.parquet`, `FEATURE_COLUMNS`.
- Produces:
  - `load_splits(path="data/processed/features.parquet", columns: list[str] | None = None) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]` (split → X, y, account_id).
  - `load_groups(account_ids: pd.Series, meta_path="data/generated/_meta/account_split.csv") -> pd.Series` (the ring_id, or the account_id itself for accounts outside a ring; used for grouped folds and the ring-level bootstrap).
  - `make_baseline_lr(random_state: int = 0) -> Pipeline` (StandardScaler + `LogisticRegression(class_weight="balanced", max_iter=1000)`). **Person B imports this** for local-search fitness, so it must fit 20k rows in under 2 s.
  - `evaluate(y_true, p, threshold: float) -> dict[str, float]` (keys `precision, recall, f1, pr_auc, recall_at_1pct_fpr`).
  - `choose_threshold(y_val, p_val) -> float` (the threshold that maximises F1).
  - `tree_depth_curve(splits, depths=range(1, 21), criterion="gini") -> pd.DataFrame` (columns `depth, train_f1, val_f1, train_pr_auc, val_pr_auc`).
  - `prune_tree(splits, max_depth: int) -> tuple[DecisionTreeClassifier, pd.DataFrame]` (cost-complexity path; picks the `ccp_alpha` with the best validation PR-AUC; the frame has `alpha, n_leaves, val_pr_auc`).
  - `export_tree_rules(tree, feature_names) -> str` (`sklearn.tree.export_text`), plus a saved figure `reports/a_figures/tree.png`.

- [ ] **Step 1: Write failing tests** (on a small synthetic parquet fixture built with `make_classification`, weights [0.97, 0.03], and the real column names)
  - `test_evaluate_perfect`: perfect scores give all metrics 1.0.
  - `test_evaluate_all_negative_predictions`: precision is 0.0 rather than a warning or crash (`zero_division=0`).
  - `test_recall_at_fpr`: on a hand-made 100-row case, the value matches a hand calculation.
  - `test_choose_threshold_in_unit_interval`.
  - `test_depth_curve_shape`: 20 rows, `train_f1` doesn't decrease with depth overall (Spearman ρ > 0.8), and the gap at depth 20 is ≥ the gap at depth 2 (overfitting shows up).
  - `test_prune_tree_fewer_leaves`: the pruned tree has ≤ the leaves of the unpruned tree at the same depth.
  - `test_baseline_lr_fast`: fitting 20k × 20 takes under 2 s.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Also add a Gini vs entropy comparison (`tree_depth_curve(criterion="entropy")`) for the syllabus item on impurity measures.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat(models): baseline LR, decision tree curves and pruning`. Tell Person B `make_baseline_lr` is importable.

### Task 10: Model zoo, leaderboard and analyses

**Files:**
- Modify: `mulehunt/models.py`
- Test: `tests/mulehunt/test_models.py`

**Interfaces:**
- Consumes: Task 9 functions; `account_scenarios` (Task 8).
- Produces:
  - `build_models(random_state: int = 0) -> dict[str, Pipeline]` with the keys `decision_tree, knn, naive_bayes, logistic_regression, bagging, adaboost, gradient_boosting, random_forest`, using the D15 imbalance handling. AdaBoost is `AdaBoostClassifier(DecisionTreeClassifier(max_depth∈{1,2,3}, class_weight="balanced"))`. KNN is an imblearn `Pipeline(StandardScaler, SMOTE, KNeighborsClassifier)` with k tuned over {3,5,…,31} on val. Gradient boosting is `HistGradientBoostingClassifier(class_weight="balanced")`. Bagging is `BaggingClassifier(DecisionTreeClassifier(class_weight="balanced"), n_estimators=100)`. Each model gets a small validation grid (≤ 12 combinations).
  - `fit_all(models, splits) -> dict[str, FittedModel]`, where `@dataclass FittedModel(name: str, pipeline, threshold: float, val_metrics: dict)`.
  - `leaderboard(fitted, splits) -> pd.DataFrame` (one row per model, with test metrics and `pr_auc_ci_low/high` and `f1_ci_low/high`, sorted by test PR-AUC). Accuracy is kept out of the table on purpose.
  - `bootstrap_ci(y, p, metric: Callable, groups: pd.Series, n: int = 1000, seed: int = 0) -> tuple[float, float]` (resamples whole groups, so ring-mates stay together, D16).
  - `single_feature_pr_auc(splits) -> pd.DataFrame` (feature, val PR-AUC using the raw feature as the score, flipped if that's higher; sorted descending). Shortcut check (D24).
  - `imbalance_comparison(splits) -> pd.DataFrame` (LR: none vs class_weight vs SMOTE).
  - `ablation(splits_all, splits_txn_only, model_name="random_forest") -> pd.DataFrame` (rows `transaction_only`, `with_network`).
  - `per_scenario_table(fitted_model, splits, scenarios: pd.Series) -> pd.DataFrame` (columns `scenario, n_fraud, recall, mean_risk_prob`; one row per scenario even if `n_fraud == 0`).
  - `lr_coefficients(fitted_lr) -> pd.DataFrame` (feature, coefficient, odds_ratio, sorted by |coefficient|).
  - Figures: PR curves for all models on one plot, and the leaderboard as a bar chart.

- [ ] **Step 1: Write failing tests** (fixture parquet)
  - `test_build_models_keys`: exactly the 8 keys above.
  - `test_all_models_predict_proba`: each fitted pipeline returns probabilities in [0, 1] of shape (n, 2).
  - `test_smote_only_on_train`: the KNN pipeline's SMOTE step is only fitted during `fit` (assert the number of val rows scored == the number of val rows, and monkeypatch-count `SMOTE.fit_resample` calls == 1 per fit, on training-sized input).
  - `test_leaderboard_no_accuracy`: `"accuracy" not in leaderboard.columns`, 8 rows, sorted by PR-AUC.
  - `test_bootstrap_ci_brackets_point`: low ≤ point estimate ≤ high.
  - `test_grouped_bootstrap_wider`: on a fixture where positives come in groups of 8 with identical scores, the grouped CI is wider than the CI from treating every row as its own group.
  - `test_single_feature_pr_auc_rows`: one row per feature in FEATURE_COLUMNS, with values in [0, 1].
  - `test_per_scenario_empty_scenario`: a scenario with no test positives still gets a row with `n_fraud == 0` and `recall` NaN (**Review Focus 4**).
  - `test_ablation_rows`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS. Then run on the real data. **Sanity gate (§5.3 check 6):** print `single_feature_pr_auc` first. If any single feature scores > 0.9, or any model's test PR-AUC is > 0.98, stop and find the shortcut or leak (the single-feature table, feature importances, `test_no_label_columns_used`). Then fix the simulator so the shortcut isn't unique to mules (D24), or raise the noise. Don't continue until the cause is understood.
- [ ] **Step 5:** Commit `feat(models): model zoo, leaderboard, CIs, ablation, per-scenario`.

### Task 11: risk_scores.csv, saved models and explanations

**Files:**
- Modify: `mulehunt/models.py` (add the CLI `python -m mulehunt.models --all`)
- Test: `tests/mulehunt/test_models.py`

**Interfaces:**
- Produces:
  - `write_risk_scores(best: FittedModel, splits, path="models/risk_scores.csv", n_folds: int = 5) -> pd.DataFrame` (D17).
  - `save_models(fitted: dict[str, FittedModel], out_dir="models") -> None` (writes `<name>.joblib` (pipeline + threshold in a dict `{"pipeline", "threshold", "feature_columns"}`), `feature_columns.json` and `best_model.txt`).
  - `load_model(name: str, models_dir="models") -> dict`.
  - `explain(model_bundle: dict, x_row: pd.Series, top_k: int = 5) -> list[tuple[str, float]]` (trees: `feature_importances_` × the row's standardised deviation from the training mean; LR: coefficient × standardised value; other models: permutation-free fallback to the RF importances, noted in the docstring).

- [ ] **Step 1: Write failing tests**
  - `test_risk_scores_columns_and_coverage`: the columns are exactly `["account_id", "risk_prob"]` and there is one row per account in features.parquet.
  - `test_risk_scores_clipped`: a model that outputs exactly 0 and 1 gives a min ≥ 1e-6 and a max ≤ 1−1e-6, and `-np.log(risk_prob)` is finite everywhere (**Review Focus 2**).
  - `test_train_scores_are_out_of_fold`: train-row scores differ from in-sample `predict_proba` for at least some rows, and the fold count is 5.
  - `test_oof_folds_keep_rings_together`: in the fold assignment `write_risk_scores` uses, every ring's members are in the same fold (`StratifiedGroupKFold`, groups from `load_groups`).
  - `test_save_load_roundtrip`: a loaded bundle predicts identically to the original.
  - `test_explain_returns_top_k_known_features`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS. Run `python -m mulehunt.models --all` on the real data.
- [ ] **Step 5:** Commit `feat(models): risk scores, saved models, explanations`. **Handoff:** `risk_scores.csv` goes to Person B (end of Week 4); `.joblib` + `feature_columns.json` follow in Week 5 once the leaderboard is final.

**Handoff check (end of Week 4):** `risk_scores.csv` delivered, so Person B's A* drops the placeholder.

---

## Week 5

### Task 12: Clustering

**Files:**
- Create: `mulehunt/clustering.py`
- Test: `tests/mulehunt/test_clustering.py`

**Interfaces:**
- Consumes: `features.parquet`; `role` from accounts.csv (for validity only).
- Produces:
  - `CLUSTER_FEATURES: list[str]` (D19); `LOG_FEATURES: list[str]` (the heavy-tailed subset).
  - `active_accounts(df: pd.DataFrame) -> pd.DataFrame` (rows with `total_inflow > 0` and `total_outflow > 0`, D19).
  - `prepare(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]`.
  - `kmeans_sweep(X, ks=range(2, 21), seed=0, silhouette_sample=5000) -> pd.DataFrame` (`k, sse, silhouette`).
  - `cluster_lift(labels, is_mule: pd.Series) -> pd.DataFrame` (cluster, size, mule_share, lift = mule_share ÷ overall mule share; sorted by lift) and `mule_capture(lift_df, min_lift=5.0) -> float` (the share of all mules in clusters with lift ≥ min_lift). "Mule" here means role ∈ {mule_t1, mule_t2, mule_t3}.
  - `fit_kmeans(X, k, seed=0) -> KMeans`; `fit_bisecting(X, k, seed=0) -> BisectingKMeans`.
  - `cohesion_separation(X, labels) -> dict[str, float]` (`sse`, `ssb`, `silhouette`).
  - `purity(true, pred) -> float`; `cluster_entropy(true, pred) -> float`; and ARI via `sklearn.metrics.adjusted_rand_score`.
  - `proximity_matrix_figure(X, labels, n_sample=2000, seed=0, path=…) -> None` (stratified sample, sorted by cluster).
  - `shuffled_baseline(X, k, n_repeats=20, seed=0) -> pd.DataFrame` (each column permuted independently; `sse, silhouette`).
  - `cluster_profiles(df, labels, roles) -> pd.DataFrame` (mean features + role composition per cluster).
  - CLI `python -m mulehunt.clustering` writes tables and figures to `reports/a_tables/` and `reports/a_figures/`.

- [ ] **Step 1: Write failing tests**
  - `test_purity_perfect`: `purity(["a","a","b","b"], [0,0,1,1]) == 1.0`.
  - `test_purity_half`: `purity(["a","b","a","b"], [0,0,1,1]) == 0.5`.
  - `test_entropy_zero_for_pure`.
  - `test_sweep_finds_blobs`: on `make_blobs(centers=4)`, the best silhouette is at k = 4 and SSE decreases with k.
  - `test_shuffle_destroys_structure`: on blobs, the mean shuffled silhouette is < the real silhouette − 0.2.
  - `test_cluster_without_labels`: `prepare` raises if passed a frame containing `label`, `role` or `is_fraud` among the selected columns (guards against accidental label use).
  - `test_bisecting_k_clusters`: it returns exactly k distinct labels.
  - `test_cluster_lift`: 100 rows with 10 mules, where cluster 0 (20 rows) holds 8 of them, give cluster 0 a `mule_share == 0.4` and `lift == 4.0`. `mule_capture(min_lift=3) == 0.8`.
  - `test_active_accounts_filter`: rows with zero inflow or zero outflow are dropped.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS. Then run the CLI on the real data. The headline table is per-cluster lift and mule capture. Purity and ARI are reported with a note that a 99% majority class inflates purity and flattens ARI. Report honestly whether any cluster is mostly mules (§8 "hope, not a guarantee").
- [ ] **Step 5:** Commit `feat: clustering with validity measures`.

### Task 13: Final models handoff

- [ ] **Step 1:** Rerun `python -m mulehunt.models --all` with the final config. Check that `models/` holds 8 `.joblib` files (each < 50 MB; if the random forest is bigger, cap `max_depth` or `n_estimators` and note it), `feature_columns.json`, `best_model.txt` and `risk_scores.csv`.
- [ ] **Step 2:** Write a 10-line loading snippet into `CONTRACTS.md` (§3.5) showing Person B how to call `load_model` and `explain`. **Tell Person B first** (shared file).
- [ ] **Step 3:** Commit `chore: final models for dashboard`. **Handoff:** the `.joblib` models go to Person B.

### Task 14: One-command pipeline and results reports (replaces notebooks, D21)

**Files:**
- Create: `mulehunt/pipeline.py`, `mulehunt/report.py`
- Test: `tests/mulehunt/test_pipeline.py`

**Interfaces:**
- Consumes: the CLIs and functions of Tasks 6–8 and 10–12.
- Produces: `write_markdown_report(path: str, title: str, sections: list[tuple[str, str | pd.DataFrame | Path]]) -> None` in `mulehunt/report.py` (a section body is prose, a DataFrame rendered with `to_markdown`-style pipes without needing `tabulate`, or a figure path embedded as `![](...)`). `run_pipeline(config_path="simulator/config.yaml", small: bool = False, stages: list[str] | None = None) -> dict[str, float]` (stage → seconds). CLI: `python -m mulehunt.pipeline [--small] [--stages generate,validate,features,models,clustering]`, which prints one progress line per stage and a final summary table to the terminal.

The three reports hold what §9 asked the notebooks to show, written in viva language:
- **`reports/a_01_simulator.md`:** config and sources table (with `unsourced_keys` listed honestly as "assumed"), toy-data usage, generation summary, one example subgraph figure per scenario (NetworkX drawing of one ring), noise examples (one sleeper timeline, one genuine outlier), and the validation tables and figures from Task 7.
- **`reports/a_05_classifiers.md`:** feature definitions table, the split diagram (D8), depth curve and Gini vs entropy, pruning path and tree figure ("mule rulebook"), a "why not accuracy" table that puts the all-negative baseline's accuracy (about 99%) next to its recall (0), the single-feature PR-AUC table, imbalance comparison, leaderboard of 8 models with ring-level CIs, PR curves, LR coefficients, ablation, per-scenario table, and a note that `risk_scores.csv` was written.
- **`reports/a_06_clustering.md`:** §8 items 1–8 in order on active accounts, with the cluster-lift table as the headline, plus short sections on K-means as SSE minimisation, its strengths and weaknesses, and its behaviour with different cluster types.

- [ ] **Step 1: Write failing tests**
  - `test_markdown_report_renders_table_and_figure`: a section with a 2×2 DataFrame produces a pipe table with a header separator row, and a section with a figure path produces `![`.
  - `test_pipeline_small_end_to_end` (marked `slow`): `run_pipeline(small=True)` in a temp copy of the output folders creates `a_01_simulator.md`, `a_05_classifiers.md`, `a_06_clustering.md`, `risk_scores.csv` and at least one `.joblib`, and every figure a report links to exists.
  - `test_pipeline_stage_subset`: `stages=["generate", "validate"]` runs only those two stages.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Each stage's CLI calls `write_markdown_report` for its own report, so the reports also come out when stages run one by one.
- [ ] **Step 4:** Run them. Expected: PASS. Then run `python -m mulehunt.pipeline` with the default config. Expected: completes without errors in under 15 minutes in total, and the three reports render correctly on GitHub.
- [ ] **Step 5:** Commit `feat: one-command pipeline and results reports` (reports and figures included, so the examiner can read them without running anything).

**Handoff check (end of Week 5):** the `.joblib` models are delivered, and `python -m mulehunt.pipeline` reruns everything and regenerates the three reports.

---

## Week 6

### Task 15: Report chapters and review

**Files:**
- Create: `reports/a_chapters/01_data_and_validation.md`, `02_supervised_models.md`, `03_clustering.md`

- [ ] **Step 1:** Draft the three chapters from the `reports/a_0*.md` results (§9 lists the required contents of each). Include the honest-framing sentence, limitations of simulated data, every `assumed` config key, and the precision/false-freeze ethics point.
- [ ] **Step 2:** Run the full pipeline from scratch (`generate → validate → features → models → clustering`) and check that the numbers in the chapters match the outputs.
- [ ] **Step 3:** Review Person B's PRs, and write your own one-page "explain B's module" notes for the viva.
- [ ] **Step 4:** Commit `docs: Person A report chapters`.

---

## Self-review notes

- **Spec coverage:** §3 → Tasks 1, 2, 11, 13. §4.1 → Task 4. §4.2–4.5 → Tasks 5, 6. §4.6 → Task 3. §4.7 → D6/D7 and Task 6 tests. §5 → Tasks 3, 7, and the sanity gate in 10. §6 → Task 8. §7 → Tasks 9–11. §8 → Task 12. §9 → Tasks 14, 15. §10 day-1 checklist → Tasks 1, 2. Weekly handoffs → the checks at the end of each week.
- **Not covered on purpose (Person B's scope):** graph, search, minimax, local search, scam text, dashboard.
- **Name consistency:** `FEATURE_COLUMNS`, `TXN_ONLY`, `FittedModel`, `load_model`, `explain`, `make_baseline_lr`, `account_scenarios`, `load_groups`, `fifo_match` and `Ring` are used with the same signatures everywhere.

## Review changelog (rev 2, 2026-10-27 review)

| Flag | Verdict | Change |
|---|---|---|
| 1. Train rings invisible to their features | Agreed | D8: ring starts and full activity confined to the feature window. Durations capped at 8 days so val/test rings fit. New test `test_ring_inside_feature_window` |
| 2. Pass-through double-counts | Agreed | D12: FIFO matching via `fifo_match`. Dwell is redefined on the same matches. New tests for double counting, FIFO order, and outflows with no inflow |
| 3. Test set only about 6 rings | Agreed, with a corrected count | D6/D7: 50k accounts and 75 rings with a 50/25/25 split. 75 × ~8 is about 600 fraud accounts, which is **~1.2%**, not the 1.6% the review estimated. The test range is now 0.8–1.8%. **Added:** normal accounts also split 50/25/25, so prevalence is the same in every split, and the CIs use a ring-level bootstrap (D16), since more rings alone doesn't fix correlated ring-mates |
| 4. Shortcut features | Agreed | D24: 8% of normal accounts open mid-window and 4% of households share a device. `single_feature_pr_auc` added to Task 10 and to the sanity gate |
| 5. Out-of-fold scores not grouped by ring | Agreed | D17: `StratifiedGroupKFold` on `ring_id`. `_meta/account_split.csv` now carries `ring_id`. New fold test |
| 6. Person B never receives the files | Agreed | D23: Person B regenerates and checks `CHECKSUMS.txt`. The toy fallback path is in `CONTRACTS.md`. The small outputs (risk scores, models) are committed. Git LFS not needed |
| 7. Clustering metrics mislead | Agreed | D19: active accounts only, K 2–20, cluster lift and mule capture as the headline. Purity/ARI kept with a caveat |
| Hop delay flaky | Agreed (σ ≈ 1.94 gives exactly 90%) | σ = 1.5, about 95% under 24 h |
| Evasion band vs feature band | Agreed | Both are [90%, 100%) |
| New-counterparty history varies | Agreed | Fixed 30-day history and a test for it |
| IBM comparison | Agreed | Presence is pass/fail, the mix is descriptive only, and the label strings are read from the file |
| AdaBoost | Agreed | Eighth model |
| Window in the future | Agreed | 2025-08-13 → 2025-11-10. Dussehra on day 51 and Diwali on day 69 check out from the start date, but the holiday dates themselves still need checking |
| NPCI P2P/P2M | Agreed | Mark it `assumed` when there's no NPCI source, and label secondary sources as secondary (Global Constraints) |

**Rev 3 (2026-10-27, Person A):** no notebooks and no Jupyter. Task 14 is now a one-command pipeline that writes the §9 content as Markdown reports in `reports/` (D21). `requirements.txt` no longer lists `jupyter`, `ipykernel` or `nbconvert`.
