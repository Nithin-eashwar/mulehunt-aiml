# MuleHunt data contracts

Shared between Person A (data & learning) and Person B (search & systems).
**Change this file only after telling the other person.** Column names and
column order below are fixed. The validators in `simulator/schema.py` enforce
them; run them on any file you produce or consume.

## 1. `data/generated/accounts.csv` (produced by A)

| Column | Type | Values |
|---|---|---|
| `account_id` | str | `ACC` + 6 digits, e.g. `ACC000123`; unique |
| `account_type` | str | `salaried, student, gig_worker, small_merchant, large_merchant` |
| `bank` | str | Indian bank name (list in `simulator/config.yaml`) |
| `state` | str | Indian state name (list in `simulator/config.yaml`) |
| `created_at` | ISO-8601 datetime with IST offset | e.g. `2024-03-02T10:15:00+05:30` |
| `role` | str | `normal, merchant, victim, fraudster, mule_t1, mule_t2, mule_t3, cashout` — ground truth, never noisy |
| `is_fraud` | int 0/1 | 1 for `fraudster, mule_t1, mule_t2, mule_t3, cashout`; victims are 0. About 1.5% of labels are deliberately flipped (label noise), prevalence unchanged |

## 2. `data/generated/transactions.csv` (produced by A)

| Column | Type | Values |
|---|---|---|
| `txn_id` | str | `TXN` + 8 digits; unique; ids ascend with time |
| `timestamp` | ISO-8601 datetime with IST offset | e.g. `2025-09-14T21:05:33+05:30` |
| `sender_id` | str | an `account_id` |
| `receiver_id` | str | an `account_id`, different from `sender_id` |
| `amount` | float | INR, > 0, 2 decimals |
| `channel` | str | `P2P` or `P2M` |
| `device_id` | str | `DEV` + 6 digits; the sender's device |
| `scenario` | str | `normal, fake_kyc, digital_arrest, job_task, loan_app, fake_investment` |
| `is_fraud` | int 0/1 | 1 for every transaction in a scam flow (victim→mule, mule→mule, mule→cash-out, investment "returns"); no label noise |

Reading: use `simulator.schema.read_accounts(path)` / `read_transactions(path)`,
which parse the datetimes as timezone-aware IST.

## 3. `data/processed/features.parquet` (produced by A)

One row per account: `account_id`, then the feature columns
(`mulehunt.features.FEATURE_COLUMNS`, in that order), then `label` (0/1, from
`accounts.csv` `is_fraud`) and `split` (`train` / `val` / `test`). Accounts are
in exactly one split.

## 4. `models/risk_scores.csv` (produced by A, end of week 4)

Columns `account_id, risk_prob`. One row per account in `features.parquet`.
`risk_prob` is clipped to `[1e-6, 1 - 1e-6]`, so `-log(risk_prob)` is always
finite (A* edge cost). Train accounts get out-of-fold scores from folds that
keep each fraud ring together; val/test accounts are scored by the model
fitted on train.

## 5. Saved models (produced by A, end of week 5)

`models/<name>.joblib`, each a dict `{"pipeline", "threshold", "feature_columns"}`
where `pipeline` has `predict_proba`. Also `models/feature_columns.json`
(ordered feature names) and `models/best_model.txt`. Load with
`mulehunt.models.load_model(name)`; top contributing features with
`mulehunt.models.explain(bundle, row)`.

The eight names are `decision_tree, logistic_regression, knn, naive_bayes,
bagging, random_forest, adaboost, gradient_boosting`. Scoring one account
for the dashboard:

```python
import pandas as pd
from mulehunt.models import explain, load_model

name = open("models/best_model.txt").read().strip()           # "random_forest"
bundle = load_model(name)
row = pd.read_parquet("data/processed/features.parquet").set_index("account_id").loc["ACC000123"]
X = row[bundle["feature_columns"]].to_frame().T.astype(float)  # one-row frame, columns in order
prob = bundle["pipeline"].predict_proba(X)[0, 1]
flagged = prob >= bundle["threshold"]                          # threshold tuned for best validation F1
print(f"{name}: risk {prob:.3f}, flagged={flagged}")
for feature, effect in explain(bundle, row):                    # top 5, largest |effect| first
    print(f"  {feature:40s} {effect:+.3f}")
```

Sign of `effect`: for `logistic_regression`, `+` pushes towards fraud. For the
other models it is importance × the row's standardised deviation from the
training mean, so `+` means "higher than a typical account" and `-` "lower".

## 6. Getting the data (no large files in git)

Git holds only: the toy CSVs in `data/generated/toy/`,
`data/generated/CHECKSUMS.txt`, `models/risk_scores.csv` and
`models/*.joblib`. Rebuild the real data yourself (the output is identical for
the same seed and the pinned `requirements.txt`):

```bash
python -m simulator.generate --config simulator/config.yaml --out data/generated
python -m mulehunt.features
```

Then compare the SHA-256 of your files with `data/generated/CHECKSUMS.txt`.

**Fallback:** when `data/generated/accounts.csv` / `transactions.csv` are
missing, code must read `data/generated/toy/accounts.csv` /
`data/generated/toy/transactions.csv` instead. The toy data (500 accounts,
3 hand-made mule chains) uses exactly these formats.
