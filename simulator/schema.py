"""Shared file formats for accounts.csv and transactions.csv (see CONTRACTS.md).

The validators return a list of human-readable problems; an empty list means
the frame follows the contract. Person B can run them on anything they read.
"""

from pathlib import Path

import pandas as pd

ACCOUNT_COLUMNS = [
    "account_id",
    "account_type",
    "bank",
    "state",
    "created_at",
    "role",
    "is_fraud",
]
TRANSACTION_COLUMNS = [
    "txn_id",
    "timestamp",
    "sender_id",
    "receiver_id",
    "amount",
    "channel",
    "device_id",
    "scenario",
    "is_fraud",
]

ACCOUNT_TYPES = ("salaried", "student", "gig_worker", "small_merchant", "large_merchant")
ROLES = ("normal", "merchant", "victim", "fraudster", "mule_t1", "mule_t2", "mule_t3", "cashout")
FRAUD_ROLES = ("fraudster", "mule_t1", "mule_t2", "mule_t3", "cashout")
SCENARIOS = ("normal", "fake_kyc", "digital_arrest", "job_task", "loan_app", "fake_investment")
CHANNELS = ("P2P", "P2M")

IST = "+05:30"
IST_TZ = "Asia/Kolkata"


def _check_columns(df, expected, name):
    if list(df.columns) == expected:
        return []
    if set(df.columns) == set(expected):
        return [f"{name}: wrong column order, expected {expected}"]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    return [f"{name}: wrong columns (missing {missing}, extra {extra}); column order must be {expected}"]


def _check_allowed(df, column, allowed, name):
    bad = sorted(set(df[column].dropna()) - set(allowed))
    if df[column].isna().any():
        bad.append("<missing>")
    return [f"{name}: {column} has values outside {list(allowed)}: {bad[:5]}"] if bad else []


def _check_binary(df, column, name):
    if not df[column].isin([0, 1]).all():
        return [f"{name}: {column} must be 0 or 1"]
    return []


def _check_unique(df, column, name):
    if df[column].isna().any() or df[column].duplicated().any():
        return [f"{name}: {column} must be present and unique"]
    return []


def _check_ist(df, column, name):
    s = df[column]
    if not isinstance(s.dtype, pd.DatetimeTZDtype):
        return [f"{name}: {column} must be timezone-aware datetimes (IST, {IST})"]
    if s.isna().any():
        return [f"{name}: {column} has missing values"]
    return []


def validate_accounts(df: pd.DataFrame) -> list[str]:
    errors = _check_columns(df, ACCOUNT_COLUMNS, "accounts")
    if errors:
        return errors
    errors += _check_unique(df, "account_id", "accounts")
    errors += _check_allowed(df, "account_type", ACCOUNT_TYPES, "accounts")
    errors += _check_allowed(df, "role", ROLES, "accounts")
    errors += _check_binary(df, "is_fraud", "accounts")
    errors += _check_ist(df, "created_at", "accounts")
    return errors


def validate_transactions(tx: pd.DataFrame, accounts: pd.DataFrame) -> list[str]:
    errors = _check_columns(tx, TRANSACTION_COLUMNS, "transactions")
    if errors:
        return errors
    errors += _check_unique(tx, "txn_id", "transactions")
    known = set(accounts["account_id"])
    for column in ("sender_id", "receiver_id"):
        unknown = sorted(set(tx[column]) - known)
        if unknown:
            errors.append(f"transactions: {column} not in accounts: {unknown[:5]}")
    if (tx["sender_id"] == tx["receiver_id"]).any():
        errors.append("transactions: receiver_id must differ from sender_id")
    if not (pd.to_numeric(tx["amount"], errors="coerce") > 0).all():
        errors.append("transactions: amount must be a positive number")
    errors += _check_allowed(tx, "channel", CHANNELS, "transactions")
    errors += _check_allowed(tx, "scenario", SCENARIOS, "transactions")
    errors += _check_binary(tx, "is_fraud", "transactions")
    errors += _check_ist(tx, "timestamp", "transactions")
    return errors


def _parse_ist(values: pd.Series) -> pd.Series:
    # Parse via UTC so mixed offsets can't produce an object column, then show in IST.
    return pd.to_datetime(values, utc=True).dt.tz_convert(IST_TZ)


def read_accounts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"account_id": str})
    df["created_at"] = _parse_ist(df["created_at"])
    return df


def read_transactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"txn_id": str, "sender_id": str, "receiver_id": str, "device_id": str})
    df["timestamp"] = _parse_ist(df["timestamp"])
    return df
