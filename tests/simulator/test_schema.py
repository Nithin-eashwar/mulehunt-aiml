import pandas as pd
import pytest

from simulator.schema import (
    ACCOUNT_COLUMNS,
    TRANSACTION_COLUMNS,
    read_transactions,
    validate_accounts,
    validate_transactions,
)


def make_accounts():
    return pd.DataFrame(
        {
            "account_id": ["ACC000001", "ACC000002"],
            "account_type": ["salaried", "small_merchant"],
            "bank": ["SBI", "HDFC"],
            "state": ["Karnataka", "Maharashtra"],
            "created_at": pd.to_datetime(
                ["2024-01-05T10:00:00+05:30", "2023-06-01T09:00:00+05:30"]
            ),
            "role": ["normal", "merchant"],
            "is_fraud": [0, 0],
        }
    )[ACCOUNT_COLUMNS]


def make_transactions():
    return pd.DataFrame(
        {
            "txn_id": ["TXN00000001"],
            "timestamp": pd.to_datetime(["2025-09-14T21:05:33+05:30"]),
            "sender_id": ["ACC000001"],
            "receiver_id": ["ACC000002"],
            "amount": [250.0],
            "channel": ["P2M"],
            "device_id": ["DEV000001"],
            "scenario": ["normal"],
            "is_fraud": [0],
        }
    )[TRANSACTION_COLUMNS]


def any_error_mentions(errors, text):
    return any(text in e for e in errors)


def test_valid_frames_pass():
    accounts = make_accounts()
    assert validate_accounts(accounts) == []
    assert validate_transactions(make_transactions(), accounts) == []


def test_wrong_column_order_fails():
    cols = list(ACCOUNT_COLUMNS)
    i, j = cols.index("bank"), cols.index("state")
    cols[i], cols[j] = cols[j], cols[i]
    errors = validate_accounts(make_accounts()[cols])
    assert any_error_mentions(errors, "column order")


def test_unknown_role_fails():
    accounts = make_accounts()
    accounts.loc[0, "role"] = "boss"
    assert any_error_mentions(validate_accounts(accounts), "role")


def test_unknown_sender_fails():
    tx = make_transactions()
    tx.loc[0, "sender_id"] = "ACC999999"
    assert any_error_mentions(validate_transactions(tx, make_accounts()), "sender_id")


def test_fraud_label_matches_role():
    accounts = make_accounts()
    accounts.loc[0, "is_fraud"] = 1  # label noise on a normal account is allowed
    assert validate_accounts(accounts) == []
    accounts.loc[0, "is_fraud"] = 2
    assert any_error_mentions(validate_accounts(accounts), "is_fraud")


def test_duplicate_ids_fail():
    accounts = make_accounts()
    accounts.loc[1, "account_id"] = "ACC000001"
    assert any_error_mentions(validate_accounts(accounts), "account_id")

    accounts = make_accounts()
    tx = pd.concat([make_transactions(), make_transactions()], ignore_index=True)
    assert any_error_mentions(validate_transactions(tx, accounts), "txn_id")


def test_channel_values():
    tx = make_transactions()
    tx.loc[0, "channel"] = "NEFT"
    assert any_error_mentions(validate_transactions(tx, make_accounts()), "channel")


def test_amount_positive():
    tx = make_transactions()
    tx.loc[0, "amount"] = 0.0
    assert any_error_mentions(validate_transactions(tx, make_accounts()), "amount")


def test_sender_equals_receiver_fails():
    tx = make_transactions()
    tx.loc[0, "receiver_id"] = "ACC000001"
    assert any_error_mentions(validate_transactions(tx, make_accounts()), "receiver_id")


def test_naive_timestamps_fail():
    tx = make_transactions()
    tx["timestamp"] = tx["timestamp"].dt.tz_localize(None)
    assert any_error_mentions(validate_transactions(tx, make_accounts()), "timestamp")


def test_csv_roundtrip_keeps_ist(tmp_path):
    path = tmp_path / "transactions.csv"
    make_transactions().to_csv(path, index=False)
    tx = read_transactions(path)
    assert tx["timestamp"].dt.tz is not None
    assert tx["timestamp"].iloc[0].utcoffset() == pd.Timedelta(hours=5, minutes=30)
    assert validate_transactions(tx, make_accounts()) == []
