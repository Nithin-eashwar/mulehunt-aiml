import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from simulator.config import load_config
from simulator.generate import apply_label_noise, generate, small_config, write_outputs
from simulator.schema import FRAUD_ROLES, read_accounts, read_transactions, validate_accounts, validate_transactions


@pytest.fixture(scope="module")
def small():
    return small_config(load_config())


@pytest.fixture(scope="module")
def small_world(small):
    return generate(small)


@pytest.fixture(scope="module")
def full_world():
    return generate(load_config())


@pytest.fixture(scope="module")
def written(small_world, tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    write_outputs(*small_world, out)
    return out


def test_small_config(small):
    assert small["population"]["n_accounts"] == 5000
    assert small["fraud"]["rings_per_scenario"] == load_config()["fraud"]["small_rings_per_scenario"]


def test_outputs_pass_schema(written):
    accounts = read_accounts(written / "accounts.csv")
    tx = read_transactions(written / "transactions.csv")
    assert validate_accounts(accounts) == []
    assert validate_transactions(tx, accounts) == []


def test_deterministic(small, written, tmp_path):
    write_outputs(*generate(small), tmp_path)
    for name in ("accounts.csv", "transactions.csv", "_meta/account_split.csv", "_meta/rings.json"):
        assert (tmp_path / name).read_bytes() == (written / name).read_bytes(), name


def test_label_noise_preserves_prevalence():
    rng = np.random.default_rng(0)
    roles = np.array(["normal"] * 9000 + ["victim"] * 600 + ["mule_t1"] * 400)
    accounts = pd.DataFrame({"account_id": np.arange(len(roles)), "role": roles,
                             "is_fraud": np.isin(roles, FRAUD_ROLES).astype(int)})
    noisy = apply_label_noise(accounts, 0.015, rng)
    assert noisy.is_fraud.sum() == pytest.approx(accounts.is_fraud.sum(), abs=1)
    flipped_pos = ((accounts.is_fraud == 1) & (noisy.is_fraud == 0)).sum()
    assert flipped_pos == pytest.approx(round(0.015 * 400), abs=1)
    assert (noisy.role == accounts.role).all()
    assert (noisy.is_fraud[noisy.role == "victim"] == 0).all()


def test_fraud_share(full_world):
    accounts, tx, _ = full_world
    assert tx.is_fraud.mean() < 0.01
    assert 0.008 <= accounts.is_fraud.mean() <= 0.018


def test_label_noise_applied_to_accounts_only(full_world):
    accounts, tx, _ = full_world
    clean = accounts.role.isin(FRAUD_ROLES).astype(int)
    flips = (clean != accounts.is_fraud).sum()
    assert flips == pytest.approx(2 * round(0.015 * clean.sum()), abs=2)
    fraud_rows = tx[tx.is_fraud == 1]
    assert fraud_rows.scenario.ne("normal").all()


def test_ids_sequential_and_time_sorted(small_world):
    accounts, tx, _ = small_world
    assert list(accounts.account_id) == [f"ACC{i:06d}" for i in range(1, len(accounts) + 1)]
    assert list(tx.txn_id) == [f"TXN{i:08d}" for i in range(1, len(tx) + 1)]
    assert tx.timestamp.is_monotonic_increasing
    assert tx.device_id.str.fullmatch(r"DEV\d{6}").all()


def test_ids_do_not_reveal_fraud(full_world):
    accounts, tx, _ = full_world
    rank = pd.Series(np.arange(len(accounts)) / len(accounts), index=accounts.index)
    assert 0.4 < rank[accounts.role.isin(FRAUD_ROLES)].mean() < 0.6
    fraud_devices = tx.device_id[tx.is_fraud == 1].str[3:].astype(int)
    all_devices = tx.device_id.str[3:].astype(int)
    assert 0.3 < fraud_devices.mean() / all_devices.max() < 0.7


def test_meta_split_covers_all_accounts(small_world, written):
    accounts, _, rings = small_world
    meta = pd.read_csv(written / "_meta/account_split.csv", dtype={"ring_id": "Int64"})
    assert list(meta.columns) == ["account_id", "split", "ring_id"]
    assert meta.account_id.is_unique and set(meta.account_id) == set(accounts.account_id)
    assert set(meta.split) <= {"train", "val", "test"}
    role = accounts.set_index("account_id").role
    fraud_role = meta.account_id.map(role).isin(FRAUD_ROLES)
    assert meta.ring_id[fraud_role].notna().all()
    victims = meta.account_id.map(role) == "victim"
    assert meta.ring_id[victims].notna().all()
    assert (meta[meta.ring_id.notna()].groupby("ring_id").split.nunique() == 1).all()
    stored = json.loads((written / "_meta/rings.json").read_text())
    assert len(stored) == len(rings)
    assert {"ring_id", "scenario", "split", "start", "fraudster", "cashout", "mules", "victims"} <= set(stored[0])


def test_checksums_match_files(written):
    lines = (written / "CHECKSUMS.txt").read_text().splitlines()
    sums = dict(reversed(line.split("  ", 1)) for line in lines)
    for name in ("accounts.csv", "transactions.csv"):
        assert sums[name] == hashlib.sha256((written / name).read_bytes()).hexdigest()
