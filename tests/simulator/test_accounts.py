import copy

import numpy as np
import pandas as pd
import pytest

from simulator.accounts import make_population
from simulator.config import load_config
from simulator.schema import ACCOUNT_COLUMNS


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config())
    c["population"]["n_accounts"] = 2000
    return c


@pytest.fixture(scope="module")
def accounts(cfg):
    return make_population(cfg, np.random.default_rng(0))


def window_start(cfg):
    return pd.Timestamp(cfg["window"]["start"], tz="Asia/Kolkata")


def test_columns(accounts):
    assert list(accounts.columns[: len(ACCOUNT_COLUMNS)]) == ACCOUNT_COLUMNS
    for col in ("primary_device", "split", "household", "profile", "shop_group"):
        assert col in accounts.columns
    assert set(accounts.role) <= {"normal", "merchant"}
    assert (accounts.is_fraud == 0).all()
    assert accounts.account_id.is_unique


def test_population_mix(cfg, accounts):
    shares = accounts.account_type.value_counts(normalize=True)
    for account_type, target in cfg["population"]["type_mix"].items():
        assert shares.get(account_type, 0) == pytest.approx(target, abs=0.02), account_type
    merchants = accounts.account_type.isin(["small_merchant", "large_merchant"])
    assert (accounts.role[merchants] == "merchant").all()
    assert (accounts.role[~merchants] == "normal").all()


def test_split_shares(cfg, accounts):
    shares = accounts.split.value_counts(normalize=True)
    for split, target in cfg["splits"]["shares"].items():
        assert shares[split] == pytest.approx(target, abs=0.01)


def test_bank_and_state_from_config(cfg, accounts):
    assert set(accounts.bank) <= set(cfg["bank_mix"])
    assert set(accounts.state) <= set(cfg["state_mix"])


def test_created_at_ranges(cfg, accounts):
    start = window_start(cfg)
    normal = accounts[accounts.role == "normal"]
    new = normal.created_at >= start
    assert new.mean() == pytest.approx(cfg["population"]["new_account_share"], abs=0.015)
    assert (normal.created_at[new] < start + pd.Timedelta(days=80)).all()
    old = accounts.created_at[accounts.created_at < start]
    assert (old <= start - pd.Timedelta(days=30)).all()
    assert (old >= start - pd.Timedelta(days=8 * 365)).all()


def test_family_shared_phones(accounts):
    shared = accounts[accounts.primary_device.duplicated(keep=False) & (accounts.shop_group < 0)]
    assert (shared.groupby("primary_device").household.nunique() == 1).all()
    assert shared.groupby("primary_device").size().between(2, 3).all()


def test_shop_phone_groups(cfg, accounts):
    # Hard negatives: a shop and its staff pay from one phone, like a mule ring does.
    hard = cfg["hard_cases"]
    groups = accounts[accounts.shop_group >= 0].groupby("shop_group")
    n_small = (accounts.account_type == "small_merchant").sum()
    assert groups.ngroups == round(hard["shop_phone_group_share"] * n_small)
    lo, hi = hard["shop_phone_group_size"]
    assert groups.size().between(lo, hi).all()
    assert (groups.primary_device.nunique() == 1).all()
    assert (groups.account_type.apply(lambda t: (t == "small_merchant").sum()) == 1).all()


def test_profiles(cfg, accounts):
    hard = cfg["hard_cases"]
    personal = accounts.role == "normal"
    collectors = accounts.profile == "collector"
    assert collectors.sum() == round(hard["money_collector_share"] * personal.sum())
    assert (accounts.created_at[collectors] < pd.Timestamp(cfg["window"]["start"], tz="Asia/Kolkata")).all()
    gig = accounts.account_type == "gig_worker"
    night = accounts.profile == "night_worker"
    assert night.sum() == round(hard["night_worker_share"] * gig.sum())
    assert (accounts.account_type[night] == "gig_worker").all()
