import copy

import numpy as np
import pandas as pd
import pytest

from simulator.accounts import make_population
from simulator.behaviors import festival_multiplier, generate_normal_transactions, outlier_events, sample_hours
from simulator.config import load_config
from simulator.schema import TRANSACTION_COLUMNS

IST = "Asia/Kolkata"


def small_cfg(n=2000):
    c = copy.deepcopy(load_config())
    c["population"]["n_accounts"] = n
    return c


@pytest.fixture(scope="module")
def cfg():
    return small_cfg()


@pytest.fixture(scope="module")
def world(cfg):
    rng = np.random.default_rng(0)
    accounts = make_population(cfg, rng)
    return accounts, generate_normal_transactions(accounts, cfg, rng)


def window(cfg):
    start = pd.Timestamp(cfg["window"]["start"], tz=IST)
    return start, start + pd.Timedelta(days=cfg["window"]["days"])


def test_columns_and_basic_validity(world):
    _, tx = world
    assert list(tx.columns) == [c for c in TRANSACTION_COLUMNS if c != "txn_id"]
    assert (tx.scenario == "normal").all() and (tx.is_fraud == 0).all()
    assert (tx.amount > 0).all()
    assert (tx.sender_id != tx.receiver_id).all()
    assert set(tx.channel) == {"P2P", "P2M"}


def test_all_timestamps_in_window(cfg, world):
    _, tx = world
    start, end = window(cfg)
    assert str(tx.timestamp.dt.tz) == IST
    assert tx.timestamp.min() >= start
    assert tx.timestamp.max() < end


def test_no_transactions_before_created_at(world):
    accounts, tx = world
    created = accounts.set_index("account_id").created_at
    assert (tx.timestamp >= tx.sender_id.map(created)).all()
    assert (tx.timestamp >= tx.receiver_id.map(created)).all()


def test_new_accounts_transact(cfg, world):
    accounts, tx = world
    start, _ = window(cfg)
    new = accounts.account_id[(accounts.role == "normal") & (accounts.created_at >= start)
                              & (accounts.created_at < start + pd.Timedelta(days=60))]
    assert tx.sender_id.isin(new).sum() > 0


def test_family_device_sharing(cfg, world):
    accounts, tx = world
    personal = accounts[accounts.role == "normal"]
    sizes = personal.groupby("household").size()
    multi = sizes.index[sizes >= 2]
    senders = tx.groupby("device_id").sender_id.unique()
    hh = accounts.set_index("account_id").household
    sharing = {hh[s[0]] for s in senders if 2 <= len(s) <= 3 and len({hh[x] for x in s}) == 1}
    share = len(sharing & set(multi)) / len(multi)
    assert share == pytest.approx(cfg["population"]["shared_device_household_share"], abs=0.01)


def test_salary_near_month_start(world):
    accounts, tx = world
    types = accounts.set_index("account_id").account_type
    # Merchants also make ordinary P2P transfers to people; salaries are the large ones.
    salary = tx[(tx.receiver_id.map(types) == "salaried") & (tx.sender_id.map(types) == "large_merchant")
                & (tx.channel == "P2P") & (tx.amount >= 15000)]
    assert len(salary) > 0
    assert (salary.timestamp.dt.day <= 5).mean() >= 0.80


def test_gig_payouts_go_to_workers(world):
    accounts, tx = world
    types = accounts.set_index("account_id").account_type
    p2p = tx[tx.channel == "P2P"]
    to_gig = p2p[(p2p.sender_id.map(types) == "large_merchant") & (p2p.receiver_id.map(types) == "gig_worker")]
    from_gig = p2p[(p2p.sender_id.map(types) == "gig_worker") & (p2p.receiver_id.map(types) == "large_merchant")]
    assert len(to_gig) > 5 * max(len(from_gig), 1)


def test_p2m_share(cfg, world):
    _, tx = world
    assert (tx.channel == "P2M").mean() == pytest.approx(cfg["channel_mix"]["P2M"], abs=0.03)


def test_mean_ticket_near_npci(cfg, world):
    _, tx = world
    assert tx.amount.mean() == pytest.approx(cfg["ticket_size"]["target_mean_inr"], rel=0.15)


def test_night_share_matches_hours(cfg, world):
    _, tx = world
    w = np.array(cfg["hour_of_day"], dtype=float)
    target = w[[23, 0, 1, 2, 3, 4]].sum() / w.sum()
    hours = tx.timestamp.dt.hour
    assert hours.isin([23, 0, 1, 2, 3, 4]).mean() == pytest.approx(target, abs=0.02)


def test_sample_hours_distribution(cfg):
    hours = sample_hours(200_000, cfg, np.random.default_rng(3))
    assert hours.min() >= 0 and hours.max() <= 23
    w = np.array(cfg["hour_of_day"], dtype=float)
    observed = np.bincount(hours, minlength=24) / len(hours)
    assert np.abs(observed - w / w.sum()).max() < 0.005


def test_festival_multiplier_values(cfg):
    assert festival_multiplier(pd.Timestamp("2025-10-18", tz=IST), cfg) == pytest.approx(1.047)
    assert festival_multiplier(pd.Timestamp("2025-10-25", tz=IST), cfg) == pytest.approx(0.887)
    assert festival_multiplier(pd.Timestamp("2025-09-10", tz=IST), cfg) == 1.0


def test_festival_spike():
    # The real Diwali uplift (+4.7%) is too small to see in 2,000 accounts, so this
    # checks the mechanism with a large test-only uplift; Task 7 checks calibration.
    cfg = small_cfg()
    fest = cfg["festivals"][1]
    fest["volume_uplift"] = 0.30
    rng = np.random.default_rng(5)
    accounts = make_population(cfg, rng)
    tx = generate_normal_transactions(accounts, cfg, rng)
    daily = tx[tx.channel == "P2M"].groupby(tx.timestamp.dt.date).size()
    daily.index = pd.to_datetime(daily.index)
    during = daily[fest["start"]:fest["end"]].mean()
    before = daily[pd.Timestamp(fest["start"]) - pd.Timedelta(days=10):
                   pd.Timestamp(fest["start"]) - pd.Timedelta(days=1)].mean()
    assert during >= (1 + 0.30 * 0.8) * before


def test_outliers_are_rare_and_large(cfg, world):
    accounts, tx = world
    events = outlier_events(accounts, tx, cfg, np.random.default_rng(1))
    personal = (accounts.role == "normal").sum()
    assert events.sender_id.nunique() / personal == pytest.approx(cfg["outliers"]["account_rate"], abs=0.005)
    medians = tx.groupby("sender_id").amount.median()
    assert (events.amount >= 10 * events.sender_id.map(medians).fillna(0)).all()
