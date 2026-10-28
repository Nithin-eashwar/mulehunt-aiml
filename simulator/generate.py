"""Generate the real MuleHunt dataset.

    python -m simulator.generate --config simulator/config.yaml --out data/generated [--small]

Writes accounts.csv and transactions.csv in the contract formats, the internal
_meta/account_split.csv (account_id, split, ring_id) and _meta/rings.json used
for splitting and analysis (never as features), and CHECKSUMS.txt so Person B
can confirm they regenerated identical files (CONTRACTS.md section 6).
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.accounts import make_population
from simulator.behaviors import generate_normal_transactions
from simulator.config import DEFAULT_PATH, load_config
from simulator.scams import MULE_TIERS, Ring, generate_fraud
from simulator.schema import ACCOUNT_COLUMNS, TRANSACTION_COLUMNS, write_accounts, write_transactions

CHECKSUM_FILES = ("accounts.csv", "transactions.csv")


def small_config(cfg: dict) -> dict:
    small = copy.deepcopy(cfg)
    small["population"]["n_accounts"] = cfg["population"]["small_n_accounts"]
    small["fraud"]["rings_per_scenario"] = cfg["fraud"]["small_rings_per_scenario"]
    return small


def apply_label_noise(accounts: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Flip round(rate x positives) fraud labels to 0 and as many non-victim
    negatives to 1, so prevalence is unchanged and `role` stays clean (plan D9)."""
    out = accounts.copy()
    positives = np.flatnonzero(out.is_fraud.to_numpy() == 1)
    negatives = np.flatnonzero((out.is_fraud.to_numpy() == 0) & (out.role.to_numpy() != "victim"))
    k = round(rate * len(positives))
    col = out.columns.get_loc("is_fraud")
    out.iloc[rng.choice(positives, size=k, replace=False), col] = 0
    out.iloc[rng.choice(negatives, size=k, replace=False), col] = 1
    return out


def _shuffled_names(values, prefix, width, rng) -> dict:
    """Random order, so an id number never hints at when (or why) an account was made."""
    values = sorted(set(values))
    order = rng.permutation(len(values))
    return {v: f"{prefix}{i + 1:0{width}d}" for v, i in zip(values, order)}


def generate(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[Ring]]:
    rng = np.random.default_rng(cfg["seed"])
    population = make_population(cfg, rng)
    normal = generate_normal_transactions(population, cfg, rng)
    accounts, fraud, rings = generate_fraud(population, cfg, rng, normal)
    accounts = apply_label_noise(accounts, cfg["label_noise"]["rate"], rng)

    acc_name = _shuffled_names(accounts.account_id, "ACC", 6, rng)
    tx = pd.concat([normal, fraud], ignore_index=True).sort_values("timestamp", kind="stable")
    tx = tx.reset_index(drop=True)
    dev_name = _shuffled_names(tx.device_id, "DEV", 6, rng)

    accounts["account_id"] = accounts.account_id.map(acc_name)
    accounts = accounts.sort_values("account_id").reset_index(drop=True)
    tx["sender_id"] = tx.sender_id.map(acc_name)
    tx["receiver_id"] = tx.receiver_id.map(acc_name)
    tx["device_id"] = tx.device_id.map(dev_name)
    tx["txn_id"] = [f"TXN{i:08d}" for i in range(1, len(tx) + 1)]

    for ring in rings:
        ring.fraudster, ring.cashout = acc_name[ring.fraudster], acc_name[ring.cashout]
        ring.mules = {tier: [acc_name[m] for m in ring.mule_list(tier)] for tier in MULE_TIERS}
        ring.victims = [acc_name[v] for v in ring.victims]
        ring.device_of = {acc_name[a]: dev_name.get(d, d) for a, d in ring.device_of.items()}
    return accounts[ACCOUNT_COLUMNS + ["split"]], tx[TRANSACTION_COLUMNS], rings


def _ring_ids(rings) -> dict:
    ids = {}
    for ring in rings:
        for acc in [ring.fraudster, ring.cashout, *ring.victims, *sum(ring.mules.values(), [])]:
            ids[acc] = ring.ring_id
    return ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(accounts: pd.DataFrame, txns: pd.DataFrame, rings: list[Ring], out_dir) -> None:
    out = Path(out_dir)
    (out / "_meta").mkdir(parents=True, exist_ok=True)
    write_accounts(accounts, out / "accounts.csv")
    write_transactions(txns, out / "transactions.csv")

    meta = accounts[["account_id", "split"]].copy()
    meta["ring_id"] = meta.account_id.map(_ring_ids(rings)).astype("Int64")
    meta.to_csv(out / "_meta/account_split.csv", index=False, lineterminator="\n")
    stored = [{"ring_id": r.ring_id, "scenario": r.scenario, "split": r.split,
               "start": r.start.isoformat(), "duration_days": r.duration_days, "evasive": r.evasive,
               "cycle": r.cycle, "fraudster": r.fraudster, "cashout": r.cashout,
               "mules": r.mules, "victims": r.victims} for r in rings]
    (out / "_meta/rings.json").write_text(json.dumps(stored, indent=1) + "\n", encoding="utf-8", newline="\n")

    lines = [f"{_sha256(out / name)}  {name}" for name in CHECKSUM_FILES]
    (out / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def summary(accounts: pd.DataFrame, txns: pd.DataFrame, rings: list[Ring]) -> str:
    grid = pd.crosstab(pd.Series([r.scenario for r in rings], name="scenario"),
                       pd.Series([r.split for r in rings], name="split"))
    return "\n".join([
        f"accounts: {len(accounts):,}  fraud label share: {accounts.is_fraud.mean():.2%}  "
        f"victims: {(accounts.role == 'victim').sum():,}",
        f"transactions: {len(txns):,}  fraud share: {txns.is_fraud.mean():.3%}",
        f"rings: {len(rings)}  evasive: {sum(r.evasive for r in rings)}  with cycle: {sum(r.cycle for r in rings)}",
        "rings per scenario x split:", grid.to_string(),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the MuleHunt dataset")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--out", default="data/generated")
    parser.add_argument("--small", action="store_true", help="5,000-account development dataset")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.small:
        cfg = small_config(cfg)
    accounts, txns, rings = generate(cfg)
    write_outputs(accounts, txns, rings, args.out)
    print(summary(accounts, txns, rings))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
