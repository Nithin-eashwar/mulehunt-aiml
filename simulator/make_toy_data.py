"""Day-1 toy dataset: 500 accounts and 3 hand-made mule rings in the contract formats.

Throwaway by design. It lets Person B build the graph and search code before the
real simulator exists; the real data (simulator.generate) is a drop-in replacement.

    python -m simulator.make_toy_data --out data/generated/toy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.schema import ACCOUNT_COLUMNS, IST_TZ, TRANSACTION_COLUMNS, write_accounts, write_transactions

WINDOW_START = pd.Timestamp("2025-08-13", tz=IST_TZ)
WINDOW_END = WINDOW_START + pd.Timedelta(days=90)
N_NORMAL_TXNS = 3000
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Kotak"]
STATES = ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Uttar Pradesh", "Telangana"]

# (scenario, start day, member counts) — fan-out, fan-in and a plain chain.
RINGS = [
    ("fake_kyc", 20, {"victim": 1, "mule_t1": 1, "mule_t2": 3, "cashout": 1}),
    ("job_task", 45, {"victim": 6, "mule_t1": 1, "mule_t2": 1, "cashout": 1}),
    ("digital_arrest", 70, {"victim": 1, "mule_t1": 1, "mule_t2": 1, "mule_t3": 1, "cashout": 1}),
]


def _hop(rng, t):
    return t + pd.Timedelta(minutes=int(rng.integers(20, 360)))


def _ring_transactions(rng, scenario, start, m):
    """Rows (time, sender, receiver, amount) for one ring; m maps role -> account ids."""
    rows = []
    t0 = start + pd.Timedelta(hours=int(rng.integers(9, 20)))
    t1, t2 = m["mule_t1"][0], m["mule_t2"]
    if scenario == "fake_kyc":  # one large payment, fast fan-out
        amount = 95000.0
        rows.append((t0, m["victim"][0], t1, amount))
        t = _hop(rng, t0)
        for mule in t2:
            part = round(amount * 0.95 / len(t2), 2)
            rows.append((t, t1, mule, part))
            rows.append((_hop(rng, t), mule, m["cashout"][0], round(part * 0.95, 2)))
            t = t + pd.Timedelta(minutes=5)
    elif scenario == "job_task":  # many small payments gathered into one account
        t, total = t0, 0.0
        for victim in m["victim"]:
            amount = float(rng.integers(5, 50) * 100)
            rows.append((t, victim, t1, amount))
            total += amount
            t = t + pd.Timedelta(minutes=int(rng.integers(10, 60)))
        t = _hop(rng, t)
        rows.append((t, t1, t2[0], round(total * 0.95, 2)))
        rows.append((_hop(rng, t), t2[0], m["cashout"][0], round(total * 0.9, 2)))
    else:  # digital_arrest: a plain chain
        path = [m["victim"][0], t1, t2[0], m["mule_t3"][0], m["cashout"][0]]
        amount, t = 240000.0, t0
        for sender, receiver in zip(path, path[1:]):
            rows.append((t, sender, receiver, amount))
            amount, t = round(amount * 0.96, 2), _hop(rng, t)
    return rows


def make_toy_data(seed: int = 7, n_accounts: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    types = rng.choice(["salaried", "student", "gig_worker", "small_merchant", "large_merchant"],
                       size=n_accounts, p=[0.45, 0.2, 0.15, 0.15, 0.05])
    accounts = pd.DataFrame({
        "account_id": [f"ACC{i:06d}" for i in range(1, n_accounts + 1)],
        "account_type": types,
        "bank": rng.choice(BANKS, size=n_accounts),
        "state": rng.choice(STATES, size=n_accounts),
        "created_at": WINDOW_START - pd.to_timedelta(rng.integers(30, 5 * 365, size=n_accounts), unit="D"),
        "role": np.where(np.isin(types, ["small_merchant", "large_merchant"]), "merchant", "normal"),
        "is_fraud": 0,
    })
    device = {a: f"DEV{i:06d}" for i, a in enumerate(accounts.account_id, start=1)}

    # Pick ring members from personal accounts; mules and cash-outs are freshly opened.
    personal = list(rng.permutation(accounts.index[accounts.role == "normal"]))
    fraud_rows = []
    for scenario, day, counts in RINGS:
        start = WINDOW_START + pd.Timedelta(days=day)
        members = {}
        for role, k in {"fraudster": 1, **counts}.items():
            idx = [personal.pop() for _ in range(k)]
            accounts.loc[idx, "role"] = role
            accounts.loc[idx, "account_type"] = "student" if role.startswith("mule") else "salaried"
            if role != "victim":
                accounts.loc[idx, "is_fraud"] = 1
            if role.startswith("mule") or role == "cashout":
                accounts.loc[idx, "created_at"] = start - pd.Timedelta(days=int(rng.integers(3, 20)))
            members[role] = list(accounts.loc[idx, "account_id"])
        fraud_rows += [(*r, scenario) for r in _ring_transactions(rng, scenario, start, members)]

    # Normal background activity: 60% merchant payments, the rest person-to-person.
    payers = accounts.account_id[accounts.role.isin(["normal", "victim"])].to_numpy()
    merchants = accounts.account_id[accounts.role == "merchant"].to_numpy()
    senders = rng.choice(payers, size=N_NORMAL_TXNS)
    p2m = rng.random(N_NORMAL_TXNS) < 0.6
    receivers = np.where(p2m, rng.choice(merchants, size=N_NORMAL_TXNS), rng.choice(payers, size=N_NORMAL_TXNS))
    keep = senders != receivers
    seconds = rng.integers(0, 90 * 86400, size=N_NORMAL_TXNS)
    normal = pd.DataFrame({
        "timestamp": WINDOW_START + pd.to_timedelta(seconds, unit="s"),
        "sender_id": senders, "receiver_id": receivers,
        "amount": np.round(np.exp(rng.normal(5.8, 1.1, size=N_NORMAL_TXNS)) + 1, 2),
        "channel": np.where(p2m, "P2M", "P2P"), "scenario": "normal", "is_fraud": 0,
    })[keep]
    fraud = pd.DataFrame(fraud_rows, columns=["timestamp", "sender_id", "receiver_id", "amount", "scenario"])
    fraud = fraud.assign(channel="P2P", is_fraud=1)

    tx = pd.concat([normal, fraud], ignore_index=True).sort_values("timestamp", kind="stable")
    tx = tx.reset_index(drop=True)
    tx["txn_id"] = [f"TXN{i:08d}" for i in range(1, len(tx) + 1)]
    tx["device_id"] = tx.sender_id.map(device)
    return accounts[ACCOUNT_COLUMNS], tx[TRANSACTION_COLUMNS]


def write_toy_data(accounts: pd.DataFrame, tx: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_accounts(accounts, out / "accounts.csv")
    write_transactions(tx, out / "transactions.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/generated/toy")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    accounts, tx = make_toy_data(seed=args.seed)
    write_toy_data(accounts, tx, args.out)
    print(f"wrote {len(accounts)} accounts, {len(tx)} transactions "
          f"({int(tx.is_fraud.sum())} fraud) to {args.out}")


if __name__ == "__main__":
    main()
