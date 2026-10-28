"""Per-account features for the fraud models (handoff §6, plan D8/D10/D12).

    python -m mulehunt.features [--data data/generated] [--out data/processed/features.parquet]

Each account belongs to one split and is described by its transactions in that
split's 30-day feature window [start, end): train days 31-60, val 46-75, test
61-90. Nothing outside the window is used, except a fixed 30-day history before
it for `new_counterparty_share`. No feature reads role, is_fraud or scenario.

Definitions (D12):
  pass_through_ratio  rupees sent on within 24 h of arriving / rupees received, with
                      outflows matched to inflows first-in-first-out (each rupee once)
  avg_dwell_hours     amount-weighted hours from arrival to the matched outflow;
                      unmatched rupees count the hours until the window ends
  night_txn_share     share of transactions between 23:00 and 04:59 IST
  round_amount_share  share of amounts >= Rs 1,000 that are a multiple of Rs 500
  below_threshold_share  share of amounts in [90%, 100%) of Rs 10k / 25k / 50k
  burstiness          (sd - mean) / (sd + mean) of gaps between transactions (Goh-Barabasi);
                      0 with fewer than 3 transactions
  neighbor_suspicious_share  share of counterparties whose own pass-through >= 0.8 (label-free)
Behavioural shares count every transaction the account is part of, sent or received.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.config import DEFAULT_PATH, load_config
from simulator.schema import IST_TZ, read_accounts, read_transactions
from simulator.validate import _max_distinct_in_window

FEATURE_GROUPS = {
    "flow": ["total_inflow", "total_outflow", "pass_through_ratio", "avg_dwell_hours"],
    "network": ["in_degree", "out_degree", "new_counterparty_share", "fan_in_max_24h", "fan_out_max_24h",
                "neighbor_suspicious_share"],
    "behavioural": ["txn_count", "night_txn_share", "round_amount_share", "below_threshold_share", "burstiness",
                    "has_large_receipt", "account_age_at_first_large_receipt_days"],
    "device_location": ["device_count", "device_shared_accounts", "counterparty_state_mismatch_share"],
}
FEATURE_COLUMNS = [c for group in FEATURE_GROUPS.values() for c in group]
# One line each, for the reports (the formulas are in the module docstring).
FEATURE_DESCRIPTIONS = {
    "total_inflow": "Rupees received in the window",
    "total_outflow": "Rupees sent in the window",
    "pass_through_ratio": "Rupees sent on within 24 h of arriving / rupees received (FIFO matching, each rupee once)",
    "avg_dwell_hours": "Amount-weighted hours money stays before it is sent on; unsent money counts until window end",
    "in_degree": "Distinct accounts that sent money to this account",
    "out_degree": "Distinct accounts this account sent money to",
    "new_counterparty_share": "Share of counterparties not seen in the 30 days before the window",
    "fan_in_max_24h": "Most distinct senders within any 24 h",
    "fan_out_max_24h": "Most distinct receivers within any 24 h",
    "neighbor_suspicious_share": "Share of counterparties whose own pass-through is >= 0.8 (label-free)",
    "txn_count": "Transactions sent or received",
    "night_txn_share": "Share of transactions between 23:00 and 04:59 IST",
    "round_amount_share": "Share of amounts >= Rs 1,000 that are a multiple of Rs 500",
    "below_threshold_share": "Share of amounts in [90%, 100%) of Rs 10k / 25k / 50k",
    "burstiness": "(sd - mean) / (sd + mean) of gaps between transactions; 0 with fewer than 3",
    "has_large_receipt": "1 if the account received a single payment >= Rs 10,000",
    "account_age_at_first_large_receipt_days": "Account age in days at its first receipt >= Rs 10,000 (age at window "
                                               "end if none)",
    "device_count": "Distinct devices the account sent from",
    "device_shared_accounts": "Other accounts that sent from any of the same devices",
    "counterparty_state_mismatch_share": "Share of counterparties registered in a different state",
}
TXN_ONLY = FEATURE_GROUPS["flow"] + FEATURE_GROUPS["behavioural"] + FEATURE_GROUPS["device_location"]

SPLITS = ("train", "val", "test")
PASS_THROUGH_HOURS = 24
SUSPICIOUS_PASS_THROUGH = 0.8
NIGHT_HOURS = (23, 0, 1, 2, 3, 4)
ROUND_FLOOR, ROUND_STEP = 1000, 500
THRESHOLDS = (10000, 25000, 50000)
THRESHOLD_BAND = 0.90
LARGE_RECEIPT = 10000
FAN_WINDOW_H = 24
HISTORY_DAYS = 30
OUT_PATH = Path("data/processed/features.parquet")


def split_windows(cfg: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Half-open feature window per split: the lookback days ending on the split's last day."""
    start = pd.Timestamp(cfg["window"]["start"], tz=IST_TZ)
    days = cfg["splits"]["lookback_days"]
    return {s: (start + pd.Timedelta(days=cfg["splits"][s][1] - days), start + pd.Timedelta(days=cfg["splits"][s][1]))
            for s in SPLITS}


# ---------------------------------------------------------------- FIFO matching

def _fifo(in_t, in_a, out_t, out_a):
    """Match outflows to earlier inflows first-in-first-out. Returns matched pieces
    (amount, inflow index, outflow time) and the unmatched rupees left on each inflow."""
    remaining = np.array(in_a, dtype=float)
    pieces, i = [], 0
    for t, amount in zip(out_t, out_a):
        while amount > 1e-9 and i < len(in_t) and in_t[i] < t:
            take = min(remaining[i], amount)
            if take > 0:
                pieces.append((take, i, t))
            remaining[i] -= take
            amount -= take
            if remaining[i] <= 1e-9:
                remaining[i] = 0.0
                i += 1
    return pieces, remaining


def fifo_match(inflows: pd.DataFrame, outflows: pd.DataFrame) -> pd.DataFrame:
    """One account's inflows and outflows (timestamp, amount) -> matched pieces
    (amount, inflow_time, outflow_time); each rupee is used at most once."""
    inflows = inflows.sort_values("timestamp", kind="stable")
    outflows = outflows.sort_values("timestamp", kind="stable")
    pieces, _ = _fifo(inflows.timestamp.to_list(), inflows.amount.to_numpy(),
                      outflows.timestamp.to_list(), outflows.amount.to_numpy())
    in_times = inflows.timestamp.to_list()
    return pd.DataFrame({"amount": [p[0] for p in pieces], "inflow_time": [in_times[p[1]] for p in pieces],
                         "outflow_time": [p[2] for p in pieces]})


def _flow_features(inflow, outflow, end_s):
    """pass_through_ratio and avg_dwell_hours for every account that received money."""
    day = PASS_THROUGH_HOURS * 3600
    inflow = inflow.sort_values(["account", "t"], kind="stable")
    outflow = outflow.sort_values(["account", "t"], kind="stable")
    out_groups = {a: (g.t.to_numpy(), g.amount.to_numpy()) for a, g in outflow.groupby("account", sort=False)}
    accounts, passed, dwell = [], [], []
    for account, g in inflow.groupby("account", sort=False):
        in_t, in_a = g.t.to_numpy(), g.amount.to_numpy()
        total = in_a.sum()
        out_t, out_a = out_groups.get(account, (np.array([]), np.array([])))
        pieces, remaining = _fifo(in_t, in_a, out_t, out_a)
        fast = sum(take for take, i, t in pieces if t - in_t[i] <= day)
        held = sum(take * (t - in_t[i]) for take, i, t in pieces) + (remaining * (end_s - in_t)).sum()
        accounts.append(account)
        passed.append(fast / total if total > 0 else 0.0)
        dwell.append(held / total / 3600 if total > 0 else np.nan)
    return pd.DataFrame({"pass_through_ratio": passed, "avg_dwell_hours": dwell}, index=accounts)


# ---------------------------------------------------------------- features

def _seconds(ts: pd.Series) -> np.ndarray:
    return ts.dt.tz_convert("UTC").dt.as_unit("s").astype("int64").to_numpy()


def _events(tx: pd.DataFrame) -> pd.DataFrame:
    """Each transaction twice: once for the sender (out) and once for the receiver (in)."""
    out = pd.DataFrame({"account": tx.sender_id, "other": tx.receiver_id, "out": True})
    inn = pd.DataFrame({"account": tx.receiver_id, "other": tx.sender_id, "out": False})
    shared = {"t": tx.t.to_numpy(), "amount": tx.amount.to_numpy(), "hour": tx.hour.to_numpy(),
              "device_id": tx.device_id.to_numpy()}
    return pd.concat([out.assign(**shared), inn.assign(**shared)], ignore_index=True)


def _fan_max(keys, others, t):
    if len(keys) == 0:
        return pd.Series(dtype=float)
    codes, names = pd.factorize(pd.concat([pd.Series(keys), pd.Series(others)], ignore_index=True))
    k, o = codes[: len(keys)], codes[len(keys):]
    best = _max_distinct_in_window(k, o, np.asarray(t), FAN_WINDOW_H * 3600)
    return pd.Series({names[c]: v for c, v in best.items()})


def _burstiness(ev):
    ev = ev.sort_values(["account", "t"], kind="stable")
    gaps = ev.t.diff().where(ev.account.eq(ev.account.shift()))
    g = gaps.groupby(ev.account)
    mean, sd, n = g.mean(), g.std(ddof=0), ev.groupby("account").size()
    b = ((sd - mean) / (sd + mean)).where((n >= 3) & ((sd + mean) > 0), 0.0)
    return b.fillna(0.0)


def compute_account_features(txns: pd.DataFrame, accounts: pd.DataFrame, account_ids, start: pd.Timestamp,
                             end: pd.Timestamp) -> pd.DataFrame:
    account_ids = list(account_ids)
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    window_hours = (end_s - start_s) / 3600
    all_t = _seconds(txns.timestamp)
    in_window = (all_t >= start_s) & (all_t < end_s)
    tx = txns[in_window].assign(t=all_t[in_window], hour=txns.timestamp[in_window].dt.tz_convert(IST_TZ).dt.hour)
    ev = _events(tx)
    inflow, outflow = ev[~ev.out], ev[ev.out]
    by = ev.groupby("account")
    f = pd.DataFrame(index=pd.Index(account_ids, name="account_id"))

    f["total_inflow"] = inflow.groupby("account").amount.sum()
    f["total_outflow"] = outflow.groupby("account").amount.sum()
    flows = _flow_features(inflow, outflow, end_s)
    f["pass_through_ratio"] = flows.pass_through_ratio
    f["avg_dwell_hours"] = flows.avg_dwell_hours

    f["in_degree"] = inflow.groupby("account").other.nunique()
    f["out_degree"] = outflow.groupby("account").other.nunique()
    pairs = ev[["account", "other"]].drop_duplicates()
    hist_mask = (all_t >= start_s - HISTORY_DAYS * 86400) & (all_t < start_s)
    hist = txns[hist_mask]
    seen = pd.concat([pd.DataFrame({"account": hist.sender_id, "other": hist.receiver_id}),
                      pd.DataFrame({"account": hist.receiver_id, "other": hist.sender_id})]).drop_duplicates()
    seen["seen"] = True
    pairs = pairs.merge(seen, on=["account", "other"], how="left")
    f["new_counterparty_share"] = pairs.seen.isna().groupby(pairs.account).mean()
    f["fan_in_max_24h"] = _fan_max(inflow.account.to_numpy(), inflow.other.to_numpy(), inflow.t.to_numpy())
    f["fan_out_max_24h"] = _fan_max(outflow.account.to_numpy(), outflow.other.to_numpy(), outflow.t.to_numpy())
    suspicious = pairs.other.map(flows.pass_through_ratio).fillna(0.0) >= SUSPICIOUS_PASS_THROUGH
    f["neighbor_suspicious_share"] = suspicious.groupby(pairs.account).mean()

    f["txn_count"] = by.size()
    f["night_txn_share"] = ev.hour.isin(NIGHT_HOURS).groupby(ev.account).mean()
    round_amount = (ev.amount >= ROUND_FLOOR) & np.isclose(ev.amount % ROUND_STEP, 0)
    f["round_amount_share"] = round_amount.groupby(ev.account).mean()
    below = np.zeros(len(ev), dtype=bool)
    for th in THRESHOLDS:
        below |= (ev.amount >= THRESHOLD_BAND * th).to_numpy() & (ev.amount < th).to_numpy()
    f["below_threshold_share"] = pd.Series(below, index=ev.index).groupby(ev.account).mean()
    f["burstiness"] = _burstiness(ev)
    first_large = inflow[inflow.amount >= LARGE_RECEIPT].groupby("account").t.min()
    created = accounts.set_index("account_id").created_at.reindex(account_ids)
    created_s = _seconds(created).astype(float)
    f["has_large_receipt"] = f.index.isin(first_large.index).astype(float)
    at = first_large.reindex(account_ids).to_numpy(dtype=float)
    at = np.where(np.isnan(at), end_s, at)
    f["account_age_at_first_large_receipt_days"] = np.maximum(0.0, (at - created_s) / 86400)

    devices = outflow[["account", "device_id"]].drop_duplicates()
    f["device_count"] = devices.groupby("account").size()
    sharing = devices.merge(devices, on="device_id", suffixes=("", "_other"))
    sharing = sharing[sharing.account != sharing.account_other]
    f["device_shared_accounts"] = sharing.groupby("account").account_other.nunique()
    state = accounts.set_index("account_id").state
    mismatch = pairs.account.map(state) != pairs.other.map(state)
    f["counterparty_state_mismatch_share"] = mismatch.groupby(pairs.account).mean()

    f = f[FEATURE_COLUMNS].astype(float)
    f["avg_dwell_hours"] = f.avg_dwell_hours.fillna(window_hours)
    return f.fillna(0.0)


def build_feature_table(accounts: pd.DataFrame, txns: pd.DataFrame, splits: pd.Series, cfg: dict) -> pd.DataFrame:
    windows = split_windows(cfg)
    parts = []
    for split in SPLITS:
        ids = splits.index[splits == split]
        if len(ids):
            parts.append(compute_account_features(txns, accounts, ids, *windows[split]).assign(split=split))
    table = pd.concat(parts).sort_index()
    table["label"] = accounts.set_index("account_id").is_fraud.reindex(table.index).astype(int)
    table = table.rename_axis("account_id").reset_index()
    return table[["account_id", *FEATURE_COLUMNS, "label", "split"]]


def account_scenarios(txns: pd.DataFrame) -> pd.Series:
    """account_id -> the scam scenario most of its fraud transactions belong to (analysis only)."""
    fraud = txns[txns.is_fraud == 1]
    long = pd.concat([fraud[["sender_id", "scenario"]].rename(columns={"sender_id": "account_id"}),
                      fraud[["receiver_id", "scenario"]].rename(columns={"receiver_id": "account_id"})])
    counts = long.groupby(["account_id", "scenario"]).size().rename("n").reset_index()
    counts = counts.sort_values(["account_id", "n", "scenario"], ascending=[True, False, True])
    return counts.drop_duplicates("account_id").set_index("account_id").scenario


def _update_checksum(checksums: Path, target: Path) -> None:
    rel = Path("..") / target.relative_to(checksums.parent.parent) if target.is_relative_to(checksums.parent.parent) else target
    line = f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {rel.as_posix()}"
    kept = [l for l in checksums.read_text().splitlines() if not l.endswith(f"  {rel.as_posix()}")] if checksums.exists() else []
    checksums.write_text("\n".join(kept + [line]) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build per-account features")
    parser.add_argument("--data", default="data/generated")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)
    data = Path(args.data)
    cfg = load_config(args.config)
    accounts = read_accounts(data / "accounts.csv")
    txns = read_transactions(data / "transactions.csv")
    splits = pd.read_csv(data / "_meta/account_split.csv").set_index("account_id").split
    table = build_feature_table(accounts, txns, splits, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False, engine="pyarrow")
    if (data / "CHECKSUMS.txt").exists():
        _update_checksum(data / "CHECKSUMS.txt", out)
    print(table.groupby("split").label.agg(accounts="size", positives="sum").to_string())
    print(f"wrote {out} ({len(table):,} accounts x {len(FEATURE_COLUMNS)} features)")


if __name__ == "__main__":
    main()
