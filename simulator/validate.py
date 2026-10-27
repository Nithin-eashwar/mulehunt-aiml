"""Validation report for the generated dataset (the "Data validation" chapter).

    python -m simulator.validate [--data data/generated] [--ibm data/raw/ibm_aml/HI-Small_Patterns.txt]

Compares the generated data with the NPCI/RBI calibration targets in
simulator/config.yaml, counts laundering patterns in the transaction graph,
and checks that every IBM AML pattern type we implement is present (a
structure check only; our pattern mix is set by our own config).
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.config import DEFAULT_PATH, load_config, load_raw_config, unsourced_keys
from simulator.schema import IST_TZ, read_accounts, read_transactions

WINDOW_H = 48
MAX_CYCLE_HOPS = 6
NIGHT_HOURS = [23, 0, 1, 2, 3, 4]
IBM_TO_OURS = {"FAN-OUT": "fan_out", "FAN-IN": "fan_in", "SCATTER-GATHER": "scatter_gather", "CYCLE": "cycle"}
DEFAULT_IBM = Path("data/raw/ibm_aml/HI-Small_Patterns.txt")
NPCI_DAILY = Path("data/raw/npci/upi_daily_2025-08_to_2025-11.csv")
REPORT = Path("reports/a_validation_report.md")
FIGURES = Path("reports/a_figures")
HONEST_FRAMING = ("Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI "
                  "statistics and validated against the IBM AML benchmark.")


# ---------------------------------------------------------------- patterns

def _seconds(ts: pd.Series) -> np.ndarray:
    return ts.dt.tz_convert("UTC").dt.as_unit("s").astype("int64").to_numpy()


def _max_distinct_in_window(keys, others, times, window) -> Counter:
    """For each key, the most distinct `others` seen inside any `window`-second span."""
    order = np.lexsort((times, keys))
    keys, others, times = keys[order], others[order], times[order]
    best, lo, seen = Counter(), 0, Counter()
    for hi in range(len(keys)):
        if hi == 0 or keys[hi] != keys[hi - 1]:
            lo, seen = hi, Counter()
        seen[others[hi]] += 1
        while times[hi] - times[lo] > window:
            seen[others[lo]] -= 1
            if not seen[others[lo]]:
                del seen[others[lo]]
            lo += 1
        best[keys[hi]] = max(best[keys[hi]], len(seen))
    return best


def _chain_starts(senders, receivers, times, window) -> int:
    """Transactions that start a time-respecting path of >= 3 hops (each hop after
    the previous one and within `window` seconds of it)."""
    inf = np.iinfo(np.int64).max
    first_out = [defaultdict(lambda: inf) for _ in range(2)]  # earliest later hop of length >= 1, >= 2
    starts = 0
    for i in np.argsort(-times, kind="stable"):
        u, v, t = senders[i], receivers[i], times[i]
        length = 1
        for k in (1, 0):
            nxt = first_out[k][v]
            if t < nxt <= t + window:
                length = k + 2
                break
        starts += length >= 3
        for k in range(min(length, 2)):
            first_out[k][u] = min(first_out[k][u], t)
    return starts


def _scatter_gather(tx, window) -> int:
    e = tx[["sender_id", "receiver_id", "t"]]
    pairs = e.merge(e, left_on="receiver_id", right_on="sender_id", suffixes=("_1", "_2"))
    pairs = pairs[(pairs.t_2 > pairs.t_1) & (pairs.t_2 <= pairs.t_1 + window)
                  & (pairs.sender_id_1 != pairs.receiver_id_2)]
    middles = pairs.groupby(["sender_id_1", "receiver_id_2"]).receiver_id_1.nunique()
    return int((middles >= 2).sum())


def _cycles(tx, window) -> int:
    out = defaultdict(list)
    for s, r, t in zip(tx.sender_id, tx.receiver_id, tx.t):
        out[s].append((t, r))
    found = set()

    def walk(start, node, t, path):
        for t2, nxt in out[node]:
            if not (t < t2 <= t + window):
                continue
            if nxt == start and len(path) >= 2:
                i = path.index(min(path))
                found.add(tuple(path[i:] + path[:i]))
            elif nxt not in path and len(path) < MAX_CYCLE_HOPS:
                walk(start, nxt, t2, path + [nxt])

    for s, r, t in zip(tx.sender_id, tx.receiver_id, tx.t):
        if s != r:
            walk(s, r, t, [s, r])
    return len(found)


def detect_patterns(txns: pd.DataFrame, window_h: int = WINDOW_H, structural: bool = True) -> dict[str, int]:
    """Count laundering shapes: fan_out / fan_in (>= 3 distinct counterparties within
    the window), chain (transactions starting a >= 3-hop time-respecting path) and,
    when `structural`, scatter_gather (source -> >= 2 middles -> same destination)
    and cycle (time-respecting loops of up to 6 hops)."""
    window = window_h * 3600
    tx = txns.assign(t=_seconds(txns.timestamp))
    if tx.empty:
        empty = {"fan_out": 0, "fan_in": 0, "chain": 0}
        return {**empty, "scatter_gather": 0, "cycle": 0} if structural else empty
    codes, names = pd.factorize(pd.concat([tx.sender_id, tx.receiver_id]))
    s, r = codes[: len(tx)], codes[len(tx):]
    t = tx.t.to_numpy()
    out = {
        "fan_out": sum(v >= 3 for v in _max_distinct_in_window(s, r, t, window).values()),
        "fan_in": sum(v >= 3 for v in _max_distinct_in_window(r, s, t, window).values()),
        "chain": _chain_starts(s, r, t, window),
    }
    if structural:
        out["scatter_gather"] = _scatter_gather(tx, window)
        out["cycle"] = _cycles(tx, window)
    return out


def compare_ibm(patterns: dict, ibm_patterns_path) -> pd.DataFrame:
    """Presence check against IBM AML HI-Small laundering patterns. A mapped type
    PASSes when it appears in our fraud graph; the share columns are reference only."""
    ours = {k: patterns.get(k, 0) for k in [*IBM_TO_OURS.values(), "chain"]}
    our_total = sum(ours.values()) or 1
    path = Path(ibm_patterns_path) if ibm_patterns_path else None
    rows = []
    if path is None or not path.exists():
        for ibm, mine in [*IBM_TO_OURS.items(), ("chain (ours only)", "chain")]:
            rows.append((ibm, mine, np.nan, np.nan, ours[mine], ours[mine] / our_total, "SKIPPED"))
    else:
        found = Counter(m.group(1).strip().upper() for m in
                        re.finditer(r"BEGIN LAUNDERING ATTEMPT - ([^:\n]+)", path.read_text(errors="ignore")))
        ibm_total = sum(found.values()) or 1
        for ibm in sorted(found):
            mine = IBM_TO_OURS.get(ibm, "")
            if mine:
                status, count = ("PASS" if ours[mine] > 0 else "FAIL"), ours[mine]
            else:
                status, count = "not implemented", np.nan
            rows.append((ibm, mine, found[ibm], found[ibm] / ibm_total, count,
                         count / our_total if mine else np.nan, status))
        rows.append(("chain (ours only)", "chain", np.nan, np.nan, ours["chain"], ours["chain"] / our_total,
                     "ours only"))
    return pd.DataFrame(rows, columns=["ibm_pattern", "our_pattern", "ibm_count", "ibm_share",
                                       "our_count", "our_share", "status"])


# ---------------------------------------------------------------- calibration checks

def expected_hour_curve(cfg: dict) -> np.ndarray:
    """Hour-of-day shares the simulator should produce: the configured day-time curve,
    blended with the night-shifted curve for the expected share of transactions that
    night-shift gig workers send (plan rev 4)."""
    w = np.asarray(cfg["hour_of_day"], dtype=float)
    day = w / w.sum()
    hard = cfg.get("hard_cases", {})
    if not hard.get("night_worker_share"):
        return day
    shifted = np.roll(day, hard["night_shift_hours"])
    rates = {t: r.get("p2m_per_day", 0) + r.get("p2p_per_day", 0)
             for t, r in cfg["behaviours"].items() if isinstance(r, dict) and "p2m_per_day" in r}
    mix = cfg["population"]["type_mix"]
    mean_rate = sum(mix[t] * rates[t] for t in rates)
    f = hard["night_worker_share"] * mix["gig_worker"] * rates["gig_worker"] / mean_rate
    return (1 - f) * day + f * shifted


def hour_tvd(observed, expected) -> float:
    p = np.asarray(observed, dtype=float)
    q = np.asarray(expected, dtype=float)
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def _daily(txns):
    day = txns.timestamp.dt.tz_convert(IST_TZ).dt.normalize().dt.tz_localize(None)
    return txns.groupby(day).agg(volume=("amount", "size"), value=("amount", "sum"))


def run_checks(accounts: pd.DataFrame, txns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []

    def add(check, target, generated, passed, source, gap=None):
        if gap is None and isinstance(target, (int, float)) and not isinstance(target, bool):
            gap = generated - target
        rows.append((check, target, generated, gap, passed, source))

    daily = _daily(txns)
    mean_ticket, target = txns.amount.mean(), cfg["ticket_size"]["target_mean_inr"]
    add("mean_ticket_inr", target, mean_ticket, abs(mean_ticket / target - 1) <= 0.15,
        "NPCI daily statistics Aug-Nov 2025 (tolerance ±15%)")
    p2m = (txns.channel == "P2M").mean()
    add("p2m_share", cfg["channel_mix"]["P2M"], p2m, abs(p2m - cfg["channel_mix"]["P2M"]) <= 0.03,
        "secondary (Outlook Money); tolerance ±3 pp")
    small = (txns.amount[txns.channel == "P2M"] <= 500).mean()
    add("p2m_share_le_500", 0.86, small, abs(small - 0.86) <= 0.05, "secondary (Outlook Money); tolerance ±5 pp")
    hours = np.bincount(txns.timestamp.dt.tz_convert(IST_TZ).dt.hour, minlength=24)
    curve = expected_hour_curve(cfg)
    tvd = hour_tvd(hours, curve)
    add("hour_of_day_tvd", 0.0, tvd, tvd <= 0.05,
        "assumed hour curve blended with night workers; total variation distance <= 0.05")
    night, target = hours[NIGHT_HOURS].sum() / hours.sum(), curve[NIGHT_HOURS].sum()
    add("night_share_23_to_05", target, night, abs(night - target) <= 0.02,
        "assumed hour curve blended with night workers; tolerance ±2 pp")

    weekend = daily.volume[daily.index.dayofweek >= 5].mean() / daily.volume.mean()
    add("weekend_volume_factor", 1.0 + cfg["weekend"]["volume_uplift"], weekend, abs(weekend - 1.0) <= 0.05,
        "NPCI daily statistics; tolerance ±0.05")
    sunday = daily.value[daily.index.dayofweek == 6].mean() / daily.value.mean()
    add("sunday_value_factor", cfg["weekend"]["sunday_value_factor"], sunday,
        abs(sunday - cfg["weekend"]["sunday_value_factor"]) <= 0.05, "NPCI daily statistics; tolerance ±0.05")
    start = daily.value[daily.index.day <= 5].mean() / daily.value[daily.index.day > 5].mean()
    target = 1 + cfg["month_start"]["value_uplift_days_1_5"]
    add("month_start_value_factor", target, start, abs(start - target) <= 0.15,
        "NPCI daily statistics; tolerance ±0.15")

    for fest in cfg["festivals"]:
        baseline = daily.volume[pd.Timestamp(fest["baseline_start"]): pd.Timestamp(fest["baseline_end"])].mean()
        uplift = daily.volume[pd.Timestamp(fest["start"]): pd.Timestamp(fest["end"])].mean() / baseline - 1
        add(f"festival_{fest['name']}", fest["observed_uplift"], uplift, abs(uplift - fest["observed_uplift"]) <= 0.03,
            f"NPCI daily statistics vs {fest['baseline_start']}..{fest['baseline_end']}; tolerance ±3 pp")

    add("txn_fraud_share", "< 0.01", txns.is_fraud.mean(), txns.is_fraud.mean() < 0.01, "handoff §4.7", gap=np.nan)
    share = accounts.is_fraud.mean()
    add("account_fraud_share", "0.008 - 0.018", share, 0.008 <= share <= 0.018, "plan D7", gap=np.nan)
    fraud = txns[txns.is_fraud == 1]
    for scenario in cfg["scenarios"]:
        if scenario in ("avg_loss_check_inr",):
            continue
        n = int((fraud.scenario == scenario).sum())
        add(f"fraud_txns_{scenario}", "> 0", n, n > 0, "handoff §4.3", gap=np.nan)
    role = accounts.set_index("account_id").role
    paid = fraud.amount[fraud.sender_id.map(role) == "victim"].sum()
    victims = int((accounts.role == "victim").sum())
    if victims:
        add("avg_victim_loss_inr", cfg["scenarios"]["avg_loss_check_inr"], paid / victims, None,
            "Lok Sabha AU2481 per-incident average; information only")
    return pd.DataFrame(rows, columns=["check", "target", "generated", "gap", "passed", "source"])


# ---------------------------------------------------------------- figures and report

def make_figures(accounts, txns, cfg, out_dir=FIGURES, ibm=None) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {}

    def save(fig, name):
        path = out_dir / f"val_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        figs[name] = path

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.logspace(0, np.log10(txns.amount.max() + 1), 60)
    ax.hist(txns.amount, bins=bins, color="#4C72B0")
    ax.set_xscale("log")
    ax.axvline(cfg["ticket_size"]["target_mean_inr"], color="#C44E52", ls="--", label="NPCI mean ticket ₹1,280")
    ax.axvline(txns.amount.mean(), color="#55A868", ls=":", label=f"generated mean ₹{txns.amount.mean():,.0f}")
    ax.set(xlabel="amount (₹, log scale)", ylabel="transactions", title="Ticket size")
    ax.legend()
    save(fig, "ticket_size")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    gen = txns.channel.value_counts(normalize=True).reindex(["P2P", "P2M"])
    x = np.arange(2)
    ax.bar(x - 0.2, [cfg["channel_mix"]["P2P"], cfg["channel_mix"]["P2M"]], 0.4, label="target", color="#C44E52")
    ax.bar(x + 0.2, gen.to_numpy(), 0.4, label="generated", color="#4C72B0")
    ax.set_xticks(x, ["P2P", "P2M"])
    ax.set(ylabel="share of transactions", title="P2P / P2M mix")
    ax.legend()
    save(fig, "channel_mix")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    hours = np.bincount(txns.timestamp.dt.tz_convert(IST_TZ).dt.hour, minlength=24)
    ax.plot(range(24), expected_hour_curve(cfg), "o-", color="#C44E52", label="expected (config + night workers)")
    ax.plot(range(24), hours / hours.sum(), "s--", color="#4C72B0", label="generated")
    ax.set(xlabel="hour of day (IST)", ylabel="share", title="Hour-of-day pattern")
    ax.legend()
    save(fig, "hour_of_day")

    if NPCI_DAILY.exists():
        npci = pd.read_csv(NPCI_DAILY, parse_dates=["date"]).set_index("date")
        daily = _daily(txns)
        npci = npci.loc[daily.index.min(): daily.index.max()]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(npci.index, npci.volume_mn / npci.volume_mn.mean(), color="#C44E52", label="NPCI (relative)")
        ax.plot(daily.index, daily.volume / daily.volume.mean(), color="#4C72B0", label="generated (relative)")
        ax.set(ylabel="daily volume / mean", title="Daily volume: NPCI vs generated")
        ax.legend()
        save(fig, "daily_volume")

    if ibm is not None and (ibm.status != "SKIPPED").any():
        rows = ibm[ibm.our_pattern.isin(list(IBM_TO_OURS.values()))]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        x = np.arange(len(rows))
        ax.bar(x - 0.2, rows.ibm_count / rows.ibm_count.sum(), 0.4, label="IBM HI-Small (attempts)", color="#C44E52")
        ax.bar(x + 0.2, rows.our_count / rows.our_count.sum(), 0.4, label="ours (shapes found)", color="#4C72B0")
        ax.set_xticks(x, rows.ibm_pattern)
        ax.set(ylabel="share among shared types",
               title="Pattern types present in both (reference only; units differ)")
        ax.legend()
        save(fig, "patterns_vs_ibm")
    return figs


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "PASS" if v else "FAIL"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(path, checks, patterns_fraud, patterns_all, ibm, figures, unsourced, summary=None) -> None:
    failed = checks[checks.passed.eq(False)]
    pattern_rows = pd.DataFrame({"pattern": list(patterns_fraud),
                                 "fraud transactions": list(patterns_fraud.values()),
                                 "all transactions": [patterns_all.get(k, np.nan) for k in patterns_fraud]})
    parts = [
        "# Data validation report", "",
        HONEST_FRAMING, "",
        "Generated by `python -m simulator.validate`. Targets and sources come from `simulator/config.yaml` "
        "and `data/raw/calibration_sources.md`.", "",
    ]
    if summary:
        parts += ["## Dataset", "", "```", summary, "```", ""]
    parts += ["## 1. Calibration checks", "", md_table(checks), ""]
    if len(failed):
        parts += ["**Known gaps (failed checks):** " + ", ".join(f"`{c}`" for c in failed.check)
                  + ". These are reported as they are, not tuned away.", ""]
    parts += [
        "## 2. Laundering patterns in the transaction graph", "",
        f"Shapes counted within a {WINDOW_H} h window. `chain` = transactions that start a time-respecting "
        "path of 3 or more hops. Scatter-gather and cycles are counted on fraud transactions only.", "",
        md_table(pattern_rows), "",
        "## 3. IBM AML benchmark (structure check)", "",
        "PASS means the IBM pattern type appears in our fraud graph. The share columns are for reference only: "
        "our mix is set by our own config, and the units differ (IBM counts laundering attempts; we count "
        "shapes found in the graph: accounts for fan-out/fan-in, account pairs for scatter-gather, distinct "
        "loops for cycles and starting transactions for chains).", "",
        md_table(ibm), "",
    ]
    if figures:
        parts += ["## 4. Figures", ""]
        rel = Path(path).parent
        parts += [f"![{name}]({Path(p).relative_to(rel).as_posix() if Path(p).is_relative_to(rel) else Path(p).as_posix()})"
                  for name, p in figures.items()] + [""]
    parts += ["## 5. Values that are still assumptions", "",
              "These config values have no public source and are marked `assumed` in `simulator/config.yaml`:", "",
              ", ".join(f"`{k}`" for k in unsourced), ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated MuleHunt dataset")
    parser.add_argument("--data", default="data/generated")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--ibm", default=str(DEFAULT_IBM))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    cfg = load_config(args.config)
    accounts = read_accounts(Path(args.data) / "accounts.csv")
    txns = read_transactions(Path(args.data) / "transactions.csv")
    checks = run_checks(accounts, txns, cfg)
    fraud_patterns = detect_patterns(txns[txns.is_fraud == 1])
    all_patterns = detect_patterns(txns, structural=False)
    ibm = compare_ibm(fraud_patterns, args.ibm)
    figures = make_figures(accounts, txns, cfg, Path(args.report).parent / "a_figures", ibm)
    summary = (f"{len(accounts):,} accounts ({accounts.is_fraud.mean():.2%} fraud labels), "
               f"{len(txns):,} transactions ({txns.is_fraud.mean():.3%} fraud)")
    write_report(args.report, checks, fraud_patterns, all_patterns, ibm, figures,
                 unsourced_keys(load_raw_config(args.config)), summary)
    print(md_table(checks))
    print("\nfraud-graph patterns:", fraud_patterns)
    print("IBM check:", ", ".join(f"{r.ibm_pattern}={r.status}" for r in ibm.itertuples()))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
