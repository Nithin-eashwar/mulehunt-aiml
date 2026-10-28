"""Markdown results reports, written from the terminal instead of notebooks (plan D21).

    reports/a_01_simulator.md     written by `python -m simulator.validate`
    reports/a_05_classifiers.md   written by `python -m mulehunt.models --all`
    reports/a_06_clustering.md    written by `python -m mulehunt.clustering`

`python -m mulehunt.pipeline` runs all of them. Every number in a report is read
from the stage's own outputs (tables in reports/a_tables/, figures in
reports/a_figures/, models/), so a report never disagrees with the files beside it.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.validate import md_table

REPORTS = Path("reports")
FIGURES = REPORTS / "a_figures"
TABLES = REPORTS / "a_tables"
SIMULATOR_REPORT = REPORTS / "a_01_simulator.md"
CLASSIFIER_REPORT = REPORTS / "a_05_classifiers.md"
CLUSTERING_REPORT = REPORTS / "a_06_clustering.md"
TIERS = ("victim", "fraudster", "mule_t1", "mule_t2", "mule_t3", "cashout")
TIER_COLOURS = {"victim": "#8C8C8C", "fraudster": "#C44E52", "mule_t1": "#DD8452", "mule_t2": "#CCB974",
                "mule_t3": "#55A868", "cashout": "#4C72B0"}
SLEEPER_HISTORY_DAYS = 21
RULE_LEVELS = 3


# ---------------------------------------------------------------- generic writer

def _cell_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: str(v).replace("|", "\\|").replace("\n", " ") if isinstance(v, str) else v)
    return out


def _body(item, report_dir: Path) -> str:
    if isinstance(item, pd.DataFrame):
        return md_table(_cell_safe(item))
    if isinstance(item, Path):
        if not item.exists():
            raise FileNotFoundError(item)
        link = Path(os.path.relpath(item.resolve(), report_dir.resolve())).as_posix()
        return f"![{item.stem}]({link})"
    if isinstance(item, (list, tuple)):
        return "\n\n".join(_body(part, report_dir) for part in item)
    return str(item).strip()


def write_markdown_report(path, title: str, sections: list[tuple[str, object]]) -> None:
    """Each section body is prose, a DataFrame (pipe table), a figure Path (embedded
    with a link relative to the report) or a list of those."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}"]
    parts += [f"## {heading}\n\n{_body(body, path.parent)}" for heading, body in sections]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _inr(x: float) -> str:
    return f"Rs {x:,.0f}"


def _savefig(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------- a_01 simulator

def load_rings(data_dir) -> list[dict]:
    return json.loads((Path(data_dir) / "_meta" / "rings.json").read_text(encoding="utf-8"))


def ring_members(ring: dict) -> dict[str, str]:
    """account_id -> tier for one ring."""
    members = {v: "victim" for v in ring["victims"]}
    members[ring["fraudster"]] = "fraudster"
    for tier, ids in ring["mules"].items():
        members.update({a: tier for a in ids})
    members[ring["cashout"]] = "cashout"
    return members


def ring_figure(ring: dict, txns: pd.DataFrame, path: Path) -> Path:
    """One ring as a left-to-right money-flow graph: victims, fraudster, mule tiers, cash-out."""
    import networkx as nx
    plt = _plt()

    members = ring_members(ring)
    tx = txns[(txns.is_fraud == 1) & txns.sender_id.isin(members) & txns.receiver_id.isin(members)]
    edges = tx.groupby(["sender_id", "receiver_id"]).amount.agg(["sum", "size"]).reset_index()
    g = nx.DiGraph()
    g.add_nodes_from(members)
    for e in edges.itertuples():
        g.add_edge(e.sender_id, e.receiver_id, amount=e[3], n=e[4])
    pos = {}
    for x, tier in enumerate(TIERS):
        ids = sorted(a for a, t in members.items() if t == tier)
        for i, a in enumerate(ids):
            pos[a] = (x, (len(ids) - 1) / 2 - i)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    widths = [0.6 + 1.2 * np.log10(max(g.edges[e]["amount"], 10)) / 5 for e in g.edges]
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, alpha=0.55, arrowsize=12, edge_color="#555555",
                           connectionstyle="arc3,rad=0.08", node_size=420)
    for tier in TIERS:
        ids = [a for a, t in members.items() if t == tier]
        if ids:
            nx.draw_networkx_nodes(g, pos, nodelist=ids, ax=ax, node_color=TIER_COLOURS[tier], node_size=420,
                                   label=tier.replace("_", " "))
    nx.draw_networkx_labels(g, pos, {a: a[-4:] for a in members}, ax=ax, font_size=7)
    victim_loss = tx[tx.sender_id.isin(ring["victims"])].amount.sum()
    flags = [f for f, on in (("evasive", ring["evasive"]), ("cycle", ring["cycle"])) if on]
    ax.set_title(f"{ring['scenario']} ring {ring['ring_id']} ({ring['split']}): {len(ring['victims'])} victims, "
                 f"{_inr(victim_loss)} lost, {len(tx)} fraud transactions" + (f" [{', '.join(flags)}]" if flags else ""),
                 fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=8, frameon=False)
    ax.axis("off")
    return _savefig(fig, path)


def _timeline(ax, txns: pd.DataFrame, account: str, title: str, marker_at=None):
    tx = txns[(txns.sender_id == account) | (txns.receiver_id == account)]
    for fraud, colour, label in ((0, "#4C72B0", "normal"), (1, "#C44E52", "fraud")):
        for out, m in ((True, "^"), (False, "v")):
            part = tx[(tx.is_fraud == fraud) & ((tx.sender_id == account) == out)]
            if len(part):
                ax.scatter(part.timestamp, part.amount, s=18, marker=m, color=colour, alpha=0.8,
                           label=f"{label} {'sent' if out else 'received'}")
    if marker_at is not None:
        ax.axvline(marker_at, color="grey", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("amount (Rs, log)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)


def find_sleeper(accounts: pd.DataFrame, rings: list[dict]) -> tuple[str, dict] | None:
    """First mule (by ring id) whose account existed >= 21 days before its ring started."""
    created = accounts.set_index("account_id").created_at
    for ring in sorted(rings, key=lambda r: r["ring_id"]):
        start = pd.Timestamp(ring["start"])
        for tier in ("mule_t1", "mule_t2", "mule_t3"):
            for a in ring["mules"][tier]:
                if created[a] <= start - pd.Timedelta(days=SLEEPER_HISTORY_DAYS):
                    return a, ring
    return None


def find_outlier(accounts: pd.DataFrame, txns: pd.DataFrame, min_ratio: float = 10.0,
                 min_sent: int = 15) -> tuple[str, float, pd.Timestamp, float] | None:
    """An ordinary user with one isolated spike: a normal payment >= min_ratio x their median,
    chosen as the account whose largest payment stands furthest above its second largest.
    Returns (account, largest / median, when, amount)."""
    plain = accounts.account_id[(accounts.role == "normal") & (accounts.is_fraud == 0)]
    sent = txns[(txns.is_fraud == 0) & txns.sender_id.isin(plain)]
    ranked = sent.sort_values(["sender_id", "amount"], ascending=[True, False])
    rank = ranked.groupby("sender_id").cumcount()
    stats = sent.groupby("sender_id").amount.agg(["median", "size"])
    stats["top"] = ranked[rank == 0].set_index("sender_id").amount
    stats["second"] = ranked[rank == 1].set_index("sender_id").amount
    spikes = stats[(stats.top >= min_ratio * stats["median"]) & (stats["size"] >= min_sent)]
    if spikes.empty:
        return None
    acc = (spikes.top / spikes.second).idxmax()
    row = ranked[(ranked.sender_id == acc)].iloc[0]
    return acc, float(spikes.top[acc] / spikes["median"][acc]), row.timestamp, float(row.amount)


def sleeper_share(accounts: pd.DataFrame, rings: list[dict]) -> float:
    created = accounts.set_index("account_id").created_at
    flags = [created[a] <= pd.Timestamp(r["start"]) - pd.Timedelta(days=SLEEPER_HISTORY_DAYS)
             for r in rings for tier in ("mule_t1", "mule_t2", "mule_t3") for a in r["mules"][tier]]
    return float(np.mean(flags)) if flags else 0.0


def noise_figure(accounts, txns, rings, path: Path) -> tuple[Path, dict]:
    plt = _plt()
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.5))
    info = {}
    sleeper = find_sleeper(accounts, rings)
    if sleeper:
        acc, ring = sleeper
        start = pd.Timestamp(ring["start"])
        before = txns[((txns.sender_id == acc) | (txns.receiver_id == acc)) & (txns.timestamp < start)]
        info["sleeper"] = (acc, ring, len(before))
        _timeline(a, txns, acc, f"Sleeper mule {acc}: {len(before)} normal transactions, then activated by "
                                f"{ring['scenario']} ring {ring['ring_id']} (dashed line)", start)
    outlier = find_outlier(accounts, txns)
    if outlier:
        acc, ratio, when, amount = outlier
        info["outlier"] = (acc, ratio, amount)
        _timeline(b, txns, acc, f"Genuine outlier {acc}: one payment of {_inr(amount)}, {ratio:.0f}x their median "
                                "(circled; not fraud)")
        b.scatter([when], [amount], s=160, facecolors="none", edgecolors="#C44E52", linewidths=1.5)
    return _savefig(fig, path), info


def config_table(raw: dict) -> pd.DataFrame:
    from simulator.config import _leaves

    rows = []
    for key, leaf in _leaves(raw):
        value = leaf["value"]
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        rows.append({"key": key, "value": text, "status": leaf["status"], "source": str(leaf["source"])})
    return pd.DataFrame(rows)


def dataset_tables(accounts, txns, rings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from simulator.schema import FRAUD_ROLES

    fraud_role = accounts.role.isin(FRAUD_ROLES)
    days = (txns.timestamp.max().normalize() - txns.timestamp.min().normalize()).days + 1
    summary = pd.DataFrame([
        ("Window", f"{txns.timestamp.min():%Y-%m-%d} to {txns.timestamp.max():%Y-%m-%d} ({days} days)"),
        ("Accounts", f"{len(accounts):,}"),
        ("Transactions", f"{len(txns):,}"),
        ("Fraud rings", f"{len(rings)}"),
        ("Fraud accounts (true role)", f"{fraud_role.sum():,} ({_pct(fraud_role.mean(), 2)})"),
        ("Fraud labels in accounts.csv (with noise)", f"{accounts.is_fraud.sum():,} ({_pct(accounts.is_fraud.mean(), 2)})"),
        ("Fraud transactions", f"{txns.is_fraud.sum():,} ({_pct(txns.is_fraud.mean(), 3)})"),
        ("Victims", f"{(accounts.role == 'victim').sum():,}"),
    ], columns=["item", "value"])
    roles = accounts.role.value_counts().rename_axis("role").reset_index(name="accounts")
    rows = []
    for scenario in sorted({r["scenario"] for r in rings}):
        rs = [r for r in rings if r["scenario"] == scenario]
        members = set().union(*(ring_members(r) for r in rs))
        victims = [v for r in rs for v in r["victims"]]
        fraud_ids = members - set(victims)
        loss = txns[(txns.is_fraud == 1) & (txns.scenario == scenario) & txns.sender_id.isin(victims)].amount
        rows.append({"scenario": scenario, "rings": len(rs), "evasive_rings": sum(r["evasive"] for r in rs),
                     "fraud_accounts": len(fraud_ids), "victims": len(victims),
                     "fraud_txns": int(((txns.is_fraud == 1) & (txns.scenario == scenario)).sum()),
                     "mean_loss_per_victim_rs": loss.sum() / max(len(set(victims)), 1)})
    return summary, roles, pd.DataFrame(rows)


def write_simulator_report(path=SIMULATOR_REPORT, *, raw_cfg: dict, accounts, txns, data_dir, checks,
                           fraud_patterns: dict, all_patterns: dict, ibm: pd.DataFrame, figures: dict,
                           fig_dir=FIGURES) -> Path:
    from simulator.schema import FRAUD_ROLES
    from simulator.validate import HONEST_FRAMING

    fig_dir = Path(fig_dir)
    rings = load_rings(data_dir)
    summary, roles, scenarios = dataset_tables(accounts, txns, rings)
    cfg_table = config_table(raw_cfg)
    assumed = cfg_table[cfg_table.status == "assumed"]

    ring_figs = []
    for scenario in scenarios.scenario:
        ring = min((r for r in rings if r["scenario"] == scenario), key=lambda r: r["ring_id"])
        ring_figs.append(f"**{scenario}**")
        ring_figs.append(ring_figure(ring, txns, fig_dir / f"sim_ring_{scenario}.png"))
    noise_path, noise = noise_figure(accounts, txns, rings, fig_dir / "sim_noise_examples.png")

    fraud_role = accounts.role.isin(FRAUD_ROLES)
    noise_table = pd.crosstab(np.where(fraud_role, "fraud role", "not fraud role"),
                              accounts.is_fraud.map({0: "label 0", 1: "label 1"})).reset_index()
    noise_table.columns = ["true role", *noise_table.columns[1:]]
    flipped_pos = int((fraud_role & (accounts.is_fraud == 0)).sum())
    flipped_neg = int((~fraud_role & (accounts.is_fraud == 1)).sum())
    noise_lines = []
    if "sleeper" in noise:
        acc, ring, n_before = noise["sleeper"]
        noise_lines.append(f"- **Sleeper mule** (top panel): `{acc}` made {n_before} ordinary transactions before "
                           f"{ring['scenario']} ring {ring['ring_id']} used it. {_pct(sleeper_share(accounts, rings), 0)} "
                           "of mules are sleepers like this (accounts at least 21 days old when their ring starts), "
                           "and they send from their own phones, so age and device features alone cannot find them.")
    if "outlier" in noise:
        acc, ratio, amount = noise["outlier"]
        noise_lines.append(f"- **Genuine outlier** (bottom panel): `{acc}` is an ordinary user who once paid "
                           f"{_inr(amount)}, {ratio:.0f}x their median payment (a wedding, a hospital bill, a phone). "
                           "It is labelled 0, so a model must not flag every spike.")
    noise_lines.append(f"- **Label noise:** in `accounts.csv`, {flipped_pos} fraud accounts are labelled 0 and "
                       f"{flipped_neg} non-fraud accounts are labelled 1 (D9). `role` and `transactions.csv` stay "
                       "clean, so validity checks use the true role.")
    noise_lines.append(f"- **Threshold splitting:** {int(scenarios.evasive_rings.sum())} of {len(rings)} rings split "
                       "victim payments to 90-99.9% of Rs 10k / 25k / 50k.")

    victims = {v for r in rings for v in r["victims"]}
    overall_loss = txns[(txns.is_fraud == 1) & txns.sender_id.isin(victims)].amount.sum() / max(len(victims), 1)

    toy = Path(data_dir) / "toy"
    toy_text = ("`python -m simulator.make_toy_data --out data/generated/toy` writes a tiny dataset with the same "
                "columns, for Person B to build against without the real data. When `data/generated/accounts.csv` "
                "is missing, code falls back to it (CONTRACTS.md).")
    if (toy / "accounts.csv").exists():
        n_acc = sum(1 for _ in open(toy / "accounts.csv", encoding="utf-8")) - 1
        n_tx = sum(1 for _ in open(toy / "transactions.csv", encoding="utf-8")) - 1
        toy_text += f" It holds {n_acc:,} accounts and {n_tx:,} transactions and is committed to git."

    pattern_table = pd.DataFrame([{"pattern": p, "fraud graph": n, "whole graph (fan/chain only)": all_patterns.get(p)}
                                  for p, n in fraud_patterns.items()])
    val_figs = [p for p in figures.values() if Path(p).exists()]
    check_6 = ("Check 6 (models must not score near 100%) needs the trained models, so it is reported in "
               "`a_05_classifiers.md` under \"Shortcut check\".")
    sections = [
        ("Why simulated data", HONEST_FRAMING),
        ("Regenerating", "```bash\npython -m simulator.generate --config simulator/config.yaml --out data/generated\n"
                         "python -m simulator.validate\n```\n\nThe output is identical for the same seed and the "
                         "pinned `requirements.txt`; compare with `data/generated/CHECKSUMS.txt`."),
        ("Dataset summary", [summary, "Accounts by true role:", roles]),
        ("The five scam scenarios", [
            f"One row per scenario. Across all scenarios a victim loses {_inr(overall_loss)} on average, against the "
            "Lok Sabha FY2024-25 average of Rs 7,761 per reported incident. The scenarios differ on purpose: a "
            "digital-arrest victim pays several large instalments, a job-task victim a few small fees.", scenarios]),
        ("Config and calibration sources",
         [f"Every number lives in `simulator/config.yaml` with its source beside it. {len(cfg_table)} values; "
          f"{len(assumed)} are marked `assumed`: design choices from the plan, or values with no public source. "
          "They are listed so the report can say so honestly:",
          ", ".join(f"`{k}`" for k in assumed.key) or "none", cfg_table]),
        ("Toy data", toy_text),
        ("Example rings and mule patterns",
         ["One ring per scenario, money flowing left to right. Node labels are the last four digits of the account "
          "id; edge width grows with the amount.", *ring_figs]),
        ("Noise", ["\n".join(noise_lines), noise_table, noise_path]),
        ("Validation (handoff §5.3)", [checks, *val_figs, check_6]),
        ("Laundering patterns and the IBM AML check",
         ["Pattern counts in our fraud transactions (48 h windows):", pattern_table,
          "Structure check against IBM AML HI-Small (D20): PASS means every IBM pattern type we implement appears "
          "in our graph. The mix is shown but not judged, because our own config sets it.", ibm]),
    ]
    write_markdown_report(path, "Simulator and data validation (Person A)", sections)
    return Path(path)


# ---------------------------------------------------------------- a_05 classifiers

def split_figure(cfg: dict, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 2.6))
    lookback = cfg["splits"]["lookback_days"]
    colours = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    for i, split in enumerate(("train", "val", "test")):
        first, last = cfg["splits"][split]
        ax.barh(i, last - first + 1, left=first - 1, color=colours[split], alpha=0.35, label=f"{split} days")
        ax.barh(i, lookback, left=last - lookback, height=0.3, color=colours[split],
                label=f"{split} feature window")
        ax.text(last + 0.5, i, f"{split}: days {first}-{last}, features days {last - lookback + 1}-{last}",
                va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, cfg["window"]["days"] + 42)
    ax.set_xlabel("simulation day")
    ax.set_title("Leakage-safe split: each account in exactly one split, features from that split's window only",
                 fontsize=10)
    return _savefig(fig, path)


def write_classifier_report(path=CLASSIFIER_REPORT, *, tab_dir=TABLES, fig_dir=FIGURES, features_path=None,
                            models_dir=None, meta_path=None, config_path=None) -> Path:
    from sklearn.metrics import accuracy_score, average_precision_score

    from mulehunt import models as m
    from mulehunt.features import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, FEATURE_GROUPS
    from simulator.config import DEFAULT_PATH, load_config

    tab_dir, fig_dir = Path(tab_dir), Path(fig_dir)
    models_dir = Path(models_dir or m.MODELS_DIR)
    cfg = load_config(config_path or DEFAULT_PATH)
    table = pd.read_parquet(features_path or m.FEATURES)
    splits = m.load_splits(features_path or m.FEATURES)
    t = {name: pd.read_csv(tab_dir / f"{name}.csv") for name in
         ("leaderboard", "tree_depth_curve", "tree_pruning_path", "imbalance_comparison", "single_feature_pr_auc",
          "lr_coefficients", "ablation", "per_scenario")}
    board = t["leaderboard"]

    best_name = (models_dir / "best_model.txt").read_text(encoding="utf-8").strip()
    bundle = m.load_model(best_name, models_dir)
    X_te, y_te, _ = splits["test"]
    p = bundle["pipeline"].predict_proba(X_te)[:, 1]
    pred = p >= bundle["threshold"]
    best_row = board.set_index("model").loc[best_name]
    accuracy = pd.DataFrame([
        {"model": "always 'not fraud'", "accuracy": accuracy_score(y_te, np.zeros_like(y_te)), "precision": 0.0,
         "recall": 0.0, "f1": 0.0, "pr_auc": float(y_te.mean())},
        {"model": best_name, "accuracy": accuracy_score(y_te, pred), "precision": best_row.precision,
         "recall": best_row.recall, "f1": best_row.f1, "pr_auc": average_precision_score(y_te, p)},
    ])

    family = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    features = pd.DataFrame([{"feature": f"`{c}`", "family": family[c], "definition": d}
                             for c, d in FEATURE_DESCRIPTIONS.items()])

    meta = pd.read_csv(meta_path or m.META, dtype={"ring_id": "Int64"})
    lookback = cfg["splits"]["lookback_days"]
    split_rows = []
    for s in m.SPLITS:
        first, last = cfg["splits"][s]
        part = table[table.split == s]
        split_rows.append({"split": s, "split days": f"{first}-{last}",
                           "feature window days": f"{last - lookback + 1}-{last}", "accounts": len(part),
                           "positives": int(part.label.sum()),
                           "rings": int(meta[meta.split == s].ring_id.dropna().nunique())})

    depth = t["tree_depth_curve"]
    best_gini = depth.loc[depth.val_pr_auc.idxmax()]
    best_entropy = depth.loc[depth.val_pr_auc_entropy.idxmax()]
    prune = t["tree_pruning_path"]
    best_prune = prune.loc[prune.val_pr_auc.idxmax()]
    rules = (tab_dir / "tree_rules.txt").read_text(encoding="utf-8").splitlines()
    shown = [r for r in rules if r.count("|") <= RULE_LEVELS]   # "|   " per level, then "|---"
    rules_text = "\n".join(shown)
    winner = "entropy" if best_entropy.val_pr_auc_entropy > best_gini.val_pr_auc else "Gini"

    lb = board.assign(pr_auc_95ci=[f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(board.pr_auc_ci_low, board.pr_auc_ci_high)])
    lb = lb[["model", "pr_auc", "pr_auc_95ci", "precision", "recall", "f1", "recall_at_1pct_fpr", "threshold",
             "params"]].sort_values("pr_auc", ascending=False)
    single = t["single_feature_pr_auc"]
    abl = t["ablation"].set_index("features")
    gain = abl.loc["with_network", "test_pr_auc"] - abl.loc["transaction_only", "test_pr_auc"]
    coefs = t["lr_coefficients"].reindex(t["lr_coefficients"].coefficient.abs().sort_values(ascending=False).index)
    corr = table[FEATURE_COLUMNS].corr().abs().where(lambda c: c < 1)
    pair = corr.stack().idxmax()
    collinear = (f"`{pair[0]}` and `{pair[1]}` are {corr.loc[pair]:.2f} correlated, so the model gives them large "
                 "weights of opposite sign that mostly cancel: read them together, not as two separate risk factors. "
                 "That is collinearity, and it is why the tree models are used for explanations.")
    imb = t["imbalance_comparison"]
    imb_best = imb.loc[imb.val_pr_auc.idxmax(), "method"]
    imb_text = (f"On validation PR-AUC, `{imb_best}` is best for logistic regression. PR-AUC only measures ranking, "
                "and re-weighting mostly shifts the scores; the decision threshold is tuned on validation anyway, "
                "so imbalance handling matters less here than its reputation suggests.")
    scen = t["per_scenario"].sort_values("recall")
    lb_rf = board.set_index("model").pr_auc.get("random_forest", np.nan)
    scen_note = (f"Weakest: `{scen.scenario.iloc[0]}` (recall {scen.recall.iloc[0]:.2f}); strongest: "
                 f"`{scen.scenario.iloc[-1]}` ({scen.recall.iloc[-1]:.2f}). Each row rests on only "
                 f"{int(scen.n_fraud.min())}-{int(scen.n_fraud.max())} fraud accounts, so gaps of a few points are noise.")

    risk = pd.read_csv(models_dir / "risk_scores.csv").merge(table[["account_id", "label", "split"]], on="account_id")
    risk_rows = pd.DataFrame([{"split": s, "accounts": int((risk.split == s).sum()),
                               "pr_auc_of_risk_prob": average_precision_score(risk.label[risk.split == s],
                                                                              risk.risk_prob[risk.split == s])}
                              for s in m.SPLITS])
    files = pd.DataFrame([{"file": f"`models/{f.name}`", "size_kb": round(f.stat().st_size / 1024)}
                          for f in sorted(models_dir.glob("*.joblib"))])
    gate = "PASS" if single.val_pr_auc.max() < 0.9 and board.pr_auc.max() < 0.98 else "FAIL"

    sections = [
        ("Why not accuracy", [
            f"Only {_pct(y_te.mean(), 2)} of test accounts are fraud. A model that always says \"not fraud\" is "
            f"{_pct(accuracy.accuracy[0], 1)} accurate and catches nobody, so accuracy is never a headline here. "
            "We report precision, recall, F1, PR-AUC and recall at a 1% false-alarm rate. Precision matters because "
            "wrongly freezing a genuine user's account causes real harm.", accuracy]),
        ("Features", ["One row per account, computed only from that account's transactions inside its split's "
                      "30-day window. No feature reads `role`, `is_fraud` or `scenario`.", features]),
        ("Leakage-safe split", [
            split_figure(cfg, fig_dir / "split_timeline.png"), pd.DataFrame(split_rows),
            "- Each fraud ring, with all its members, is in exactly one split; so is each normal account.\n"
            "- SMOTE runs inside the training pipeline only, never on validation or test rows.\n"
            "- `neighbor_suspicious_share` uses counterparties' pass-through, not their labels.\n"
            "- Thresholds and hyperparameters are chosen on validation; test is used once, for the leaderboard.\n"
            "- Train risk scores are out-of-fold, with each ring kept inside one fold."]),
        ("Decision tree: overfitting curve, Gini vs entropy", [
            fig_dir / "tree_depth_curve.png",
            f"Training PR-AUC keeps rising with depth while validation peaks: best validation PR-AUC "
            f"{best_gini.val_pr_auc:.3f} at depth {int(best_gini.depth)} with Gini, {best_entropy.val_pr_auc_entropy:.3f} "
            f"at depth {int(best_entropy.depth)} with entropy, so {winner} splits better here. Deeper trees only "
            "memorise the training accounts.",
            depth[["depth", "train_pr_auc", "val_pr_auc", "train_pr_auc_entropy", "val_pr_auc_entropy"]]]),
        ("Decision tree: cost-complexity pruning and the mule rulebook", [
            fig_dir / "tree_pruning_path.png",
            f"Best validation PR-AUC {best_prune.val_pr_auc:.3f} with {int(best_prune.n_leaves)} leaves "
            f"(alpha = {best_prune.alpha:.2e}).", fig_dir / "tree.png",
            f"The pruned tree as rules, first {RULE_LEVELS} levels (the full rulebook is "
            "`reports/a_tables/tree_rules.txt`). Each line is an attribute test condition:",
            f"```text\n{rules_text}\n```"]),
        ("Class imbalance", ["Logistic regression with no handling, balanced class weights and SMOTE (training rows "
                             "only). The other models use balanced class weights, except KNN (SMOTE) and Naive "
                             "Bayes (priors from the data) (D15).", t["imbalance_comparison"], imb_text]),
        ("Leaderboard (test split)", [
            "All eight models, same features, same splits. Hyperparameters and the decision threshold (best F1) are "
            "chosen on validation. The 95% interval resamples whole rings, because ring-mates are correlated.",
            lb, fig_dir / "leaderboard.png", fig_dir / "pr_curves.png",
            "Naive Bayes assumes features are independent given the class. Flow features are strongly correlated "
            "(inflow, outflow, pass-through, dwell), so it counts the same evidence several times and ranks poorly."]),
        ("Shortcut check (handoff §5.3 check 6)", [
            f"**{gate}.** The best single feature reaches validation PR-AUC {single.val_pr_auc.max():.3f} on its own "
            f"(gate 0.9) and the best model {board.pr_auc.max():.3f} on test (gate 0.98).", single.head(10)]),
        ("Logistic regression coefficients as risk factors", [
            "Standardised features, so coefficients are comparable. An odds ratio above 1 raises the fraud risk.",
            coefs, collinear]),
        ("Ablation: what network features add", [
            f"Adding the six network features changes test PR-AUC by {gain:+.3f} (random forest).",
            t["ablation"],
            f"Both rows use the random forest's default settings, without the validation grid, so the with-network "
            f"row differs slightly from the leaderboard ({lb_rf:.3f}); the comparison between the rows is fair."]),
        ("Per-scenario results", ["Recall of the best model on each scam's fraud accounts in test.", scen,
                                  scen_note]),
        ("Handoff files", [
            f"`models/risk_scores.csv`: {len(risk):,} accounts from `{best_name}`, clipped to [1e-6, 1 - 1e-6] for "
            "Person B's A* cost. Train rows are out-of-fold, so their PR-AUC is not better than test:",
            risk_rows, "Saved models (load with `mulehunt.models.load_model`, see CONTRACTS.md §5):", files]),
    ]
    write_markdown_report(path, "Supervised fraud models (Person A)", sections)
    return Path(path)


# ---------------------------------------------------------------- a_06 clustering

def write_clustering_report(path=CLUSTERING_REPORT, *, tab_dir=TABLES, fig_dir=FIGURES) -> Path:
    from mulehunt.clustering import CLUSTER_FEATURES, LOG_FEATURES, MULE_ROLES

    tab_dir, fig_dir = Path(tab_dir), Path(fig_dir)
    sweep = pd.read_csv(tab_dir / "cluster_sweep.csv")
    validity = pd.read_csv(tab_dir / "cluster_validity.csv")
    shuffle = pd.read_csv(tab_dir / "cluster_shuffle.csv")
    ks = sorted(validity.k.unique())
    k_best, k_fine = ks[0], ks[-1]
    best_sil_k = int(sweep.loc[sweep.silhouette.idxmax(), "k"])
    lift = {(meth, k): pd.read_csv(tab_dir / f"cluster_lift_{meth}_k{k}.csv")
            for meth in ("kmeans", "bisecting_kmeans") for k in ks}
    fine = lift[("kmeans", k_fine)]
    n_active, n_mules = int(fine["size"].sum()), int(fine.mules.sum())
    base = n_mules / n_active
    top = fine.iloc[0]
    capture = validity.set_index(["method", "k"]).mule_capture_lift5
    mostly = fine[fine.mule_share > 0.5]
    verdict = (f"{len(mostly)} cluster(s) are mostly mules." if len(mostly) else
               f"**No cluster is mostly mules**: the best holds {_pct(top.mule_share)} mules, so it is a strong "
               "pointer, not a mule list. The handoff called this a hope, not a guarantee; this is what the data shows.")

    profiles = pd.read_csv(tab_dir / f"cluster_profiles_k{k_fine}.csv")
    role_cols = [f"role_{r}" for r in MULE_ROLES if f"role_{r}" in profiles]
    profiles["mule_share"] = profiles[role_cols].sum(axis=1)
    keep = ["cluster", "size", "mule_share", "pass_through_ratio", "avg_dwell_hours", "fan_in_max_24h",
            "fan_out_max_24h", "in_degree", "night_txn_share"] + [c for c in ("role_merchant", "role_normal")
                                                                    if c in profiles]
    prof = profiles.sort_values("mule_share", ascending=False)[keep]
    overall = profiles[keep[3:]].mul(profiles["size"], axis=0).sum() / profiles["size"].sum()
    head = prof.iloc[0]
    others = prof.iloc[1:][prof.iloc[1:].mule_share > 5 * base]
    profile_text = (
        f"Mean feature values and role mix per cluster. The top cluster ({int(head.cluster)}) forwards most of what it "
        f"receives (pass-through {head.pass_through_ratio:.2f} vs {overall.pass_through_ratio:.2f} overall), quickly "
        f"(dwell {head.avg_dwell_hours:.0f} h vs {overall.avg_dwell_hours:.0f} h) and from many senders (24 h fan-in "
        f"{head.fan_in_max_24h:.1f} vs {overall.fan_in_max_24h:.1f}). {_pct(1 - head.mule_share, 0)} of its members are "
        "not mules: the money collectors, shop phones and new accounts with large first receipts were added as hard "
        "cases (plan rev 4) precisely so that this shape is not unique to mules.")
    if len(others):
        profile_text += (" The other mule-rich clusters (" + ", ".join(str(int(c)) for c in others.cluster) +
                         ") look different, e.g. pass-through " +
                         ", ".join(f"{x:.2f}" for x in others.pass_through_ratio) +
                         ": mules are not one shape, which is why one cluster cannot hold them all.")

    v = validity.copy()
    shuf = shuffle.groupby("k").silhouette.agg(["mean", "std"]).reset_index()
    sig = v[v.method == "kmeans"][["k", "silhouette", "sse"]].merge(shuf, on="k")
    sig = sig.rename(columns={"silhouette": "real_silhouette", "mean": "shuffled_silhouette_mean",
                              "std": "shuffled_silhouette_sd", "sse": "real_sse"})
    sig["shuffled_sse_mean"] = sig.k.map(v.groupby("k").shuffled_sse_mean.first())
    sig["silhouette_gap_in_sd"] = (sig.real_silhouette - sig.shuffled_silhouette_mean) / sig.shuffled_silhouette_sd
    sig_text = ("Each feature column is permuted independently (same distributions, no joint structure) and clustered "
                f"again, {int(shuffle.groupby('k').size().min())} times per K. ")
    if (sig.silhouette_gap_in_sd > 3).all() and (sig.real_sse < sig.shuffled_sse_mean).all():
        sig_text += (f"Real silhouette is at least {sig.silhouette_gap_in_sd.min():.1f} standard deviations above the "
                     "shuffled mean at every K reported, and real SSE is lower, so the structure is not chance.")
    else:
        sig_text += "At some K the real silhouette is within 3 standard deviations of shuffled data; see the table."
    fine_v = v[v.k == k_fine].set_index("method")
    km_wins = (fine_v.loc["kmeans", "sse"] < fine_v.loc["bisecting_kmeans", "sse"],
               fine_v.loc["kmeans", "mule_capture_lift5"] > fine_v.loc["bisecting_kmeans", "mule_capture_lift5"])
    bis_text = (f"At K = {k_best} " + ("the two are identical: bisecting K-means starts with one 2-way K-means split. "
                                       if k_best == 2 else "they are compared in the table. ")
                + f"At K = {k_fine}, K-means has {'lower' if km_wins[0] else 'higher'} SSE and "
                f"{'captures more' if km_wins[1] else 'captures fewer'} mules in high-lift clusters. Bisecting K-means "
                "never revisits a split once made, which usually costs it some SSE; it is faster and its clusters "
                "form a hierarchy.")
    drops = sweep.sse.diff().abs()
    big = int(sweep.k[drops.idxmax()])

    sections = [
        ("Headline: do mules cluster together without labels?", [
            f"{n_active:,} active accounts, {n_mules} of them mules (base rate {_pct(base, 2)}). Clustering never "
            "sees a label; true roles are joined afterwards. Lift = a cluster's mule share / the base rate.",
            f"At K = {k_fine}, K-means puts {int(top.mules)} mules in one cluster of {int(top['size']):,} accounts "
            f"({_pct(top.mule_share)} mules, lift {top.lift:.1f}). {_pct(capture[('kmeans', k_fine)], 0)} of all "
            f"mules sit in clusters with lift >= 5 (bisecting K-means: {_pct(capture[('bisecting_kmeans', k_fine)], 0)}). "
            f"At K = {k_best}, the small cluster holds {_pct(capture[('kmeans', k_best)], 0)} of mules at lift "
            f"{lift[('kmeans', k_best)].lift.iloc[0]:.1f}.", verdict,
            f"K-means, K = {k_fine}, clusters by lift:", fine.head(8), fig_dir / f"cluster_lift_kmeans_k{k_fine}.png"]),
        ("1. Features and scaling", [
            f"Flow and network features plus night share and burstiness ({len(CLUSTER_FEATURES)} columns): "
            + ", ".join(f"`{c}`" for c in CLUSTER_FEATURES) + ". Heavy-tailed columns get `log1p` first ("
            + ", ".join(f"`{c}`" for c in LOG_FEATURES) + "), then every column is standardised so no single "
            "feature dominates the distance. Only active accounts (money in and out of the window) are clustered; "
            "the rest have no flow to describe. `prepare()` refuses `label`, `role` and `is_fraud`."]),
        ("2. K-means and choosing K", [
            fig_dir / "cluster_sweep.png",
            f"The largest SSE drop is from K = {big - 1} to {big}, after which it flattens gradually rather than at "
            f"one sharp elbow. Silhouette (on a 5,000-account sample) is highest at "
            f"K = {best_sil_k}, so that is the reported K. Because a 2-way split cannot show the finer mule groups, "
            f"the report also uses K = {k_fine}, the top of the sweep. K is never chosen using labels; "
            "`mule_capture_lift5` is shown only for reading.", sweep]),
        ("3. Bisecting K-means vs K-means", [
            bis_text,
            v[["method", "k", "sse", "silhouette", "top_lift", "mule_capture_lift5"]],
            f"Bisecting K-means, K = {k_fine}, clusters by lift:", lift[("bisecting_kmeans", k_fine)].head(5)]),
        ("4. Proximity matrix", [
            "Pairwise distances of a 2,000-account sample stratified by cluster, rows and columns sorted by cluster. "
            "Light blocks on the diagonal are tight clusters; dark bands are small, far-out clusters.",
            *[fig_dir / f"cluster_proximity_k{k}.png" for k in ks]]),
        ("5. Cohesion and separation", [
            "SSE is within-cluster scatter (cohesion), SSB is between-cluster scatter (separation); SSE + SSB is the "
            "fixed total, so lowering one raises the other.", v[["method", "k", "sse", "ssb", "silhouette"]]]),
        ("6. Supervised validity against true roles", [
            "About 99% of accounts are not mules, so purity is high for any clustering and ARI stays near zero even "
            "when mules are concentrated. That is why lift is the headline.",
            v[["method", "k", "purity_role", "entropy_role", "ari_role", "ari_mule"]]]),
        ("7. Significance: shuffled baseline", [
            sig_text, sig]),
        (f"8. Cluster profiles (K = {k_fine}, sorted by mule share)", [
            profile_text, prof]),
        ("K-means as an optimisation problem", [
            "K-means minimises SSE, the sum of squared distances from each point to its cluster's centroid. Lloyd's "
            "algorithm alternates two steps that can only lower SSE: assign each point to its nearest centroid, then "
            "move each centroid to the mean of its points. It always converges, but to a local minimum that depends "
            "on the start, so each fit keeps the best of 10 k-means++ starts."]),
        ("Strengths, weaknesses and cluster types", [
            "- **Strengths:** simple, fast (linear in the number of accounts), and easy to explain; centroids are "
            "readable profiles.\n"
            "- **Weaknesses:** K must be chosen; it is sensitive to scaling and outliers (hence log1p and "
            "standardising); and it only finds local minima.\n"
            "- **Cluster types:** K-means favours round clusters of similar size and spread. Mules are a small, "
            f"varied group among {n_active - n_mules:,} other accounts, so K-means splits them across clusters and "
            "merges them with similar-looking hard cases. Density-based methods would handle that shape better; "
            "they are outside this unit."]),
    ]
    write_markdown_report(path, "Clustering accounts without labels (Person A)", sections)
    return Path(path)
