"""Supervised fraud models (syllabus Units III and IV).

    python -m mulehunt.models --tree      decision-tree study (depth curve, pruning, rulebook)

Every model uses the same features (mulehunt.features) and the same account-
disjoint, time-based splits. Accuracy is never a headline metric: with about 1%
fraud, predicting "not fraud" for everyone is 99% accurate and catches nothing.

Person B imports `make_baseline_lr` as the fast fitness function for local search.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, \
    recall_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from mulehunt.features import FEATURE_COLUMNS

FEATURES = Path("data/processed/features.parquet")
META = Path("data/generated/_meta/account_split.csv")
FIGURES = Path("reports/a_figures")
TABLES = Path("reports/a_tables")
SPLITS = ("train", "val", "test")
FPR_LIMIT = 0.01
MAX_PRUNE_ALPHAS = 50
TREE_PLOT_DEPTH = 4


# ---------------------------------------------------------------- data

def load_splits(path=FEATURES, columns: list[str] | None = None) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """split -> (X, y, account_id), all models use exactly these rows."""
    columns = list(columns or FEATURE_COLUMNS)
    table = pd.read_parquet(path)
    out = {}
    for split in SPLITS:
        part = table[table.split == split].reset_index(drop=True)
        out[split] = (part[columns], part.label.astype(int), part.account_id)
    return out


def load_groups(account_ids: pd.Series, meta_path=META) -> pd.Series:
    """Fraud ring of each account ("ring-<id>"), or the account itself outside a ring.
    Used to keep ring-mates together in folds and bootstrap resamples."""
    meta = pd.read_csv(meta_path, dtype={"ring_id": "Int64"}).set_index("account_id").ring_id
    ring = account_ids.map(meta)
    return pd.Series(np.where(ring.isna(), account_ids, "ring-" + ring.astype("string")), index=account_ids.index)


# ---------------------------------------------------------------- metrics

def make_baseline_lr(random_state: int = 0) -> Pipeline:
    """Fast, reproducible logistic regression (Person B's local-search fitness)."""
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state))])


def evaluate(y_true, p, threshold: float) -> dict[str, float]:
    y_true, p = np.asarray(y_true), np.asarray(p, dtype=float)
    pred = p >= threshold
    fpr, tpr, _ = roc_curve(y_true, p)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, p),
        "recall_at_1pct_fpr": float(tpr[fpr <= FPR_LIMIT].max()),
    }


def choose_threshold(y_val, p_val) -> float:
    """The decision threshold that maximises F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(np.clip(thresholds[int(np.argmax(f1))], 0.0, 1.0))


# ---------------------------------------------------------------- decision tree (Unit III)

def _tree(max_depth, criterion="gini", ccp_alpha=0.0):
    return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, ccp_alpha=ccp_alpha,
                                  class_weight="balanced", random_state=0)


def tree_depth_curve(splits, depths=range(1, 21), criterion: str = "gini") -> pd.DataFrame:
    """Training vs validation performance as the tree grows: the overfitting curve."""
    (X_tr, y_tr, _), (X_va, y_va, _) = splits["train"], splits["val"]
    rows = []
    for depth in depths:
        tree = _tree(depth, criterion).fit(X_tr, y_tr)
        p_tr, p_va = tree.predict_proba(X_tr)[:, 1], tree.predict_proba(X_va)[:, 1]
        rows.append((depth, f1_score(y_tr, p_tr >= 0.5, zero_division=0), f1_score(y_va, p_va >= 0.5, zero_division=0),
                     average_precision_score(y_tr, p_tr), average_precision_score(y_va, p_va)))
    return pd.DataFrame(rows, columns=["depth", "train_f1", "val_f1", "train_pr_auc", "val_pr_auc"])


def prune_tree(splits, max_depth: int) -> tuple[DecisionTreeClassifier, pd.DataFrame]:
    """Cost-complexity pruning: keep the alpha with the best validation PR-AUC
    (the larger alpha, i.e. the simpler tree, on ties)."""
    (X_tr, y_tr, _), (X_va, y_va, _) = splits["train"], splits["val"]
    alphas = _tree(max_depth).cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas
    alphas = np.unique(np.quantile(np.maximum(alphas, 0.0), np.linspace(0, 1, MAX_PRUNE_ALPHAS)))
    rows, best = [], None
    for alpha in alphas:
        tree = _tree(max_depth, ccp_alpha=alpha).fit(X_tr, y_tr)
        score = average_precision_score(y_va, tree.predict_proba(X_va)[:, 1])
        rows.append((alpha, tree.get_n_leaves(), score))
        if best is None or score >= best[0]:
            best = (score, tree)
    return best[1], pd.DataFrame(rows, columns=["alpha", "n_leaves", "val_pr_auc"])


def export_tree_rules(tree, feature_names) -> str:
    return export_text(tree, feature_names=list(feature_names), show_weights=False, decimals=3)


def run_tree_study(splits, fig_dir=FIGURES, tab_dir=TABLES) -> dict:
    """Depth curve (Gini and entropy), pruning path and the pruned tree as a 'mule rulebook'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    gini = tree_depth_curve(splits, criterion="gini")
    entropy = tree_depth_curve(splits, criterion="entropy")
    curve = gini.merge(entropy, on="depth", suffixes=("", "_entropy"))
    curve.to_csv(tab_dir / "tree_depth_curve.csv", index=False)
    best_depth = int(gini.depth[gini.val_pr_auc.idxmax()])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gini.depth, gini.train_pr_auc, "o-", color="#4C72B0", label="train PR-AUC (Gini)")
    ax.plot(gini.depth, gini.val_pr_auc, "s-", color="#C44E52", label="validation PR-AUC (Gini)")
    ax.plot(entropy.depth, entropy.val_pr_auc, "s--", color="#DD8452", label="validation PR-AUC (entropy)")
    ax.axvline(best_depth, color="grey", ls=":", label=f"best validation depth = {best_depth}")
    ax.set(xlabel="max depth", ylabel="PR-AUC", title="Decision tree: overfitting as depth grows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "tree_depth_curve.png", dpi=120)
    plt.close(fig)

    pruned, path = prune_tree(splits, max_depth=int(gini.depth.max()))
    path.to_csv(tab_dir / "tree_pruning_path.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(path.n_leaves, path.val_pr_auc, "o-", color="#4C72B0")
    ax.axvline(pruned.get_n_leaves(), color="#C44E52", ls=":", label=f"chosen: {pruned.get_n_leaves()} leaves")
    ax.set_xscale("log")
    ax.set(xlabel="leaves (log scale)", ylabel="validation PR-AUC", title="Cost-complexity pruning path")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "tree_pruning_path.png", dpi=120)
    plt.close(fig)

    names = list(splits["train"][0].columns)
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(pruned, feature_names=names, class_names=["normal", "fraud"], max_depth=TREE_PLOT_DEPTH,
              filled=True, impurity=False, proportion=True, fontsize=7, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "tree.png", dpi=110)
    plt.close(fig)
    (tab_dir / "tree_rules.txt").write_text(export_tree_rules(pruned, names), encoding="utf-8", newline="\n")

    X_va, y_va, _ = splits["val"]
    return {"best_depth": best_depth, "best_depth_val_pr_auc": float(gini.val_pr_auc.max()),
            "ccp_alpha": float(pruned.ccp_alpha), "n_leaves": int(pruned.get_n_leaves()),
            "val_pr_auc": float(average_precision_score(y_va, pruned.predict_proba(X_va)[:, 1]))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the fraud models")
    parser.add_argument("--features", default=str(FEATURES))
    parser.add_argument("--tree", action="store_true", help="run the decision-tree study")
    args = parser.parse_args()
    splits = load_splits(args.features)
    if args.tree:
        print(run_tree_study(splits))


if __name__ == "__main__":
    main()
