"""Supervised fraud models (syllabus Units III and IV).

    python -m mulehunt.models --tree      decision-tree study (depth curve, pruning, rulebook)

Every model uses the same features (mulehunt.features) and the same account-
disjoint, time-based splits. Accuracy is never a headline metric: with about 1%
fraud, predicting "not fraud" for everyone is 99% accurate and catches nothing.

Person B imports `make_baseline_lr` as the fast fitness function for local search.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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
    curve.to_csv(tab_dir / "tree_depth_curve.csv", index=False, lineterminator="\n")
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
    path.to_csv(tab_dir / "tree_pruning_path.csv", index=False, lineterminator="\n")
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


# ---------------------------------------------------------------- the model zoo (Units III and IV)

MODEL_NAMES = ("decision_tree", "knn", "naive_bayes", "logistic_regression", "bagging", "adaboost",
               "gradient_boosting", "random_forest")

# Validation grids (<= 12 settings each). Imbalance handling (plan D15): balanced class
# weights everywhere except KNN (SMOTE on training rows only) and Naive Bayes (data priors).
MODEL_GRIDS = {
    "decision_tree": [{"model__max_depth": d} for d in (4, 6, 7, 8, 10)],
    "knn": [{"model__n_neighbors": k} for k in (3, 5, 7, 9, 11, 15, 19, 23, 27, 31)],
    "naive_bayes": [{"model__var_smoothing": v} for v in (1e-9, 1e-7, 1e-5, 1e-3)],
    "logistic_regression": [{"model__C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "bagging": [{"model__estimator__max_depth": d} for d in (6, 10, None)],
    "adaboost": [{"model__estimator__max_depth": d, "model__learning_rate": lr} for d in (1, 2, 3) for lr in (0.5, 1.0)],
    "gradient_boosting": [{"model__learning_rate": lr, "model__max_leaf_nodes": n} for lr in (0.05, 0.1)
                          for n in (15, 31)],
    "random_forest": [{"model__max_depth": d, "model__min_samples_leaf": m} for d in (12, None) for m in (1, 5)],
}


def build_models(random_state: int = 0) -> dict[str, Pipeline]:
    rs = random_state
    balanced_tree = DecisionTreeClassifier(class_weight="balanced", random_state=rs)
    return {
        "decision_tree": Pipeline([("model", DecisionTreeClassifier(max_depth=7, class_weight="balanced",
                                                                    random_state=rs))]),
        "knn": ImbPipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=rs)),
                            ("model", KNeighborsClassifier(n_neighbors=15))]),
        "naive_bayes": Pipeline([("model", GaussianNB())]),
        "logistic_regression": make_baseline_lr(rs),
        "bagging": Pipeline([("model", BaggingClassifier(balanced_tree, n_estimators=100, n_jobs=-1,
                                                         random_state=rs))]),
        "adaboost": Pipeline([("model", AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, class_weight="balanced",
                                                                                  random_state=rs),
                                                           n_estimators=200, algorithm="SAMME", random_state=rs))]),
        "gradient_boosting": Pipeline([("model", HistGradientBoostingClassifier(class_weight="balanced", max_iter=300,
                                                                                random_state=rs))]),
        "random_forest": Pipeline([("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                                    n_jobs=-1, random_state=rs))]),
    }


@dataclass
class FittedModel:
    name: str
    pipeline: object
    threshold: float
    val_metrics: dict
    params: dict = field(default_factory=dict)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


def fit_all(models: dict, splits, grids: dict | None = None) -> dict[str, FittedModel]:
    """Fit every setting on train, keep the one with the best validation PR-AUC, and
    fix its decision threshold on validation (max F1)."""
    grids = MODEL_GRIDS if grids is None else grids
    (X_tr, y_tr, _), (X_va, y_va, _) = splits["train"], splits["val"]
    fitted = {}
    for name, base in models.items():
        best = None
        for params in grids.get(name, [{}]):
            model = clone(base).set_params(**params).fit(X_tr, y_tr)
            p = model.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, p)
            if best is None or score > best[0]:
                best = (score, model, params, p)
        _, model, params, p = best
        threshold = choose_threshold(y_va, p)
        fitted[name] = FittedModel(name, model, threshold, evaluate(y_va, p, threshold), dict(params))
    return fitted


def bootstrap_ci(y, p, metric: Callable, groups: pd.Series, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of metric(y, p, sample_weight) when whole groups are resampled with
    replacement, so the members of a fraud ring stay together (plan D16)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    codes, _ = pd.factorize(pd.Series(groups).to_numpy())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        weight = np.bincount(rng.integers(0, codes.max() + 1, size=codes.max() + 1), minlength=codes.max() + 1)[codes]
        if weight[y == 1].sum() == 0 or weight[y == 0].sum() == 0:
            continue
        values.append(metric(y, p, weight))
    if not values:  # every resample had a single class (tiny or one-class input)
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def leaderboard(fitted: dict[str, FittedModel], splits, groups: pd.Series | None = None,
                n_boot: int = 1000) -> pd.DataFrame:
    """Test-set metrics for every model at its validation threshold, with ring-level
    bootstrap intervals. Accuracy is left out on purpose."""
    X_te, y_te, ids = splits["test"]
    groups = pd.Series(np.arange(len(y_te))) if groups is None else groups
    rows = []
    for name, fm in fitted.items():
        p = fm.predict_proba(X_te)
        m = evaluate(y_te, p, fm.threshold)
        pr_lo, pr_hi = bootstrap_ci(y_te, p, lambda y, q, w: average_precision_score(y, q, sample_weight=w),
                                    groups, n_boot)
        f1_lo, f1_hi = bootstrap_ci(y_te, p, lambda y, q, w, t=fm.threshold: f1_score(y, q >= t, sample_weight=w,
                                                                                     zero_division=0),
                                    groups, n_boot)
        rows.append({"model": name, **m, "pr_auc_ci_low": pr_lo, "pr_auc_ci_high": pr_hi,
                     "f1_ci_low": f1_lo, "f1_ci_high": f1_hi, "threshold": fm.threshold,
                     "val_pr_auc": fm.val_metrics["pr_auc"], "params": str(fm.params)})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False, ignore_index=True)


def single_feature_pr_auc(splits) -> pd.DataFrame:
    """Validation PR-AUC of each raw feature used as the score on its own (either
    direction): a feature near 1.0 would be a shortcut that gives the answer away."""
    X_va, y_va, _ = splits["val"]
    rows = [(c, max(average_precision_score(y_va, X_va[c]), average_precision_score(y_va, -X_va[c])))
            for c in X_va.columns]
    return pd.DataFrame(rows, columns=["feature", "val_pr_auc"]).sort_values("val_pr_auc", ascending=False,
                                                                            ignore_index=True)


def imbalance_comparison(splits) -> pd.DataFrame:
    """Logistic regression with no imbalance handling, balanced class weights and SMOTE."""
    variants = {
        "none": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
        "class_weight": make_baseline_lr(),
        "smote": ImbPipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=0)),
                              ("model", LogisticRegression(max_iter=1000))]),
    }
    fitted = fit_all(variants, splits, grids={k: [{}] for k in variants})
    X_te, y_te, _ = splits["test"]
    rows = []
    for name, fm in fitted.items():
        m = evaluate(y_te, fm.predict_proba(X_te), fm.threshold)
        rows.append({"method": name, "val_pr_auc": fm.val_metrics["pr_auc"], "test_pr_auc": m["pr_auc"],
                     "test_precision": m["precision"], "test_recall": m["recall"], "test_f1": m["f1"],
                     "test_recall_at_1pct_fpr": m["recall_at_1pct_fpr"]})
    return pd.DataFrame(rows)


def ablation(splits_all, splits_txn_only, model_name: str = "random_forest") -> pd.DataFrame:
    """How much the network features add over transaction-only features."""
    rows = []
    for label, splits in (("transaction_only", splits_txn_only), ("with_network", splits_all)):
        fm = fit_all({model_name: build_models()[model_name]}, splits, grids={model_name: [{}]})[model_name]
        X_te, y_te, _ = splits["test"]
        m = evaluate(y_te, fm.predict_proba(X_te), fm.threshold)
        rows.append({"features": label, "n_features": X_te.shape[1], "test_pr_auc": m["pr_auc"],
                     "test_f1": m["f1"], "test_recall_at_1pct_fpr": m["recall_at_1pct_fpr"]})
    return pd.DataFrame(rows)


def per_scenario_table(fitted_model: FittedModel, splits, scenarios: pd.Series) -> pd.DataFrame:
    """Recall on the test set's fraud accounts, one row per scam scenario."""
    from simulator.scams import SCENARIOS

    X_te, y_te, ids = splits["test"]
    p = pd.Series(fitted_model.predict_proba(X_te), index=ids.index)
    scenario = ids.map(scenarios)
    rows = []
    for s in SCENARIOS:
        mask = (y_te == 1) & (scenario == s)
        n = int(mask.sum())
        rows.append({"scenario": s, "n_fraud": n,
                     "recall": float((p[mask] >= fitted_model.threshold).mean()) if n else np.nan,
                     "mean_risk_prob": float(p[mask].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def lr_coefficients(fitted_lr) -> pd.DataFrame:
    """Logistic-regression coefficients on standardised features, read as risk factors."""
    pipeline = fitted_lr.pipeline if isinstance(fitted_lr, FittedModel) else fitted_lr
    model = pipeline.named_steps["model"]
    names = getattr(pipeline, "feature_names_in_", FEATURE_COLUMNS)
    table = pd.DataFrame({"feature": list(names), "coefficient": model.coef_[0]})
    table["odds_ratio"] = np.exp(table.coefficient)
    return table.reindex(table.coefficient.abs().sort_values(ascending=False).index).reset_index(drop=True)


def _save_tables(tab_dir: Path, **tables) -> None:
    tab_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tab_dir / f"{name}.csv", index=False, lineterminator="\n", float_format="%.6g")


def run_model_study(splits, splits_txn_only, groups, scenarios, fig_dir=FIGURES, tab_dir=TABLES,
                    n_boot: int = 1000) -> tuple[dict[str, FittedModel], pd.DataFrame]:
    """Shortcut check, model zoo, leaderboard, imbalance comparison, ablation,
    per-scenario recall, LR coefficients, PR curves and the leaderboard chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve as pr_curve

    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    shortcuts = single_feature_pr_auc(splits)
    fitted = fit_all(build_models(), splits)
    board = leaderboard(fitted, splits, groups, n_boot)
    best = max(fitted.values(), key=lambda fm: fm.val_metrics["pr_auc"])  # chosen on validation, never test
    _save_tables(tab_dir, single_feature_pr_auc=shortcuts, leaderboard=board,
                 imbalance_comparison=imbalance_comparison(splits),
                 ablation=ablation(splits, splits_txn_only),
                 per_scenario=per_scenario_table(best, splits, scenarios),
                 lr_coefficients=lr_coefficients(fitted["logistic_regression"]))

    X_te, y_te, _ = splits["test"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in board.model:
        precision, recall, _ = pr_curve(y_te, fitted[name].predict_proba(X_te))
        ax.plot(recall, precision, label=f"{name} ({board.set_index('model').pr_auc[name]:.2f})")
    ax.axhline(y_te.mean(), color="grey", ls=":", label=f"base rate ({y_te.mean():.3f})")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall curves (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = board.iloc[::-1]
    ax.barh(order.model, order.pr_auc, color="#4C72B0",
            xerr=[order.pr_auc - order.pr_auc_ci_low, order.pr_auc_ci_high - order.pr_auc], capsize=3)
    ax.set(xlabel="test PR-AUC (95% ring-level bootstrap interval)", title="Model leaderboard")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_dir / "leaderboard.png", dpi=120)
    plt.close(fig)
    return fitted, board


# ---------------------------------------------------------------- handoffs: risk scores and saved models

RISK_SCORES = Path("models/risk_scores.csv")
MODELS_DIR = Path("models")
RISK_CLIP = 1e-6
TREE_MODELS = ("decision_tree", "random_forest", "adaboost")


def oof_folds(y, groups: pd.Series, n_folds: int = 5) -> np.ndarray:
    """Fold number per training row; members of a fraud ring share one fold (plan D17)."""
    from sklearn.model_selection import StratifiedGroupKFold

    folds = np.empty(len(y), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for k, (_, held) in enumerate(splitter.split(np.zeros(len(y)), np.asarray(y), np.asarray(groups))):
        folds[held] = k
    return folds


def write_risk_scores(best: FittedModel, splits, path=RISK_SCORES, n_folds: int = 5,
                      groups: pd.Series | None = None) -> pd.DataFrame:
    """account_id, risk_prob for Person B's A* (edge cost = -log risk_prob).

    Train accounts get out-of-fold scores, so tracing is never fed memorised labels;
    val and test accounts are scored by the model fitted on train. Clipped to
    [1e-6, 1 - 1e-6] so -log is always finite."""
    X_tr, y_tr, ids_tr = splits["train"]
    groups = pd.Series(np.arange(len(y_tr))) if groups is None else groups
    folds = oof_folds(y_tr, groups, n_folds)
    oof = np.empty(len(y_tr))
    for k in range(n_folds):
        model = clone(best.pipeline).fit(X_tr[folds != k], y_tr[folds != k])
        oof[folds == k] = model.predict_proba(X_tr[folds == k])[:, 1]
    parts = [pd.DataFrame({"account_id": ids_tr, "risk_prob": oof})]
    for split in ("val", "test"):
        X, _, ids = splits[split]
        parts.append(pd.DataFrame({"account_id": ids, "risk_prob": best.predict_proba(X)}))
    out = pd.concat(parts, ignore_index=True).sort_values("account_id", ignore_index=True)
    out["risk_prob"] = out.risk_prob.clip(RISK_CLIP, 1 - RISK_CLIP)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n", float_format="%.8g")
    return out


def save_models(fitted: dict[str, FittedModel], splits, out_dir=MODELS_DIR) -> None:
    """models/<name>.joblib bundles for the dashboard, feature_columns.json and
    best_model.txt (the model with the best validation PR-AUC)."""
    import joblib
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr, _ = splits["train"]
    columns = list(X_tr.columns)
    if "random_forest" in fitted:
        fallback = fitted["random_forest"].pipeline.named_steps["model"].feature_importances_
    else:
        fallback = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1,
                                          random_state=0).fit(X_tr, y_tr).feature_importances_
    for name, fm in fitted.items():
        bundle = {"name": name, "pipeline": fm.pipeline, "threshold": fm.threshold, "feature_columns": columns,
                  "params": fm.params, "val_metrics": fm.val_metrics,
                  "feature_mean": X_tr.mean().to_numpy(), "feature_std": X_tr.std(ddof=0).replace(0, 1).to_numpy(),
                  "fallback_importances": np.asarray(fallback)}
        joblib.dump(bundle, out / f"{name}.joblib", compress=3)
    (out / "feature_columns.json").write_text(json.dumps(columns, indent=1) + "\n", encoding="utf-8", newline="\n")
    best = max(fitted, key=lambda k: fitted[k].val_metrics["pr_auc"])
    (out / "best_model.txt").write_text(best + "\n", encoding="utf-8", newline="\n")


def load_model(name: str, models_dir=MODELS_DIR) -> dict:
    import joblib
    return joblib.load(Path(models_dir) / f"{name}.joblib")


def explain(model_bundle: dict, x_row: pd.Series, top_k: int = 5) -> list[tuple[str, float]]:
    """Top contributing features for one account, largest effect first.

    Logistic regression: coefficient x standardised value. Tree models (decision tree,
    random forest, AdaBoost): feature importance x the row's standardised deviation
    from the training mean. Other models (KNN, Naive Bayes, bagging, gradient boosting)
    have no per-feature weights, so they fall back to the random forest's importances
    with the same deviation."""
    columns = model_bundle["feature_columns"]
    z = (x_row[columns].to_numpy(dtype=float) - model_bundle["feature_mean"]) / model_bundle["feature_std"]
    model = model_bundle["pipeline"].named_steps["model"] if hasattr(model_bundle["pipeline"], "named_steps") else None
    if model is not None and hasattr(model, "coef_"):
        contribution = model.coef_[0] * z
    elif model is not None and model_bundle.get("name") in TREE_MODELS and hasattr(model, "feature_importances_"):
        contribution = model.feature_importances_ * z
    else:
        contribution = model_bundle["fallback_importances"] * z
    order = np.argsort(-np.abs(contribution))[:top_k]
    return [(columns[i], float(contribution[i])) for i in order]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the fraud models")
    parser.add_argument("--features", default=str(FEATURES))
    parser.add_argument("--data", default=str(META.parents[1]), help="generated dataset (transactions, _meta)")
    parser.add_argument("--tree", action="store_true", help="run the decision-tree study")
    parser.add_argument("--models", action="store_true", help="run the model zoo, leaderboard and analyses")
    parser.add_argument("--all", action="store_true",
                        help="tree study + model study, then write models/risk_scores.csv and models/*.joblib")
    args = parser.parse_args(argv)
    splits = load_splits(args.features)
    if args.tree or args.all:
        print(run_tree_study(splits))
    if args.models or args.all:
        from mulehunt.features import TXN_ONLY, account_scenarios
        from simulator.schema import read_transactions

        print(single_feature_pr_auc(splits).head(5).to_string(index=False))
        meta = Path(args.data) / "_meta" / "account_split.csv"
        scenarios = account_scenarios(read_transactions(Path(args.data) / "transactions.csv"))
        groups = load_groups(splits["test"][2], meta)
        fitted, board = run_model_study(splits, load_splits(args.features, columns=TXN_ONLY), groups, scenarios)
        cols = ["model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "precision", "recall", "f1",
                "recall_at_1pct_fpr"]
        print(board[cols].round(3).to_string(index=False))
        if args.all:
            best = max(fitted, key=lambda k: fitted[k].val_metrics["pr_auc"])
            risk = write_risk_scores(fitted[best], splits, groups=load_groups(splits["train"][2], meta))
            save_models(fitted, splits)
            print(f"risk scores from {best}: {len(risk):,} accounts -> {RISK_SCORES}; models saved to {MODELS_DIR}/")
            from mulehunt.report import write_classifier_report

            print(f"wrote {write_classifier_report(features_path=args.features, meta_path=meta)}")


if __name__ == "__main__":
    main()
