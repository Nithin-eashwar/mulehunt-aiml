import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.datasets import make_classification

from mulehunt.features import FEATURE_COLUMNS
from mulehunt.models import (
    choose_threshold,
    evaluate,
    export_tree_rules,
    load_groups,
    load_splits,
    make_baseline_lr,
    prune_tree,
    run_tree_study,
    tree_depth_curve,
)


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    X, y = make_classification(n_samples=6000, n_features=len(FEATURE_COLUMNS), n_informative=6, n_redundant=4,
                               weights=[0.97, 0.03], flip_y=0.01, random_state=0)
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df.insert(0, "account_id", [f"ACC{i:06d}" for i in range(len(df))])
    df["label"] = y
    df["split"] = np.random.default_rng(0).choice(["train", "val", "test"], size=len(df), p=[0.5, 0.25, 0.25])
    path = tmp_path_factory.mktemp("feat") / "features.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def splits(parquet):
    return load_splits(parquet)


def test_load_splits(splits):
    assert set(splits) == {"train", "val", "test"}
    X, y, ids = splits["train"]
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(ids)
    assert ids.str.startswith("ACC").all()


def test_load_splits_column_subset(parquet):
    X, _, _ = load_splits(parquet, columns=FEATURE_COLUMNS[:4])["val"]
    assert list(X.columns) == FEATURE_COLUMNS[:4]


def test_load_groups(tmp_path):
    meta = tmp_path / "account_split.csv"
    meta.write_text("account_id,split,ring_id\nACC1,train,3\nACC2,train,3\nACC3,val,\nACC4,test,7\n")
    groups = load_groups(pd.Series(["ACC1", "ACC2", "ACC3", "ACC4"]), meta)
    assert list(groups) == ["ring-3", "ring-3", "ACC3", "ring-7"]


def test_evaluate_perfect():
    y = np.array([0, 0, 1, 1, 0, 1])
    m = evaluate(y, y.astype(float), 0.5)
    assert set(m) == {"precision", "recall", "f1", "pr_auc", "recall_at_1pct_fpr"}
    assert all(v == pytest.approx(1.0) for v in m.values())


def test_evaluate_all_negative_predictions():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.3, 0.4]), 0.9)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_recall_at_fpr():
    # 90 negatives scored 0.00..0.89; 5 positives above all negatives, 5 in the middle.
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.arange(90) / 100, [0.99, 0.98, 0.97, 0.96, 0.95], [0.505, 0.515, 0.525, 0.535, 0.545]])
    # 1% of 90 negatives is 0.9, so no false positive is allowed: only the top 5 positives count.
    assert evaluate(y, p, 0.5)["recall_at_1pct_fpr"] == pytest.approx(0.5)


def test_choose_threshold_in_unit_interval(splits):
    X, y, _ = splits["val"]
    p = make_baseline_lr().fit(*splits["train"][:2]).predict_proba(X)[:, 1]
    t = choose_threshold(y, p)
    assert 0.0 <= t <= 1.0
    best = evaluate(y, p, t)["f1"]
    for other in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert best >= evaluate(y, p, other)["f1"] - 1e-9


def test_depth_curve_shape(splits):
    curve = tree_depth_curve(splits)
    assert list(curve.columns) == ["depth", "train_f1", "val_f1", "train_pr_auc", "val_pr_auc"]
    assert len(curve) == 20
    assert spearmanr(curve.depth, curve.train_f1).statistic > 0.8
    gap = (curve.train_f1 - curve.val_f1).to_numpy()
    assert gap[19] >= gap[1]


def test_depth_curve_entropy(splits):
    curve = tree_depth_curve(splits, depths=range(1, 6), criterion="entropy")
    assert len(curve) == 5 and curve.val_pr_auc.between(0, 1).all()


def test_prune_tree_fewer_leaves(splits):
    from sklearn.tree import DecisionTreeClassifier
    X, y, _ = splits["train"]
    full = DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=0).fit(X, y)
    pruned, path = prune_tree(splits, max_depth=12)
    assert pruned.get_n_leaves() <= full.get_n_leaves()
    assert list(path.columns) == ["alpha", "n_leaves", "val_pr_auc"]
    assert path.val_pr_auc.max() == pytest.approx(path.set_index("alpha").val_pr_auc[pruned.ccp_alpha])


def test_export_tree_rules(splits):
    pruned, _ = prune_tree(splits, max_depth=4)
    text = export_tree_rules(pruned, FEATURE_COLUMNS)
    assert "class:" in text and any(c in text for c in FEATURE_COLUMNS)


def test_run_tree_study_outputs(splits, tmp_path):
    out = run_tree_study(splits, tmp_path / "fig", tmp_path / "tab")
    for name in ("tree_depth_curve.png", "tree_pruning_path.png", "tree.png"):
        assert (tmp_path / "fig" / name).exists()
    for name in ("tree_depth_curve.csv", "tree_pruning_path.csv", "tree_rules.txt"):
        assert (tmp_path / "tab" / name).exists()
    assert {"best_depth", "ccp_alpha", "n_leaves", "val_pr_auc"} <= set(out)


def test_baseline_lr_fast():
    X, y = make_classification(n_samples=20000, n_features=20, weights=[0.97], random_state=1)
    start = time.perf_counter()
    make_baseline_lr().fit(X, y)
    assert time.perf_counter() - start < 2.0
