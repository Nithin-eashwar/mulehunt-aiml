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


# ---------------------------------------------------------------- Task 10: model zoo and analyses

from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from mulehunt.features import TXN_ONLY  # noqa: E402
from mulehunt.models import (  # noqa: E402
    MODEL_NAMES,
    FittedModel,
    ablation,
    bootstrap_ci,
    build_models,
    fit_all,
    imbalance_comparison,
    leaderboard,
    lr_coefficients,
    per_scenario_table,
    single_feature_pr_auc,
)

ONE_SETTING = {name: [{}] for name in MODEL_NAMES}


@pytest.fixture(scope="module")
def fitted(splits):
    return fit_all(build_models(), splits, grids=ONE_SETTING)


def test_build_models_keys():
    assert set(build_models()) == {"decision_tree", "knn", "naive_bayes", "logistic_regression", "bagging",
                                   "adaboost", "gradient_boosting", "random_forest"}
    assert list(build_models()) == list(MODEL_NAMES)


def test_grids_are_small():
    from mulehunt.models import MODEL_GRIDS
    assert set(MODEL_GRIDS) == set(MODEL_NAMES)
    assert all(1 <= len(grid) <= 12 for grid in MODEL_GRIDS.values())


def test_all_models_predict_proba(fitted, splits):
    X_va = splits["val"][0]
    for name, fm in fitted.items():
        p = fm.pipeline.predict_proba(X_va)
        assert p.shape == (len(X_va), 2), name
        assert ((p >= 0) & (p <= 1)).all(), name
        assert 0 <= fm.threshold <= 1 and "pr_auc" in fm.val_metrics


def test_smote_only_on_train(splits, monkeypatch):
    calls = []
    original = SMOTE.fit_resample

    def counting(self, X, y):
        calls.append(len(X))
        return original(self, X, y)

    monkeypatch.setattr(SMOTE, "fit_resample", counting)
    (X_tr, y_tr, _), (X_va, _, _) = splits["train"], splits["val"]
    knn = build_models()["knn"].fit(X_tr, y_tr)
    assert calls == [len(X_tr)]
    assert len(knn.predict_proba(X_va)) == len(X_va)
    assert calls == [len(X_tr)]


def test_fit_all_picks_best_setting(splits):
    grids = {"logistic_regression": [{"model__C": 1e-4}, {"model__C": 1.0}]}
    fm = fit_all({"logistic_regression": build_models()["logistic_regression"]}, splits, grids=grids)
    X_va, y_va, _ = splits["val"]
    scores = []
    for params in grids["logistic_regression"]:
        m = build_models()["logistic_regression"].set_params(**params).fit(*splits["train"][:2])
        scores.append(average_precision_score(y_va, m.predict_proba(X_va)[:, 1]))
    assert fm["logistic_regression"].val_metrics["pr_auc"] == pytest.approx(max(scores))
    assert fm["logistic_regression"].params == grids["logistic_regression"][int(np.argmax(scores))]


def test_leaderboard_no_accuracy(fitted, splits):
    board = leaderboard(fitted, splits, n_boot=50)
    assert "accuracy" not in board.columns
    assert len(board) == 8 and set(board.model) == set(MODEL_NAMES)
    assert board.pr_auc.is_monotonic_decreasing
    for col in ("precision", "recall", "f1", "pr_auc", "recall_at_1pct_fpr",
                "pr_auc_ci_low", "pr_auc_ci_high", "f1_ci_low", "f1_ci_high", "threshold"):
        assert col in board.columns
    assert (board.pr_auc_ci_low <= board.pr_auc + 1e-9).all() and (board.pr_auc <= board.pr_auc_ci_high + 1e-9).all()


def ap(y, p, w):
    return average_precision_score(y, p, sample_weight=w)


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = np.array([0] * 400 + [1] * 40)
    p = np.concatenate([rng.uniform(0, 0.7, 400), rng.uniform(0.3, 1.0, 40)])
    lo, hi = bootstrap_ci(y, p, ap, pd.Series(np.arange(len(y))), n=300)
    assert lo <= average_precision_score(y, p) <= hi


def test_grouped_bootstrap_wider():
    rng = np.random.default_rng(1)
    pos_scores = np.repeat(rng.uniform(0.2, 1.0, 10), 8)  # 10 rings x 8 identical members
    y = np.array([0] * 920 + [1] * 80)
    p = np.concatenate([rng.uniform(0, 0.8, 920), pos_scores])
    groups = pd.Series([f"n{i}" for i in range(920)] + [f"ring{i // 8}" for i in range(80)])
    lo_g, hi_g = bootstrap_ci(y, p, ap, groups, n=400, seed=2)
    lo_r, hi_r = bootstrap_ci(y, p, ap, pd.Series(np.arange(1000)), n=400, seed=2)
    assert hi_g - lo_g > hi_r - lo_r


def test_single_feature_pr_auc_rows(splits):
    table = single_feature_pr_auc(splits)
    assert list(table.columns) == ["feature", "val_pr_auc"]
    assert sorted(table.feature) == sorted(FEATURE_COLUMNS)
    assert table.val_pr_auc.between(0, 1).all() and table.val_pr_auc.is_monotonic_decreasing


def test_imbalance_comparison(splits):
    table = imbalance_comparison(splits)
    assert list(table.method) == ["none", "class_weight", "smote"]
    assert {"val_pr_auc", "test_pr_auc", "test_recall", "test_f1"} <= set(table.columns)


def test_ablation_rows(parquet):
    table = ablation(load_splits(parquet), load_splits(parquet, columns=TXN_ONLY), model_name="logistic_regression")
    assert list(table.features) == ["transaction_only", "with_network"]
    assert list(table.n_features) == [len(TXN_ONLY), len(FEATURE_COLUMNS)]
    assert table.test_pr_auc.between(0, 1).all()


def test_per_scenario_empty_scenario(fitted, splits):
    _, y_te, ids = splits["test"]
    positives = ids[y_te == 1].to_list()
    scenarios = pd.Series({a: ("fake_kyc" if i % 2 else "job_task") for i, a in enumerate(positives)})
    table = per_scenario_table(fitted["logistic_regression"], splits, scenarios).set_index("scenario")
    assert list(table.columns) == ["n_fraud", "recall", "mean_risk_prob"]
    assert len(table) == 5
    assert table.loc["loan_app", "n_fraud"] == 0 and np.isnan(table.loc["loan_app", "recall"])
    assert table.loc["fake_kyc", "n_fraud"] + table.loc["job_task", "n_fraud"] == len(positives)
    assert 0 <= table.loc["fake_kyc", "recall"] <= 1


def test_lr_coefficients(fitted):
    table = lr_coefficients(fitted["logistic_regression"])
    assert list(table.columns) == ["feature", "coefficient", "odds_ratio"]
    assert sorted(table.feature) == sorted(FEATURE_COLUMNS)
    assert np.allclose(table.odds_ratio, np.exp(table.coefficient))
    assert table.coefficient.abs().is_monotonic_decreasing


# ---------------------------------------------------------------- Task 11: risk scores, saving, explain

import json  # noqa: E402

from mulehunt.models import explain, load_model, oof_folds, save_models, write_risk_scores  # noqa: E402


def ring_groups(ids, y, size=4):
    """Positives in rings of `size`; everyone else is their own group."""
    pos = ids[y == 1].reset_index(drop=True)
    ring = {a: f"ring-{i // size}" for i, a in enumerate(pos)}
    return ids.map(lambda a: ring.get(a, a))


def test_risk_scores_columns_and_coverage(fitted, splits, parquet, tmp_path):
    X_tr, y_tr, ids_tr = splits["train"]
    out = write_risk_scores(fitted["logistic_regression"], splits, tmp_path / "risk.csv",
                            groups=ring_groups(ids_tr, y_tr))
    on_disk = pd.read_csv(tmp_path / "risk.csv")
    assert list(on_disk.columns) == ["account_id", "risk_prob"]
    assert len(on_disk) == len(pd.read_parquet(parquet)) and on_disk.account_id.is_unique
    assert len(out) == len(on_disk)


def test_risk_scores_clipped(splits, tmp_path):
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    X_tr, y_tr, _ = splits["train"]
    tree = Pipeline([("model", DecisionTreeClassifier(random_state=0))]).fit(X_tr, y_tr)  # outputs exactly 0 or 1
    out = write_risk_scores(FittedModel("tree", tree, 0.5, {}), splits, tmp_path / "risk.csv")
    on_disk = pd.read_csv(tmp_path / "risk.csv")
    for frame in (out, on_disk):
        assert frame.risk_prob.min() >= 1e-6 and frame.risk_prob.max() <= 1 - 1e-6
        assert np.isfinite(-np.log(frame.risk_prob)).all()


def test_train_scores_are_out_of_fold(fitted, splits, tmp_path):
    fm = fitted["random_forest"]
    X_tr, y_tr, ids_tr = splits["train"]
    out = write_risk_scores(fm, splits, tmp_path / "risk.csv").set_index("account_id")
    in_sample = np.clip(fm.predict_proba(X_tr), 1e-6, 1 - 1e-6)
    assert (np.abs(out.risk_prob.loc[ids_tr].to_numpy() - in_sample) > 1e-6).any()
    assert len(np.unique(oof_folds(y_tr, pd.Series(np.arange(len(y_tr))), 5))) == 5


def test_val_test_scores_from_train_model(fitted, splits, tmp_path):
    fm = fitted["logistic_regression"]
    out = write_risk_scores(fm, splits, tmp_path / "risk.csv").set_index("account_id")
    X_te, _, ids_te = splits["test"]
    assert np.allclose(out.risk_prob.loc[ids_te].to_numpy(), np.clip(fm.predict_proba(X_te), 1e-6, 1 - 1e-6))


def test_oof_folds_keep_rings_together(splits):
    _, y_tr, ids_tr = splits["train"]
    groups = ring_groups(ids_tr, y_tr)
    folds = pd.Series(oof_folds(y_tr, groups, 5), index=groups.index)
    assert (folds.groupby(groups.to_numpy()).nunique() == 1).all()


def test_save_load_roundtrip(fitted, splits, tmp_path):
    subset = {k: fitted[k] for k in ("logistic_regression", "random_forest", "knn")}
    save_models(subset, splits, tmp_path)
    X_va = splits["val"][0]
    for name, fm in subset.items():
        bundle = load_model(name, tmp_path)
        assert set(bundle) >= {"pipeline", "threshold", "feature_columns"}
        assert bundle["threshold"] == fm.threshold
        assert np.allclose(bundle["pipeline"].predict_proba(X_va)[:, 1], fm.predict_proba(X_va))
    assert json.loads((tmp_path / "feature_columns.json").read_text()) == FEATURE_COLUMNS
    best = (tmp_path / "best_model.txt").read_text().strip()
    assert best == max(subset, key=lambda k: subset[k].val_metrics["pr_auc"])


def test_explain_returns_top_k_known_features(fitted, splits, tmp_path):
    save_models({k: fitted[k] for k in ("logistic_regression", "random_forest", "knn")}, splits, tmp_path)
    row = splits["test"][0].iloc[0]
    for name in ("logistic_regression", "random_forest", "knn"):
        out = explain(load_model(name, tmp_path), row, top_k=5)
        assert len(out) == 5
        assert all(f in FEATURE_COLUMNS and np.isfinite(v) for f, v in out)
        assert [abs(v) for _, v in out] == sorted((abs(v) for _, v in out), reverse=True)


def test_contracts_loading_snippet_runs(capsys):
    # The dashboard snippet in CONTRACTS.md §5 must run as written against the committed models.
    import re
    from pathlib import Path

    if not Path("data/processed/features.parquet").exists():
        pytest.skip("real features not built (python -m mulehunt.features)")
    text = Path("CONTRACTS.md").read_text(encoding="utf-8")
    section = text[text.index("## 5. Saved models"):text.index("## 6.")]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    exec(compile(code, "CONTRACTS.md", "exec"), {})
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith(Path("models/best_model.txt").read_text().strip() + ": risk ")
    assert len(lines) == 6
