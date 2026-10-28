import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from mulehunt.clustering import (CLUSTER_FEATURES, active_accounts, cluster_entropy, cluster_lift,
                                 cluster_profiles, cohesion_separation, fit_bisecting, fit_kmeans, kmeans_sweep,
                                 mule_capture, prepare, proximity_matrix_figure, purity, shuffled_baseline)


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=3)
    return X, y


def frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, len(CLUSTER_FEATURES))) * 100, columns=CLUSTER_FEATURES)
    df.insert(0, "account_id", [f"ACC{i:06d}" for i in range(n)])
    return df


def test_purity_perfect():
    assert purity(["a", "a", "b", "b"], [0, 0, 1, 1]) == 1.0


def test_purity_half():
    assert purity(["a", "b", "a", "b"], [0, 0, 1, 1]) == 0.5


def test_entropy_zero_for_pure():
    assert cluster_entropy(["a", "a", "b", "b"], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert cluster_entropy(["a", "b", "a", "b"], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_sweep_finds_blobs(blobs):
    X, _ = blobs
    sweep = kmeans_sweep(X, ks=range(2, 8))
    assert list(sweep.columns) == ["k", "sse", "silhouette"]
    assert sweep.loc[sweep.silhouette.idxmax(), "k"] == 4
    assert sweep.sse.is_monotonic_decreasing


def test_shuffle_destroys_structure(blobs):
    X, _ = blobs
    real = cohesion_separation(X, fit_kmeans(X, 4).labels_)["silhouette"]
    shuffled = shuffled_baseline(X, 4, n_repeats=5)
    assert list(shuffled.columns) == ["sse", "silhouette"] and len(shuffled) == 5
    assert shuffled.silhouette.mean() < real - 0.2


def test_cohesion_separation_adds_up(blobs):
    X, _ = blobs
    m = cohesion_separation(X, fit_kmeans(X, 4).labels_)
    total = ((X - X.mean(axis=0)) ** 2).sum()
    assert m["sse"] + m["ssb"] == pytest.approx(total)


@pytest.mark.parametrize("col", ["label", "role", "is_fraud"])
def test_cluster_without_labels(col):
    df = frame()
    df[col] = 1
    with pytest.raises(ValueError):
        prepare(df, columns=CLUSTER_FEATURES + [col])


def test_prepare_scales_and_logs():
    X, scaler = prepare(frame(200))
    assert X.shape == (200, len(CLUSTER_FEATURES))
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)
    assert hasattr(scaler, "mean_")


def test_bisecting_k_clusters(blobs):
    X, _ = blobs
    assert len(set(fit_bisecting(X, 5).labels_)) == 5


def test_cluster_lift():
    labels = np.array([0] * 20 + [1] * 80)
    is_mule = pd.Series([True] * 8 + [False] * 12 + [True] * 2 + [False] * 78)
    lift = cluster_lift(labels, is_mule)
    assert list(lift.columns) == ["cluster", "size", "mules", "mule_share", "lift"]
    top = lift.iloc[0]
    assert top.cluster == 0 and top["size"] == 20
    assert top.mule_share == pytest.approx(0.4)
    assert top.lift == pytest.approx(4.0)
    assert mule_capture(lift, min_lift=3) == pytest.approx(0.8)
    assert mule_capture(lift, min_lift=5) == 0.0


def test_active_accounts_filter():
    df = frame(4)
    df["total_inflow"] = [10, 0, 5, 7]
    df["total_outflow"] = [3, 4, 0, 1]
    assert active_accounts(df).account_id.tolist() == ["ACC000000", "ACC000003"]


def test_cluster_profiles():
    df = frame(6)
    labels = np.array([0, 0, 0, 1, 1, 1])
    roles = pd.Series(["normal", "mule_t1", "normal", "merchant", "merchant", "normal"])
    prof = cluster_profiles(df, labels, roles)
    assert len(prof) == 2
    assert prof.loc[0, "role_mule_t1"] == pytest.approx(1 / 3)
    assert prof.loc[1, "role_merchant"] == pytest.approx(2 / 3)
    assert prof.loc[0, "total_inflow"] == pytest.approx(df.total_inflow[:3].mean())


def test_proximity_figure(blobs, tmp_path):
    X, y = blobs
    path = tmp_path / "prox.png"
    proximity_matrix_figure(X, y, n_sample=200, path=path)
    assert path.stat().st_size > 0


def test_study_end_to_end(tmp_path):
    from mulehunt.clustering import run_clustering_study

    df = frame(300, seed=1)
    df.loc[:9, "total_inflow"] = 0                      # inactive, dropped
    df.loc[10:39, ["pass_through_ratio", "fan_in_max_24h"]] = [0.95, 400.0]   # a mule-like group
    df["label"], df["split"] = 0, "train"
    df.to_parquet(tmp_path / "f.parquet")
    roles = ["normal"] * 300
    roles[10:40] = ["mule_t1"] * 30
    pd.DataFrame({"account_id": df.account_id, "role": roles}).to_csv(tmp_path / "a.csv", index=False)
    out = run_clustering_study(tmp_path / "f.parquet", tmp_path / "a.csv", ks=range(2, 6),
                               fig_dir=tmp_path / "fig", tab_dir=tmp_path / "tab", n_repeats=2)
    assert out["n_active"] == 290 and out["n_mules"] == 30
    assert out["report_ks"] == sorted({out["k_best"], 5})
    assert set(out["validity"].method) == {"kmeans", "bisecting_kmeans"}
    assert max(lift.lift.iloc[0] for lift in out["lifts"].values()) >= 5
    for k in out["report_ks"]:
        assert (tmp_path / "fig" / f"cluster_proximity_k{k}.png").exists()
        assert (tmp_path / "tab" / f"cluster_lift_kmeans_k{k}.csv").exists()
    assert (tmp_path / "fig" / "cluster_sweep.png").exists()
