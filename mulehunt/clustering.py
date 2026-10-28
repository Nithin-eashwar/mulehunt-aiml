"""Label-free clustering of accounts (syllabus Unit V, handoff §8).

    python -m mulehunt.clustering        sweep K, fit K-means and bisecting K-means, validity tables

Clustering never sees a label. True roles are used only afterwards, to check whether
mules land together. About 99% of accounts are not mules, so purity is high for any
clustering and ARI sits near zero; per-cluster mule lift is the number that matters (D19).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mulehunt.features import FEATURE_GROUPS

FEATURES = Path("data/processed/features.parquet")
ACCOUNTS = Path("data/generated/accounts.csv")
FIGURES = Path("reports/a_figures")
TABLES = Path("reports/a_tables")

CLUSTER_FEATURES = FEATURE_GROUPS["flow"] + FEATURE_GROUPS["network"] + ["night_txn_share", "burstiness"]
LOG_FEATURES = ["total_inflow", "total_outflow", "avg_dwell_hours", "in_degree", "out_degree",
                "fan_in_max_24h", "fan_out_max_24h"]
LABEL_COLUMNS = {"label", "role", "is_fraud"}
MULE_ROLES = ("mule_t1", "mule_t2", "mule_t3")
SILHOUETTE_SAMPLE = 5000
N_INIT = 10


# ---------------------------------------------------------------- data

def active_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Accounts with money both in and out of the window (D19); the rest have no flow to cluster."""
    return df[(df.total_inflow > 0) & (df.total_outflow > 0)].reset_index(drop=True)


def prepare(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[np.ndarray, StandardScaler]:
    """log1p on heavy-tailed columns, then standardise. Refuses label columns."""
    columns = list(columns or CLUSTER_FEATURES)
    leaked = LABEL_COLUMNS & set(columns)
    if leaked:
        raise ValueError(f"clustering must not use labels: {sorted(leaked)}")
    X = df[columns].astype(float).copy()
    for c in LOG_FEATURES:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    scaler = StandardScaler()
    return scaler.fit_transform(X.to_numpy()), scaler


# ---------------------------------------------------------------- fitting

def fit_kmeans(X, k: int, seed: int = 0) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)


def fit_bisecting(X, k: int, seed: int = 0) -> BisectingKMeans:
    return BisectingKMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)


def _silhouette(X, labels, sample: int = SILHOUETTE_SAMPLE, seed: int = 0) -> float:
    if len(set(labels)) < 2:
        return 0.0
    return float(silhouette_score(X, labels, sample_size=min(sample, len(X)), random_state=seed))


def kmeans_sweep(X, ks=range(2, 21), seed: int = 0, silhouette_sample: int = SILHOUETTE_SAMPLE,
                 labels_out: dict | None = None) -> pd.DataFrame:
    """SSE (elbow) and sampled silhouette for each K. Pass `labels_out` to keep each fit's labels."""
    rows = []
    for k in ks:
        model = fit_kmeans(X, k, seed)
        if labels_out is not None:
            labels_out[k] = model.labels_
        rows.append({"k": k, "sse": float(model.inertia_),
                     "silhouette": _silhouette(X, model.labels_, silhouette_sample, seed)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- validity

def cohesion_separation(X, labels) -> dict[str, float]:
    """SSE within clusters, SSB between them (SSE + SSB = total sum of squares), and silhouette."""
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    centre = X.mean(axis=0)
    sse = ssb = 0.0
    for c in np.unique(labels):
        part = X[labels == c]
        mean = part.mean(axis=0)
        sse += ((part - mean) ** 2).sum()
        ssb += len(part) * ((mean - centre) ** 2).sum()
    return {"sse": float(sse), "ssb": float(ssb), "silhouette": _silhouette(X, labels)}


def _contingency(true, pred) -> pd.DataFrame:
    return pd.crosstab(pd.Series(np.asarray(pred), name="cluster"), pd.Series(np.asarray(true), name="class"))


def purity(true, pred) -> float:
    """Share of points in their cluster's majority class."""
    table = _contingency(true, pred)
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def cluster_entropy(true, pred) -> float:
    """Size-weighted mean of each cluster's class entropy, in bits (0 = every cluster pure)."""
    table = _contingency(true, pred)
    p = table.div(table.sum(axis=1), axis=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cluster = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
    weights = table.sum(axis=1).to_numpy() / table.to_numpy().sum()
    return float((weights * per_cluster).sum())


def cluster_lift(labels, is_mule: pd.Series) -> pd.DataFrame:
    """Per cluster: size, mules, mule share and lift over the overall mule share. Highest lift first."""
    df = pd.DataFrame({"cluster": np.asarray(labels), "mule": np.asarray(is_mule, dtype=bool)})
    base = df.mule.mean()
    out = df.groupby("cluster").mule.agg(size="size", mules="sum").reset_index()
    out["mule_share"] = out.mules / out["size"]
    out["lift"] = out.mule_share / base if base > 0 else 0.0
    return out.sort_values(["lift", "size"], ascending=[False, False]).reset_index(drop=True)


def mule_capture(lift_df: pd.DataFrame, min_lift: float = 5.0) -> float:
    """Share of all mules that sit in clusters with lift >= min_lift."""
    total = lift_df.mules.sum()
    return float(lift_df.mules[lift_df.lift >= min_lift].sum() / total) if total else 0.0


def shuffled_baseline(X, k: int, n_repeats: int = 20, seed: int = 0) -> pd.DataFrame:
    """Permute each column independently (keeps each feature's distribution, destroys the
    joint structure) and recluster: the SSE and silhouette that structure-free data gives."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    rows = []
    for _ in range(n_repeats):
        shuffled = np.column_stack([rng.permutation(col) for col in X.T])
        model = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(shuffled)
        rows.append({"sse": float(model.inertia_), "silhouette": _silhouette(shuffled, model.labels_, seed=seed)})
    return pd.DataFrame(rows)


def cluster_profiles(df: pd.DataFrame, labels, roles: pd.Series) -> pd.DataFrame:
    """Mean of each clustering feature and the role mix, per cluster."""
    cols = [c for c in CLUSTER_FEATURES if c in df]
    feats = df[cols].reset_index(drop=True).assign(cluster=np.asarray(labels))
    prof = feats.groupby("cluster").mean()
    mix = pd.crosstab(np.asarray(labels), np.asarray(roles), normalize="index").add_prefix("role_")
    prof = prof.join(mix).fillna(0.0)
    prof.insert(0, "size", feats.groupby("cluster").size())
    prof.index.name = "cluster"
    return prof


# ---------------------------------------------------------------- figures

def proximity_matrix_figure(X, labels, n_sample: int = 2000, seed: int = 0, path=FIGURES / "cluster_proximity.png",
                            title: str = "Distance matrix, sorted by cluster") -> None:
    """Pairwise distances for a cluster-stratified sample, rows sorted by cluster.
    Dark blocks on the diagonal mean tight, separate clusters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import pairwise_distances

    X, labels = np.asarray(X), np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = min(n_sample, len(X))
    picks = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        take = max(1, round(n * len(idx) / len(X)))  # every cluster shows, even the small mule ones
        picks.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    idx = np.concatenate(picks)
    idx = idx[np.argsort(labels[idx], kind="stable")]
    D = pairwise_distances(X[idx])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(D, cmap="viridis_r", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Euclidean distance (scaled features)")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _sweep_figure(sweep: pd.DataFrame, shuffle_sil: float, chosen_k: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(sweep.k, sweep.sse, "o-", color="#4C72B0")
    a.set(xlabel="K", ylabel="SSE", title="Elbow: SSE by K")
    b.plot(sweep.k, sweep.silhouette, "o-", color="#4C72B0", label="real data")
    b.axhline(shuffle_sil, color="#C44E52", ls="--", label=f"shuffled baseline (K={chosen_k})")
    b.axvline(chosen_k, color="grey", ls=":")
    b.set(xlabel="K", ylabel="silhouette (5,000-account sample)", title="Silhouette by K")
    b.legend()
    for ax in (a, b):
        ax.set_xticks(sweep.k[::2])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _lift_figure(lift: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lift.cluster.astype(str), lift.lift, color="#4C72B0")
    ax.axhline(1, color="grey", lw=1)
    ax.axhline(5, color="#C44E52", ls="--", label="lift 5")
    ax.set(xlabel="cluster (sorted by lift)", ylabel="mule lift over base rate", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- study

def _validity_row(method, k, X, labels, roles, is_mule) -> dict:
    lift = cluster_lift(labels, is_mule)
    return {"method": method, "k": k, **cohesion_separation(X, labels),
            "purity_role": purity(roles, labels), "entropy_role": cluster_entropy(roles, labels),
            "ari_role": adjusted_rand_score(roles, labels), "ari_mule": adjusted_rand_score(is_mule, labels),
            "top_lift": float(lift.lift.iloc[0]), "mule_capture_lift5": mule_capture(lift, 5.0)}


def run_clustering_study(features_path=FEATURES, accounts_path=ACCOUNTS, ks=range(2, 21), k: int | None = None,
                         fig_dir=FIGURES, tab_dir=TABLES, seed: int = 0, n_repeats: int = 20) -> dict:
    """D19 end to end, reported at two K: the best silhouette in the sweep (or `k`), and the
    largest K swept. At K = 2 bisecting K-means is one split, the same as K-means, so the
    finer K is where the two methods and the mule lift are actually compared."""
    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    table = active_accounts(pd.read_parquet(features_path))
    roles = table.account_id.map(pd.read_csv(accounts_path, usecols=["account_id", "role"])
                                 .set_index("account_id").role)
    is_mule = roles.isin(MULE_ROLES)
    X, _ = prepare(table)

    fits: dict = {}
    sweep = kmeans_sweep(X, ks, seed=seed, labels_out=fits)
    sweep["mule_capture_lift5"] = [mule_capture(cluster_lift(fits[kk], is_mule)) for kk in sweep.k]
    k_best = k or int(sweep.loc[sweep.silhouette.idxmax(), "k"])
    report_ks = sorted({k_best, int(sweep.k.max())})

    validity, lifts, shuffles = [], {}, {}
    for kk in report_ks:
        km = fits[kk] if kk in fits else fit_kmeans(X, kk, seed).labels_
        bis = fit_bisecting(X, kk, seed).labels_
        shuffle = shuffled_baseline(X, kk, n_repeats=n_repeats, seed=seed)
        shuffles[kk] = shuffle.assign(k=kk)
        for method, labels in (("kmeans", km), ("bisecting_kmeans", bis)):
            row = _validity_row(method, kk, X, labels, roles, is_mule)
            validity.append({**row, "shuffled_silhouette_mean": shuffle.silhouette.mean(),
                             "shuffled_sse_mean": shuffle.sse.mean()})
            lifts[(method, kk)] = cluster_lift(labels, is_mule)
        _lift_figure(lifts[("kmeans", kk)], fig_dir / f"cluster_lift_kmeans_k{kk}.png",
                     f"K-means (K={kk}): mule lift per cluster")
        proximity_matrix_figure(X, km, seed=seed, path=fig_dir / f"cluster_proximity_k{kk}.png",
                                title=f"Distance matrix, K-means K={kk}, sorted by cluster")
        cluster_profiles(table, km, roles).reset_index().to_csv(
            tab_dir / f"cluster_profiles_k{kk}.csv", index=False, lineterminator="\n", float_format="%.6g")
    validity = pd.DataFrame(validity)

    tables = {"cluster_sweep": sweep, "cluster_validity": validity,
              "cluster_shuffle": pd.concat(shuffles.values(), ignore_index=True)}
    tables |= {f"cluster_lift_{m}_k{kk}": t for (m, kk), t in lifts.items()}
    for name, t in tables.items():
        t.to_csv(tab_dir / f"{name}.csv", index=False, lineterminator="\n", float_format="%.6g")
    _sweep_figure(sweep, shuffles[k_best].silhouette.mean(), k_best, fig_dir / "cluster_sweep.png")
    return {"n_active": len(table), "n_mules": int(is_mule.sum()), "k_best": k_best, "report_ks": report_ks,
            "sweep": sweep, "validity": validity, "lifts": lifts}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Label-free clustering with validity checks against true roles")
    parser.add_argument("--features", default=str(FEATURES))
    parser.add_argument("--accounts", default=str(ACCOUNTS))
    parser.add_argument("--k", type=int, default=None, help="number of clusters (default: best silhouette)")
    args = parser.parse_args(argv)
    out = run_clustering_study(args.features, args.accounts, k=args.k)
    print(f"{out['n_active']:,} active accounts, {out['n_mules']} mules; best-silhouette K = {out['k_best']}")
    print(out["sweep"].round(3).to_string(index=False))
    print(out["validity"].round(3).T.to_string())
    for (method, k), lift in out["lifts"].items():
        print(f"\n{method}, K={k}: mule lift per cluster (top 5)")
        print(lift.head(5).round(3).to_string(index=False))
    from mulehunt.report import write_clustering_report

    print(f"\nwrote {write_clustering_report()}")


if __name__ == "__main__":
    main()
