# Chapter 3. Clustering

Chapter 2 needed labels. This chapter asks whether mule accounts group together *without* labels: if they do, an investigator could start from a suspicious cluster even with no confirmed cases. The full tables and figures are in [`reports/a_06_clustering.md`](../a_06_clustering.md), which `python -m mulehunt.clustering` regenerates.

## 3.1 Method

**Accounts.** We cluster the 48,515 *active* accounts, meaning those with money both coming in and going out in their window. The others have no flow to describe. Of these accounts, 455 are mules (tier 1–3), a base rate of 0.94%.

**Features.** We use 12 features: the flow and network families plus night share and burstiness. Pass-through, dwell and fan-in/fan-out are the behaviours the handoff names. Heavy-tailed columns (amounts, degrees, fan counts, dwell) get `log1p` first. Then every column is standardised, so no single feature dominates the Euclidean distance. The code refuses to run if `label`, `role` or `is_fraud` is among the inputs.

**Algorithms.**
- **K-means** for K = 2 to 20, each fit keeping the best of 10 k-means++ starts.
- **Bisecting K-means** (scikit-learn `BisectingKMeans`), which splits clusters one at a time.

**Evaluation.**
- **Unsupervised:** SSE, SSB and silhouette, with silhouette computed on a 5,000-account sample.
- **Supervised, against the true roles, which the algorithms never see:** purity, entropy, adjusted Rand index, and per-cluster **mule lift** (a cluster's mule share ÷ the base rate).
- **Significance:** a shuffled baseline.

**K-means as an optimisation problem.** K-means minimises SSE, the sum of squared distances from each point to its cluster's centroid. Lloyd's algorithm alternates two steps, each of which can only lower SSE:
1. Assign each point to its nearest centroid.
2. Move each centroid to the mean of its points.

It always converges, but to a *local* minimum that depends on the starting centroids. That is why each fit keeps the best of 10 starts.

## 3.2 Choosing K

![SSE and silhouette by K](../a_figures/cluster_sweep.png)

SSE falls fastest from K = 2 to 3 and then flattens gradually, with no single sharp elbow. Silhouette is highest at **K = 2** (0.383), so that is the K the data prefers. But a two-way split can only separate "busy, fast-forwarding accounts" from everyone else. To see finer groups we also report **K = 20**, the top of the sweep. K is never chosen using labels.

## 3.3 Headline: do mules cluster together?

| | K = 2 | K = 20 K-means | K = 20 bisecting |
|---|---|---|---|
| Top cluster | 4,612 accounts, 294 mules | 1,098 accounts, 167 mules | 1,963 accounts, 270 mules |
| Mule share of top cluster | 6.4% | 15.2% | 13.8% |
| Lift over base rate | 6.8× | **16.2×** | 14.7× |
| Share of all mules in clusters with lift ≥ 5 | 65% | **77%** | 59% |

![Mule lift per cluster, K-means K = 20](../a_figures/cluster_lift_kmeans_k20.png)

Without seeing a single label, K-means at K = 20 puts 77% of all mules into four clusters where mules are 5–16 times more common than average. **But no cluster is mostly mules.** The best holds 15%, so a cluster is a strong place to start looking, not a list of mules. The handoff described a mostly-mule cluster as "a hope, not a guarantee", and this is the honest outcome.

**Why the clusters are not pure.** Clustering finds accounts that *behave* alike, and we deliberately made normal accounts that behave like mules (Chapter 1.3). The top cluster forwards most of what it receives (pass-through 0.74 against 0.10 overall), forwards it quickly (dwell 32 h against 174 h) and hears from many senders at once (24-hour fan-in 8.0 against 2.3). Money collectors, shop phones and new accounts with large first receipts were built to share that shape.

The other mule-rich clusters look different. One has pass-through of only 0.06 and a dwell of 190 hours, closer to an ordinary account. Mules are not one shape, which is why one cluster cannot hold them all.

## 3.4 K-means against bisecting K-means

At K = 2 the two are identical, because bisecting K-means starts with exactly one two-way K-means split. At K = 20:
- **K-means fits better:** SSE 206,146 against 223,703, and silhouette 0.127 against 0.097.
- **K-means also gathers more mules** into high-lift clusters (77% against 59%).

Bisecting K-means never revisits a split once it is made, which usually costs it some SSE. In return it is faster and its clusters form a hierarchy.

## 3.5 Proximity matrix

![Distance matrix, K-means K = 20](../a_figures/cluster_proximity_k20.png)

This is the pairwise distance matrix for a 2,000-account sample, stratified by cluster and sorted by cluster. Light blocks along the diagonal are tight clusters. Dark bands are small clusters far from the rest. The blocks are visible but not crisp, which matches the modest silhouette.

## 3.6 Cohesion and separation

| Method | K | SSE (cohesion) | SSB (separation) | Silhouette |
|---|---|---|---|---|
| K-means | 2 | 472,949 | 109,231 | 0.383 |
| K-means | 20 | 206,146 | 376,034 | 0.127 |
| Bisecting K-means | 20 | 223,703 | 358,477 | 0.097 |

SSE + SSB is the fixed total sum of squares, so lowering one raises the other. More clusters always lower SSE. Silhouette, which compares each point's own cluster with the nearest other cluster, is what penalises splitting too far.

## 3.7 Supervised validity against true roles

| Method | K | Purity | Entropy (bits) | ARI (roles) | ARI (mule vs not) |
|---|---|---|---|---|---|
| K-means | 2 | 0.726 | 1.014 | 0.127 | 0.090 |
| K-means | 20 | 0.946 | 0.301 | 0.077 | 0.001 |
| Bisecting K-means | 20 | 0.922 | 0.382 | 0.066 | 0.001 |

These measures are misleading on their own. About 99% of accounts are not mules, so purity is high for almost any clustering. At K = 20 most clusters are simply "mostly normal accounts" or "mostly merchants". ARI stays near zero even when mules are concentrated, because it scores the whole partition, and the whole partition is dominated by normal accounts being split into many groups. That is why lift is the headline number.

## 3.8 Significance: is the structure real?

Each feature column was permuted independently (same distributions, no joint structure) and clustered again, 20 times:

| K | Real silhouette | Shuffled mean (sd) | Real SSE | Shuffled SSE |
|---|---|---|---|---|
| 2 | 0.383 | 0.114 (0.040) | 472,949 | 547,546 |
| 20 | 0.127 | 0.064 (0.003) | 206,146 | 328,783 |

At both values of K, the real silhouette is at least 6.7 standard deviations above the shuffled data and the real SSE is lower. The structure is not chance.

## 3.9 Strengths, weaknesses and cluster types

- **Strengths.** K-means is simple, fast (linear in the number of accounts) and easy to explain, and its centroids read as profiles ("forwards 74% within a day").
- **Weaknesses.**
  - K must be chosen.
  - It is sensitive to scaling and outliers, hence `log1p` and standardisation.
  - It finds only local minima.
  - Every point must belong to some cluster, including the unusual ones.
- **Cluster types.** K-means favours round clusters of similar size and spread. The mules are a small, varied group among 48,060 other accounts, so K-means splits them across several clusters and merges each part with look-alike hard cases. Density-based methods would cope better with small, irregular groups; they are outside this unit.

## 3.10 Limitations

- **Simulated data.** The hard cases that make clusters impure are our own inventions (Chapter 1.6). Real look-alikes would differ.
- **One snapshot per account**, as in Chapter 2.
- **The silhouette is sampled** (5,000 accounts) and the proximity matrix uses 2,000, so both carry some sampling noise.
- **K = 20 is a reporting choice**, the top of the sweep, not an optimum. K = 10–19 capture 62–68% of mules in lift ≥ 5 clusters, so K = 20's 77% is the best of the sweep, not typical of it.
