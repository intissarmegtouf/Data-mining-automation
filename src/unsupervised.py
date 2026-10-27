"""Unsupervised learning: clustering algorithms and quality metrics.

All functions take a numeric ``numpy`` array ``X`` of shape (n_samples, n_features)
and return cluster labels (noise points of DBSCAN are labelled -1).
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score

from src import RANDOM_STATE

# KMedoids and DIANA need the full n x n distance matrix.
PAIRWISE_LIMIT = 3000
LINKAGE_METHODS = ["ward", "complete", "average", "single"]


# ---------------------------------------------------------------- partitioning
def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)


def run_kmedoids(X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """K-Medoids with alternating assignment/update steps (k-medoids++ initialisation).

    This is the simple "Voronoi iteration" variant, not the full PAM swap search.
    """
    D = pairwise_distances(X)
    n = len(X)
    rng = np.random.default_rng(RANDOM_STATE)

    medoids = [int(rng.integers(n))]
    while len(medoids) < k:
        d2 = D[:, medoids].min(axis=1) ** 2
        if d2.sum() == 0:  # all remaining points coincide with a medoid
            candidates = np.setdiff1d(np.arange(n), medoids)
            medoids.append(int(rng.choice(candidates)))
        else:
            medoids.append(int(rng.choice(n, p=d2 / d2.sum())))
    medoids = np.array(medoids)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = medoids.copy()
        for c in range(k):
            members = np.flatnonzero(labels == c)
            if len(members):
                new_medoids[c] = members[np.argmin(D[np.ix_(members, members)].sum(axis=1))]
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    return np.argmin(D[:, medoids], axis=1)


# ---------------------------------------------------------------- hierarchical
def run_agnes(X: np.ndarray, k: int, method: str = "ward"):
    """Agglomerative (bottom-up) clustering. Returns (labels, linkage matrix)."""
    Z = linkage(X, method=method)
    return fcluster(Z, k, criterion="maxclust") - 1, Z


def run_diana(X: np.ndarray, k: int) -> np.ndarray:
    """DIANA: divisive (top-down) clustering (Kaufman & Rousseeuw, 1990).

    Repeatedly takes the cluster with the largest diameter and splits it with the
    "splinter group" procedure until k clusters are obtained.
    """
    D = pairwise_distances(X)
    clusters = [np.arange(len(X))]

    while len(clusters) < k:
        diameters = [D[np.ix_(c, c)].max() if len(c) > 1 else -1.0 for c in clusters]
        target = int(np.argmax(diameters))
        if diameters[target] <= 0:  # only identical points left: cannot split further
            break
        first, second = _split_cluster(D, clusters.pop(target))
        clusters += [first, second]

    labels = np.empty(len(X), dtype=int)
    for label, members in enumerate(clusters):
        labels[members] = label
    return labels


def _split_cluster(D: np.ndarray, members: np.ndarray):
    """Split one cluster in two with the DIANA splinter-group procedure."""
    sub = D[np.ix_(members, members)]
    n = len(members)
    in_splinter = np.zeros(n, dtype=bool)

    seed = int(np.argmax(sub.sum(axis=1)))  # most dissimilar object starts the splinter group
    in_splinter[seed] = True
    sum_rest = sub.sum(axis=1) - sub[:, seed]  # distance to objects still in the main group
    sum_splinter = sub[:, seed].copy()  # distance to the splinter group

    while (~in_splinter).sum() > 1:
        rest = np.flatnonzero(~in_splinter)
        gain = sum_rest[rest] / (len(rest) - 1) - sum_splinter[rest] / in_splinter.sum()
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        moved = rest[best]
        in_splinter[moved] = True
        sum_rest -= sub[:, moved]
        sum_splinter += sub[:, moved]
    return members[in_splinter], members[~in_splinter]


# ---------------------------------------------------------------- density-based
def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


# ---------------------------------------------------------------- evaluation
def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Same definition of every metric for every algorithm, so results are comparable.

    * intra: sum of squared distances of points to their cluster centroid
    * inter: sum over clusters of size * squared distance of the centroid to the global mean
    * silhouette: computed on non-noise points only (needs >= 2 clusters)

    Noise points (label -1, DBSCAN) are excluded from all three.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    mask = labels != -1
    Xc, lc = X[mask], labels[mask]
    clusters = np.unique(lc)

    result = {
        "n_clusters": int(len(clusters)),
        "n_noise": int((~mask).sum()),
        "intra": None,
        "inter": None,
        "silhouette": None,
    }
    if len(clusters) == 0:
        return result

    global_mean = Xc.mean(axis=0)
    intra = inter = 0.0
    for c in clusters:
        points = Xc[lc == c]
        centroid = points.mean(axis=0)
        intra += float(((points - centroid) ** 2).sum())
        inter += len(points) * float(((centroid - global_mean) ** 2).sum())
    result["intra"], result["inter"] = intra, inter

    if 2 <= len(clusters) < len(Xc):
        sample = 5000 if len(Xc) > 5000 else None
        result["silhouette"] = float(
            silhouette_score(Xc, lc, sample_size=sample, random_state=RANDOM_STATE)
        )
    return result


def k_selection_curves(X: np.ndarray, ks) -> dict:
    """Inertia (elbow method) and silhouette score of KMeans for each k."""
    inertias, silhouettes = [], []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        inertias.append(float(model.inertia_))
        silhouettes.append(cluster_metrics(X, model.labels_)["silhouette"])
    return {"k": list(ks), "inertia": inertias, "silhouette": silhouettes}
