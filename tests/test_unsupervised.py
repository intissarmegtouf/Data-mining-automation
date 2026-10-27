import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from src import unsupervised as uns


@pytest.fixture(scope="module")
def blobs():
    return make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=7)


@pytest.mark.parametrize(
    "runner",
    [
        lambda X: uns.run_kmeans(X, 3),
        lambda X: uns.run_kmedoids(X, 3),
        lambda X: uns.run_agnes(X, 3)[0],
        lambda X: uns.run_agnes(X, 3, "complete")[0],
        lambda X: uns.run_diana(X, 3),
    ],
)
def test_algorithms_recover_well_separated_blobs(blobs, runner):
    X, y = blobs
    labels = runner(X)
    assert len(np.unique(labels)) == 3
    assert adjusted_rand_score(y, labels) > 0.9


def test_dbscan_finds_clusters_and_noise():
    X, _ = make_blobs(n_samples=100, centers=2, cluster_std=0.4, random_state=1)
    X = np.vstack([X, [[50.0, 50.0]]])  # far outlier
    labels = uns.run_dbscan(X, eps=1.0, min_samples=5)
    assert labels[-1] == -1
    assert len(set(labels) - {-1}) == 2


def test_inertia_decomposition(blobs):
    """For a hard partition without noise: intra + inter = total sum of squares."""
    X, _ = blobs
    labels = uns.run_kmeans(X, 3)
    m = uns.cluster_metrics(X, labels)
    total = ((X - X.mean(axis=0)) ** 2).sum()
    assert m["intra"] + m["inter"] == pytest.approx(total)


def test_metrics_ignore_noise_and_handle_degenerate_cases(blobs):
    X, _ = blobs
    all_noise = uns.cluster_metrics(X, np.full(len(X), -1))
    assert all_noise["n_clusters"] == 0 and all_noise["silhouette"] is None
    one_cluster = uns.cluster_metrics(X, np.zeros(len(X), dtype=int))
    assert one_cluster["silhouette"] is None
    labels = uns.run_dbscan(X, eps=0.3, min_samples=5)
    assert uns.cluster_metrics(X, labels)["n_noise"] == int((labels == -1).sum())


def test_diana_stops_on_identical_points():
    X = np.ones((10, 2))
    assert len(np.unique(uns.run_diana(X, 4))) == 1


def test_k_selection_curves(blobs):
    X, _ = blobs
    curves = uns.k_selection_curves(X, range(2, 6))
    assert curves["inertia"] == sorted(curves["inertia"], reverse=True)
    assert np.argmax(curves["silhouette"]) == 1  # k = 3 is the best silhouette
