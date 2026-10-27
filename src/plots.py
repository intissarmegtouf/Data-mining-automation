"""Matplotlib figures (returned, never shown: the UI decides how to display them).

Figures are created with ``Figure`` instead of ``pyplot`` so they are not kept in
pyplot's global state (no memory build-up when the app re-runs).
"""
from __future__ import annotations

import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree

_GRID = {"alpha": 0.3}


def _new_figure(figsize) -> Figure:
    """A figure with an Agg canvas (some helpers, e.g. plot_tree, need a renderer)."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


# ---------------------------------------------------------------- classification
def confusion_matrix_fig(cm: np.ndarray, class_names) -> Figure:
    fig = _new_figure((6, 5))
    ax = fig.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=class_names, yticklabels=class_names,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def roc_fig(y_test, y_proba: np.ndarray, classes, class_names) -> Figure:
    """ROC curve: binary, or one-vs-rest curves for multi-class."""
    fig = _new_figure((6, 5))
    ax = fig.subplots()

    if y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=classes[1])
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        y_bin = label_binarize(y_test, classes=classes)
        for i, name in enumerate(class_names):
            if y_bin[:, i].min() == y_bin[:, i].max():  # class absent/only class in test set
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve" + (" (one-vs-rest)" if y_proba.shape[1] > 2 else ""))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(**_GRID)
    fig.tight_layout()
    return fig


def tree_fig(model, feature_names, class_names) -> Figure:
    fig = _new_figure((20, 10))
    ax = fig.subplots()
    plot_tree(
        model, feature_names=list(feature_names), class_names=list(class_names),
        filled=True, rounded=True, fontsize=9, ax=ax,
    )
    return fig


# ---------------------------------------------------------------- regression
def predictions_vs_actual_fig(y_test, y_pred) -> Figure:
    fig = _new_figure((6, 5))
    ax = fig.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="black", linewidth=0.4)
    low, high = min(np.min(y_test), np.min(y_pred)), max(np.max(y_test), np.max(y_pred))
    ax.plot([low, high], [low, high], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Predictions vs actual")
    ax.legend()
    ax.grid(**_GRID)
    fig.tight_layout()
    return fig


def residuals_fig(y_test, y_pred) -> Figure:
    fig = _new_figure((6, 5))
    ax = fig.subplots()
    ax.scatter(y_pred, np.asarray(y_test) - y_pred, alpha=0.6, edgecolors="black", linewidth=0.4)
    ax.axhline(0, color="green", linestyle="--", lw=2)
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals")
    ax.grid(**_GRID)
    fig.tight_layout()
    return fig


def loss_curve_fig(loss_curve) -> Figure:
    fig = _new_figure((6, 4))
    ax = fig.subplots()
    ax.plot(loss_curve, lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss")
    ax.set_title("Loss curve")
    ax.grid(**_GRID)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- clustering
def k_selection_fig(curves: dict) -> Figure:
    fig = _new_figure((11, 4))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(curves["k"], curves["inertia"], "-o")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Intra-cluster inertia")
    ax1.set_title("Elbow method")

    valid = [(k, s) for k, s in zip(curves["k"], curves["silhouette"]) if s is not None]
    if valid:
        ax2.plot(*zip(*valid), "-o", color="tab:orange")
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette score")
    for ax in (ax1, ax2):
        ax.grid(**_GRID)
    fig.tight_layout()
    return fig


def dendrogram_fig(linkage_matrix) -> Figure:
    fig = _new_figure((10, 4))
    ax = fig.subplots()
    dendrogram(linkage_matrix, ax=ax, no_labels=len(linkage_matrix) > 60)
    ax.set_title("Dendrogram")
    fig.tight_layout()
    return fig


def clusters_fig(X: np.ndarray, labels: np.ndarray) -> Figure:
    """2-D projection (PCA) of the data, coloured by cluster."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] >= 2:
        pca = PCA(n_components=2).fit(X)
        coords = pca.transform(X)
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)"
    else:
        coords = np.column_stack([X[:, 0], np.zeros(len(X))])
        xlabel, ylabel = "Feature", ""

    fig = _new_figure((7, 5))
    ax = fig.subplots()
    palette = sns.color_palette("tab10")
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        is_noise = label == -1
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=25, alpha=0.8,
            color="lightgray" if is_noise else palette[i % len(palette)],
            label="Noise" if is_noise else f"Cluster {label}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Clusters (PCA projection)")
    ax.legend(fontsize=8)
    ax.grid(**_GRID)
    fig.tight_layout()
    return fig


def comparison_fig(results: dict) -> Figure:
    """Compare intra/inter inertia and silhouette across the clustering runs."""
    names = list(results)
    intra = [results[n]["metrics"]["intra"] or 0 for n in names]
    inter = [results[n]["metrics"]["inter"] or 0 for n in names]
    silhouette = [results[n]["metrics"]["silhouette"] or 0 for n in names]
    x = np.arange(len(names))

    fig = _new_figure((12, 4.5))
    ax1, ax2 = fig.subplots(1, 2)
    width = 0.38
    ax1.bar(x - width / 2, intra, width, label="Intra-cluster", color="skyblue")
    ax1.bar(x + width / 2, inter, width, label="Inter-cluster", color="salmon")
    ax1.set_title("Intra vs inter-cluster inertia")
    ax1.legend()

    ax2.bar(x, silhouette, color="seagreen")
    ax2.set_title("Silhouette score")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", **_GRID)
    fig.tight_layout()
    return fig
