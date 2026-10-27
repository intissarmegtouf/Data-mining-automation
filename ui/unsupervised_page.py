"""Streamlit page for unsupervised learning (clustering)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import plots, unsupervised
from src.preprocessing import (
    PreprocessingConfig,
    Preprocessor,
    clean_structure,
    validate_model_ready,
)

RESULTS_KEY = "cluster_results"

KMEANS, KMEDOIDS, AGNES, DIANA, DBSCAN = "KMeans", "KMedoids", "AGNES", "DIANA", "DBSCAN"
DESCRIPTIONS = {
    KMEANS: "Partitioning around mean centroids.",
    KMEDOIDS: "Partitioning around real data points (more robust to outliers).",
    AGNES: "Agglomerative hierarchical clustering (bottom-up), with dendrogram.",
    DIANA: "Divisive hierarchical clustering (top-down).",
    DBSCAN: "Density-based clustering; points in low-density areas are labelled as noise.",
}


def render_unsupervised(df: pd.DataFrame, config: PreprocessingConfig) -> None:
    st.header("Unsupervised learning (clustering)")

    # No train/test split here, so fitting the preprocessing on all rows is not leakage.
    data = clean_structure(df, config)
    processed = Preprocessor(config).fit_transform(data)
    try:
        validate_model_ready(processed)
    except ValueError as exc:
        st.error(str(exc))
        return

    features = st.multiselect(
        "Features used for clustering",
        list(processed.columns),
        default=list(processed.columns),
        help="Remove label/target columns here if the dataset contains one.",
    )
    if not features:
        st.warning("Select at least one feature.")
        return
    X = processed[features].to_numpy(dtype=float)
    st.caption(f"{X.shape[0]} samples × {X.shape[1]} features")
    if not config.scaling_enabled:
        st.info("No normalization selected: features with large scales will dominate the distances.")

    _render_k_selection(X)
    st.divider()
    _render_algorithm_runner(X, data, features)
    st.divider()
    _render_comparison()


def _render_k_selection(X: np.ndarray) -> None:
    st.subheader("Choosing the number of clusters")
    left, right = st.columns(2)
    k_min = int(left.number_input("k minimum", 2, 20, 2))
    k_max = int(right.number_input("k maximum", 3, 30, 10))
    if k_min >= k_max:
        st.warning("k maximum must be greater than k minimum.")
    elif st.button("Plot elbow and silhouette curves"):
        ks = range(k_min, min(k_max, len(X) - 1) + 1)
        with st.spinner("Running KMeans for each k..."):
            curves = unsupervised.k_selection_curves(X, ks)
        st.pyplot(plots.k_selection_fig(curves))
        st.caption("Look for the 'elbow' of the inertia curve and for a high silhouette score.")


def _render_algorithm_runner(X: np.ndarray, data: pd.DataFrame, features: list[str]) -> None:
    st.subheader("Run a clustering algorithm")
    algo = st.selectbox("Algorithm", list(DESCRIPTIONS))
    st.caption(DESCRIPTIONS[algo])

    k, linkage_method, eps, min_samples = None, "ward", 0.5, 5
    if algo == DBSCAN:
        eps = st.number_input("Epsilon (neighbourhood radius)", 0.01, 50.0, 0.5, step=0.05)
        min_samples = int(st.slider("MinPts", 1, 20, 5))
    else:
        k = int(st.slider("Number of clusters (k)", 2, 10, 3))
    if algo == AGNES:
        linkage_method = st.selectbox("Linkage", unsupervised.LINKAGE_METHODS)

    too_big = algo in (KMEDOIDS, DIANA) and len(X) > unsupervised.PAIRWISE_LIMIT
    if too_big:
        st.warning(
            f"{algo} needs an n×n distance matrix: limited to {unsupervised.PAIRWISE_LIMIT} rows "
            f"(you have {len(X)}). Use another algorithm or a sample of the data."
        )

    if not st.button(f"Run {algo}", type="primary", disabled=too_big):
        return

    linkage_matrix = None
    with st.spinner(f"Running {algo}..."):
        if algo == KMEANS:
            labels = unsupervised.run_kmeans(X, k)
        elif algo == KMEDOIDS:
            labels = unsupervised.run_kmedoids(X, k)
        elif algo == AGNES:
            labels, linkage_matrix = unsupervised.run_agnes(X, k, linkage_method)
        elif algo == DIANA:
            labels = unsupervised.run_diana(X, k)
        else:
            labels = unsupervised.run_dbscan(X, eps, min_samples)

    metrics = unsupervised.cluster_metrics(X, labels)
    if algo == DBSCAN:
        run_name = f"DBSCAN (eps={eps:g}, minPts={min_samples})"
    elif algo == AGNES:
        run_name = f"AGNES ({linkage_method}, k={k})"
    else:
        run_name = f"{algo} (k={k})"
    st.session_state.setdefault(RESULTS_KEY, {})[run_name] = {"metrics": metrics}

    _show_run(run_name, X, labels, metrics, linkage_matrix, data, features)


def _show_run(run_name, X, labels, metrics, linkage_matrix, data, features) -> None:
    st.success(f"{run_name}: {metrics['n_clusters']} cluster(s) found")
    if metrics["n_noise"]:
        st.warning(f"{metrics['n_noise']} points labelled as noise (-1)")
    if metrics["n_clusters"] == 0:
        st.error("No cluster found: try a larger epsilon or a smaller MinPts.")
        return

    cols = st.columns(3)
    cols[0].metric("Intra-cluster inertia", f"{metrics['intra']:.2f}")
    cols[1].metric("Inter-cluster inertia", f"{metrics['inter']:.2f}")
    cols[2].metric(
        "Silhouette", "n/a" if metrics["silhouette"] is None else f"{metrics['silhouette']:.3f}"
    )

    left, right = st.columns(2)
    left.pyplot(plots.clusters_fig(X, labels))
    if linkage_matrix is not None:
        right.pyplot(plots.dendrogram_fig(linkage_matrix))
    else:
        right.write("**Cluster sizes**")
        right.dataframe(
            pd.Series(labels).value_counts().sort_index().rename("Instances").rename_axis("Cluster")
        )

    labelled = data.copy()
    labelled["cluster"] = labels
    with st.expander("Instances and their cluster"):
        st.dataframe(labelled)
    st.download_button(
        "Download data with cluster labels (CSV)",
        labelled.to_csv(index=False).encode("utf-8"),
        file_name="clustered_data.csv",
        mime="text/csv",
    )


def _render_comparison() -> None:
    st.subheader("Comparison of the runs")
    results = st.session_state.get(RESULTS_KEY, {})
    if not results:
        st.caption("Run at least one algorithm to compare results here.")
        return
    table = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
    st.dataframe(table[["n_clusters", "n_noise", "intra", "inter", "silhouette"]])
    st.pyplot(plots.comparison_fig(results))
    if st.button("Clear results"):
        st.session_state[RESULTS_KEY] = {}
        st.rerun()
