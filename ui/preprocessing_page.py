"""Streamlit widgets for the preprocessing step. Returns a PreprocessingConfig."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from src.preprocessing import (
    MISSING_OPTIONS,
    Preprocessor,
    PreprocessingConfig,
    clean_structure,
)


def render_preprocessing(df: pd.DataFrame) -> PreprocessingConfig:
    config = PreprocessingConfig()

    st.subheader(" Dataset preview")
    st.dataframe(df.head())
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
    with st.expander("Statistical summary"):
        numeric_part = df.select_dtypes(include="number")
        other_part = df.select_dtypes(exclude="number")
        if not numeric_part.empty:
            st.dataframe(numeric_part.describe())
        if not other_part.empty:  # separate table: mixed types in one table break Arrow
            st.dataframe(other_part.describe().astype(str))

    # --- Columns to exclude (IDs, free text, ...)
    st.subheader("Columns to exclude")
    config.drop_columns = st.multiselect(
        "Columns that should not be used (e.g. identifiers)", list(df.columns)
    )
    work = df.drop(columns=config.drop_columns)

    # --- Duplicates
    st.subheader("Duplicate handling")
    n_duplicates = int(work.duplicated().sum())
    if n_duplicates:
        st.warning(f"{n_duplicates} duplicated rows detected")
        config.remove_duplicates = st.checkbox("Remove duplicates")
    else:
        st.success("No duplicates found")
    work = work.drop_duplicates() if config.remove_duplicates else work

    # --- Missing values
    st.subheader("Missing values")
    missing_cols = list(work.columns[work.isnull().any()])
    if not missing_cols:
        st.success("No missing values")
    for col in missing_cols:
        options = MISSING_OPTIONS if is_numeric_dtype(work[col]) else [
            o for o in MISSING_OPTIONS if o != "Mean"
        ]
        config.missing[col] = st.selectbox(
            f"`{col}` — {int(work[col].isnull().sum())} missing", options, key=f"missing_{col}"
        )

    # --- Encoding
    st.subheader("Encoding categorical columns")
    categorical = list(work.select_dtypes(exclude="number").columns)
    if categorical:
        config.one_hot_encode = st.checkbox(
            "One-hot encode categorical columns (needed by the models)",
            value=True,
            help="One-hot avoids creating a fake order between categories, "
            "unlike label encoding. The target column is handled separately.",
        )
    else:
        st.info("No categorical columns")

    # --- Normalization
    st.subheader("Normalization")
    numeric = list(work.select_dtypes(include="number").columns)
    config.minmax_cols = st.multiselect("Min-Max normalization", numeric)
    config.zscore_cols = st.multiselect(
        "Z-Score normalization", [c for c in numeric if c not in config.minmax_cols]
    )

    # --- Preview
    with st.expander("Preview of the preprocessed data"):
        st.caption(
            "Statistics are computed on the whole dataset for this preview only. In supervised "
            "mode they are recomputed on the training split to avoid data leakage."
        )
        try:
            st.dataframe(
                Preprocessor(config).fit_transform(clean_structure(df, config)).head()
            )
        except Exception as exc:  # preview must never crash the page
            st.warning(f"Preview unavailable: {exc}")

    return config
