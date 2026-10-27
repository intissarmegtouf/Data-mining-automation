"""Preprocessing: duplicates, missing values, encoding and normalization.

Design note (avoiding data leakage)
-----------------------------------
Steps are split in two groups:

* ``clean_structure`` only removes rows/columns. It learns nothing from the
  data, so it can safely run *before* the train/test split.
* ``Preprocessor`` learns statistics (imputation values, one-hot categories,
  min/max, mean/std). For supervised learning it is fitted on the training
  set only and then applied to the test set, so no information from the test
  set leaks into training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_numeric_dtype

MISSING_OPTIONS = ["Do nothing", "Drop rows", "Drop column", "Mean", "Mode"]


@dataclass
class PreprocessingConfig:
    """User choices for the preprocessing pipeline."""

    drop_columns: list[str] = field(default_factory=list)
    remove_duplicates: bool = False
    missing: dict[str, str] = field(default_factory=dict)  # column -> option
    one_hot_encode: bool = False
    minmax_cols: list[str] = field(default_factory=list)
    zscore_cols: list[str] = field(default_factory=list)

    @property
    def scaling_enabled(self) -> bool:
        return bool(self.minmax_cols or self.zscore_cols)


def clean_structure(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Drop columns, duplicates and rows/columns with missing values (no learning)."""
    out = df.drop(columns=[c for c in config.drop_columns if c in df.columns])
    if config.remove_duplicates:
        out = out.drop_duplicates()

    for col, option in config.missing.items():
        if col not in out.columns:
            continue
        if option == "Drop rows":
            out = out.dropna(subset=[col])
        elif option == "Drop column":
            out = out.drop(columns=[col])
    return out.reset_index(drop=True)


class Preprocessor:
    """Imputation -> one-hot encoding -> scaling, fitted on one dataset and reusable on another."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        # 1. Imputation values
        self.fill_values_: dict = {}
        for col, option in self.config.missing.items():
            if col not in df.columns:
                continue
            if option == "Mean" and is_numeric_dtype(df[col]):
                self.fill_values_[col] = df[col].mean()
            elif option == "Mode":
                mode = df[col].mode()
                if not mode.empty:
                    self.fill_values_[col] = mode.iloc[0]
        filled = df.fillna(self.fill_values_)

        # 2. One-hot encoding (categories learned here)
        self.categorical_cols_: list[str] = (
            list(filled.select_dtypes(exclude="number").columns)
            if self.config.one_hot_encode
            else []
        )
        encoded = self._encode(filled)
        self.columns_: list[str] = list(encoded.columns)

        # 3. Scaling statistics (computed on the imputed/encoded fit data)
        self.minmax_ = {
            c: (encoded[c].min(), encoded[c].max())
            for c in self.config.minmax_cols
            if c in encoded.columns
        }
        self.zscore_ = {
            c: (encoded[c].mean(), encoded[c].std())
            for c in self.config.zscore_cols
            if c in encoded.columns and c not in self.minmax_
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded = self._encode(df.fillna(self.fill_values_))
        encoded = encoded.reindex(columns=self.columns_, fill_value=0.0)

        for col, (lo, hi) in self.minmax_.items():
            encoded[col] = (encoded[col] - lo) / ((hi - lo) or 1.0)
        for col, (mean, std) in self.zscore_.items():
            encoded[col] = (encoded[col] - mean) / (std if std and std == std else 1.0)
        return encoded

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_cols_:
            return df.copy()
        return pd.get_dummies(df, columns=self.categorical_cols_, dtype=float)


def validate_model_ready(df: pd.DataFrame) -> None:
    """Raise ValueError if the data cannot be fed to a scikit-learn model yet."""
    non_numeric = list(df.select_dtypes(exclude="number").columns)
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns remain: {non_numeric}. "
            "Enable one-hot encoding or exclude these columns."
        )
    with_nan = list(df.columns[df.isnull().any()])
    if with_nan:
        raise ValueError(
            f"Missing values remain in: {with_nan}. "
            "Choose a handling method (drop rows/column, mean, mode) for these columns."
        )
