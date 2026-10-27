"""Supervised learning: task detection, data splitting, models and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier

from src import RANDOM_STATE

CART = "Decision Tree (CART, Gini)"
ENTROPY_TREE = "Decision Tree (entropy, C4.5-like)"
KNN = "KNN"
NAIVE_BAYES = "Naive Bayes"
NEURAL_NETWORK = "Neural Network"
LINEAR_REGRESSION = "Linear Regression"

ALGORITHMS = {
    "classification": [KNN, NAIVE_BAYES, CART, ENTROPY_TREE, NEURAL_NETWORK],
    "regression": [LINEAR_REGRESSION, NEURAL_NETWORK],
}
TREE_ALGORITHMS = (CART, ENTROPY_TREE)
SCALE_SENSITIVE = (KNN, NEURAL_NETWORK)


def detect_task(y: pd.Series, max_classes: int = 10) -> str:
    """Guess 'classification' or 'regression' from the target column."""
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= max_classes:
        return "classification"
    return "regression"


def split_data(X, y, test_size: float, task: str):
    """Train/test split, stratified for classification whenever possible."""
    if task == "classification":
        try:
            return train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError:  # a class is too rare to be stratified
            pass
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def build_model(algo: str, task: str, params: dict):
    """Instantiate a scikit-learn estimator from the algorithm name and UI parameters."""
    if algo == KNN:
        return KNeighborsClassifier(n_neighbors=params["k"])
    if algo == NAIVE_BAYES:
        return GaussianNB()
    if algo == CART:
        return DecisionTreeClassifier(
            criterion="gini", max_depth=params["max_depth"], random_state=RANDOM_STATE
        )
    if algo == ENTROPY_TREE:
        # scikit-learn has no real C4.5: this is a CART-style tree using entropy.
        return DecisionTreeClassifier(
            criterion="entropy", max_depth=params["max_depth"], random_state=RANDOM_STATE
        )
    if algo == LINEAR_REGRESSION:
        return LinearRegression()
    if algo == NEURAL_NETWORK:
        mlp = MLPClassifier if task == "classification" else MLPRegressor
        return mlp(
            hidden_layer_sizes=tuple(params["hidden_layers"]),
            activation=params["activation"],
            solver=params["solver"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=params["max_iter"],
            batch_size=params["batch_size"],
            early_stopping=params["early_stopping"],
            validation_fraction=params["validation_fraction"],
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown algorithm: {algo!r}")


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Standard classification metrics (weighted averages for multi-class)."""
    y_pred = model.predict(X_test)
    auc, proba = _auc(model, X_test, y_test)
    return {
        "y_pred": y_pred,
        "y_proba": proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=model.classes_),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }


def evaluate_regressor(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "y_pred": y_pred,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def _auc(model, X_test, y_test):
    """Return (AUC or None, predicted probabilities or None)."""
    if not hasattr(model, "predict_proba"):
        return None, None
    proba = model.predict_proba(X_test)
    if len(np.unique(y_test)) < 2:  # AUC is undefined with a single class in the test set
        return None, proba
    try:
        if proba.shape[1] == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(
                y_test, proba, multi_class="ovr", average="weighted", labels=model.classes_
            )
    except ValueError:  # e.g. a class is absent from the test set
        auc = None
    if auc is not None and not np.isfinite(auc):
        auc = None
    return auc, proba
