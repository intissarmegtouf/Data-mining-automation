import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression

from src import supervised as sup


def test_detect_task():
    assert sup.detect_task(pd.Series(["a", "b", "a"])) == "classification"
    assert sup.detect_task(pd.Series([0, 1, 0, 1])) == "classification"
    assert sup.detect_task(pd.Series(np.linspace(0, 100, 200))) == "regression"


def test_split_is_stratified_for_classification():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0] * 80 + [1] * 20)
    _, _, y_train, y_test = sup.split_data(X, y, 0.2, "classification")
    assert y_test.mean() == pytest.approx(0.2)
    assert y_train.mean() == pytest.approx(0.2)


def test_split_falls_back_when_class_too_rare():
    X = pd.DataFrame({"a": range(20)})
    y = pd.Series([0] * 19 + [1])  # one sample: cannot be stratified
    sup.split_data(X, y, 0.2, "classification")  # must not raise


@pytest.mark.parametrize(
    "algo,params",
    [
        (sup.KNN, {"k": 5}),
        (sup.NAIVE_BAYES, {}),
        (sup.CART, {"max_depth": 4}),
        (sup.ENTROPY_TREE, {"max_depth": 4}),
    ],
)
def test_classifiers_reach_good_accuracy_on_iris(algo, params):
    data = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = sup.split_data(data.data, data.target, 0.3, "classification")
    model = sup.build_model(algo, "classification", params).fit(X_train, y_train)
    metrics = sup.evaluate_classifier(model, X_test, y_test)
    assert metrics["accuracy"] > 0.85
    assert metrics["auc"] > 0.9
    assert metrics["confusion_matrix"].shape == (3, 3)


NN_PARAMS = {
    "hidden_layers": [10], "activation": "relu", "solver": "adam", "learning_rate_init": 0.01,
    "max_iter": 300, "batch_size": "auto", "early_stopping": False, "validation_fraction": 0.1,
}


def test_neural_network_classifier():
    data = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = sup.split_data(data.data, data.target, 0.3, "classification")
    model = sup.build_model(sup.NEURAL_NETWORK, "classification", NN_PARAMS).fit(X_train, y_train)
    assert sup.evaluate_classifier(model, X_test, y_test)["accuracy"] > 0.8


def test_regressors():
    X, y = make_regression(n_samples=200, n_features=5, noise=1.0, random_state=0)
    X, y = pd.DataFrame(X), pd.Series(y)
    X_train, X_test, y_train, y_test = sup.split_data(X, y, 0.2, "regression")
    for algo, params in [(sup.LINEAR_REGRESSION, {}), (sup.NEURAL_NETWORK, NN_PARAMS)]:
        model = sup.build_model(algo, "regression", params).fit(X_train, y_train)
        metrics = sup.evaluate_regressor(model, X_test, y_test)
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))
    # linear data: linear regression must be almost perfect
    lin = sup.build_model(sup.LINEAR_REGRESSION, "regression", {}).fit(X_train, y_train)
    assert sup.evaluate_regressor(lin, X_test, y_test)["r2"] > 0.99


def test_auc_is_none_when_test_set_has_one_class():
    data = load_iris(as_frame=True)
    model = sup.build_model(sup.CART, "classification", {"max_depth": 3}).fit(data.data, data.target)
    mask = data.target == 0
    assert sup.evaluate_classifier(model, data.data[mask], data.target[mask])["auc"] is None
