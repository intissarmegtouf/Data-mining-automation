"""Streamlit page for supervised learning (classification and regression)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from src import plots, supervised
from src.preprocessing import (
    PreprocessingConfig,
    Preprocessor,
    clean_structure,
    validate_model_ready,
)

RESULT_KEY = "supervised_result"


def render_supervised(df: pd.DataFrame, config: PreprocessingConfig) -> None:
    st.header("Supervised learning")

    data = clean_structure(df, config)
    if data.shape[1] < 2:
        st.error("At least two columns are needed (features + target).")
        return

    target = st.selectbox("Target variable", list(data.columns))
    data = data.dropna(subset=[target])  # rows without a label cannot be used
    y_raw, X_raw = data[target], data.drop(columns=[target])

    # --- Task
    detected = supervised.detect_task(y_raw)
    task = st.radio(
        "Task",
        ["classification", "regression"],
        index=["classification", "regression"].index(detected),
        horizontal=True,
        help=f"Auto-detected: {detected}. Change it if the detection is wrong.",
    )
    if task == "regression" and not pd.api.types.is_numeric_dtype(y_raw):
        st.error("Regression needs a numeric target.")
        return
    if task == "classification" and y_raw.nunique() > 20:
        st.warning(f"The target has {y_raw.nunique()} distinct values: is this really a classification?")

    test_size = st.slider("Test size (%)", 10, 50, 20) / 100

    # --- Algorithm and hyper-parameters
    algo = st.selectbox("Algorithm", supervised.ALGORITHMS[task])
    params = _algorithm_params(algo, task, y_raw.nunique())
    if algo in supervised.SCALE_SENSITIVE and not config.scaling_enabled:
        st.info(f"{algo} is sensitive to feature scale: consider normalizing the features.")

    if not st.button("Train and evaluate", type="primary"):
        _show_previous_result()
        return

    # --- Split FIRST, then fit the preprocessing on the training set only
    class_names = None
    y = y_raw
    if task == "classification":
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y_raw), index=y_raw.index)
        class_names = [str(c) for c in encoder.classes_]

    X_train_raw, X_test_raw, y_train, y_test = supervised.split_data(X_raw, y, test_size, task)
    preprocessor = Preprocessor(config).fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    try:
        validate_model_ready(X_train)
        validate_model_ready(X_test)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.spinner("Training..."):
        model = supervised.build_model(algo, task, params)
        model.fit(X_train, y_train)

    st.session_state[RESULT_KEY] = {
        "task": task,
        "algo": algo,
        "model": model,
        "features": list(X_train.columns),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": y_test,
        "class_names": class_names,
        "hidden_layers": params.get("hidden_layers"),
        "metrics": (
            supervised.evaluate_classifier(model, X_test, y_test)
            if task == "classification"
            else supervised.evaluate_regressor(model, X_test, y_test)
        ),
    }
    _show_previous_result()


# ---------------------------------------------------------------- parameters
def _algorithm_params(algo: str, task: str, n_classes: int) -> dict:
    if algo == supervised.KNN:
        return {"k": st.slider("K value", 1, 15, 5)}
    if algo in supervised.TREE_ALGORITHMS:
        return {"max_depth": st.slider("Max tree depth", 1, 10, 4)}
    if algo == supervised.NEURAL_NETWORK:
        return _neural_network_params(task, n_classes)
    return {}


def _neural_network_params(task: str, n_classes: int) -> dict:
    """Widgets shared by the classification and regression neural networks."""
    st.subheader("Neural network configuration")
    left, right = st.columns(2)

    with left:
        st.markdown("**Architecture**")
        n_layers = st.number_input("Hidden layers", 1, 10, 2)
        hidden_layers = [
            int(st.number_input(f"Neurons in layer {i + 1}", 1, 500, 100, key=f"nn_layer_{i}"))
            for i in range(int(n_layers))
        ]
        activation = st.selectbox(
            "Hidden activation",
            ["relu", "tanh", "logistic"],
            format_func={"relu": "ReLU", "tanh": "Tanh", "logistic": "Sigmoid"}.get,
        )
        if task == "regression":
            st.caption("Output layer: 1 neuron, identity activation (set automatically).")
        elif n_classes == 2:
            st.caption("Output layer: 1 neuron, sigmoid activation (set automatically).")
        else:
            st.caption(f"Output layer: {n_classes} neurons, softmax activation (set automatically).")

    with right:
        st.markdown("**Training**")
        solver = st.selectbox("Optimizer", ["adam", "sgd", "lbfgs"])
        learning_rate = st.number_input(
            "Initial learning rate", 0.0001, 1.0, 0.001, format="%.4f"
        )
        max_iter = int(st.number_input("Max iterations", 100, 10000, 200))
        batch_size = st.selectbox("Batch size", ["auto", 32, 64, 128, 256])
        early_stopping = st.checkbox(
            "Early stopping", value=True, help="Ignored by the lbfgs optimizer."
        )
        validation_fraction = (
            st.slider("Validation fraction", 0.1, 0.3, 0.1, step=0.05) if early_stopping else 0.1
        )

    return {
        "hidden_layers": hidden_layers,
        "activation": activation,
        "solver": solver,
        "learning_rate_init": learning_rate,
        "max_iter": max_iter,
        "batch_size": batch_size,
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
    }


# ---------------------------------------------------------------- results
def _show_previous_result() -> None:
    result = st.session_state.get(RESULT_KEY)
    if result is None:
        return
    st.divider()
    st.caption(
        f"Last run: **{result['algo']}** ({result['task']}) — "
        f"{result['n_train']} training rows, {result['n_test']} test rows."
    )
    if result["task"] == "classification":
        _show_classification(result)
    else:
        _show_regression(result)
    if result["algo"] == supervised.NEURAL_NETWORK:
        _show_network_details(result)


def _show_classification(result: dict) -> None:
    m, model, names = result["metrics"], result["model"], result["class_names"]
    st.subheader("Classification results")

    cols = st.columns(5 if m["auc"] is not None else 4)
    cols[0].metric("Accuracy", f"{m['accuracy']:.4f}")
    cols[1].metric("Precision", f"{m['precision']:.4f}")
    cols[2].metric("Recall", f"{m['recall']:.4f}")
    cols[3].metric("F1-score", f"{m['f1']:.4f}")
    if m["auc"] is not None:
        cols[4].metric("AUC", f"{m['auc']:.4f}")
    st.caption("Precision, recall and F1 are weighted averages over the classes.")

    left, right = st.columns(2)
    left.pyplot(plots.confusion_matrix_fig(m["confusion_matrix"], names))
    if m["y_proba"] is not None:
        right.pyplot(
            plots.roc_fig(result["y_test"], m["y_proba"], model.classes_, names)
        )
    with st.expander("Detailed classification report"):
        st.text(m["report"])

    if result["algo"] in supervised.TREE_ALGORITHMS:
        st.subheader("Decision tree")
        st.pyplot(plots.tree_fig(model, result["features"], names))


def _show_regression(result: dict) -> None:
    m = result["metrics"]
    st.subheader("Regression results")
    cols = st.columns(4)
    cols[0].metric("MSE", f"{m['mse']:.4f}")
    cols[1].metric("RMSE", f"{m['rmse']:.4f}")
    cols[2].metric("MAE", f"{m['mae']:.4f}")
    cols[3].metric("R²", f"{m['r2']:.4f}")

    left, right = st.columns(2)
    left.pyplot(plots.predictions_vs_actual_fig(result["y_test"], m["y_pred"]))
    right.pyplot(plots.residuals_fig(result["y_test"], m["y_pred"]))


def _show_network_details(result: dict) -> None:
    model = result["model"]
    st.subheader("Neural network details")
    left, right = st.columns(2)
    with left:
        if hasattr(model, "loss_curve_"):
            st.pyplot(plots.loss_curve_fig(model.loss_curve_))
        st.metric("Iterations", model.n_iter_)
    with right:
        st.write(f"**Input layer:** {len(result['features'])} features")
        for i, neurons in enumerate(result["hidden_layers"], start=1):
            st.write(f"**Hidden layer {i}:** {neurons} neurons ({model.activation})")
        st.write(f"**Output layer:** {model.n_outputs_} neuron(s) ({model.out_activation_})")
