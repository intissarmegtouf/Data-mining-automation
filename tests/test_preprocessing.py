import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    PreprocessingConfig,
    Preprocessor,
    clean_structure,
    validate_model_ready,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, np.nan, 40.0, 30.0],
            "city": ["a", "b", "a", None, "b"],
            "income": [1.0, 2.0, 3.0, 4.0, 2.0],
        }
    )


def test_clean_structure_drops_duplicates_rows_and_columns(df):
    config = PreprocessingConfig(
        drop_columns=["income"], remove_duplicates=True, missing={"age": "Drop rows"}
    )
    out = clean_structure(df, config)
    assert "income" not in out.columns
    assert out["age"].notna().all()
    # without 'income', row 4 duplicates row 1; then the row with a missing age is dropped
    assert len(out) == 3


def test_clean_structure_drop_column(df):
    out = clean_structure(df, PreprocessingConfig(missing={"city": "Drop column"}))
    assert "city" not in out.columns


def test_imputation_uses_fit_data_only(df):
    config = PreprocessingConfig(missing={"age": "Mean"})
    train = pd.DataFrame({"age": [10.0, 20.0]})
    test = pd.DataFrame({"age": [np.nan]})
    out = Preprocessor(config).fit(train).transform(test)
    assert out.loc[0, "age"] == 15.0  # mean of the TRAIN set


def test_scaling_uses_fit_statistics_only():
    config = PreprocessingConfig(minmax_cols=["x"])
    train = pd.DataFrame({"x": [0.0, 10.0]})
    test = pd.DataFrame({"x": [20.0]})
    out = Preprocessor(config).fit(train).transform(test)
    assert out.loc[0, "x"] == 2.0  # (20 - 0) / (10 - 0): not rescaled to [0, 1]


def test_zscore():
    config = PreprocessingConfig(zscore_cols=["x"])
    out = Preprocessor(config).fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    assert out["x"].mean() == pytest.approx(0.0)
    assert out["x"].std() == pytest.approx(1.0)


def test_constant_column_does_not_produce_nan():
    config = PreprocessingConfig(minmax_cols=["x"], zscore_cols=["y"])
    out = Preprocessor(config).fit_transform(pd.DataFrame({"x": [5.0, 5.0], "y": [2.0, 2.0]}))
    assert not out.isna().any().any()


def test_one_hot_aligns_test_columns_with_train():
    config = PreprocessingConfig(one_hot_encode=True)
    train = pd.DataFrame({"c": ["a", "b", "a"], "n": [1, 2, 3]})
    test = pd.DataFrame({"c": ["a", "z"], "n": [4, 5]})  # unseen category "z"
    prep = Preprocessor(config).fit(train)
    out = prep.transform(test)
    assert list(out.columns) == list(prep.transform(train).columns)
    assert out.loc[1, ["c_a", "c_b"]].sum() == 0


def test_validate_model_ready_flags_strings_and_nans():
    with pytest.raises(ValueError, match="Non-numeric"):
        validate_model_ready(pd.DataFrame({"c": ["a", "b"]}))
    with pytest.raises(ValueError, match="Missing values"):
        validate_model_ready(pd.DataFrame({"x": [1.0, np.nan]}))
    validate_model_ready(pd.DataFrame({"x": [1.0, 2.0]}))
