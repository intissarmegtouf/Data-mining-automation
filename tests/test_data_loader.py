from pathlib import Path

import numpy as np
import pytest

from src.data_loader import load_dataset

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample"


def test_load_csv():
    df = load_dataset((SAMPLES / "iris.csv").read_bytes(), "iris.csv")
    assert df.shape == (150, 5)


def test_load_arff_decodes_nominal_values():
    df = load_dataset((SAMPLES / "iris.arff").read_bytes(), "iris.arff")
    assert df.shape == (150, 5)
    assert set(df["species"]) == {"setosa", "versicolor", "virginica"}


def test_arff_missing_values_become_nan():
    arff = b"@relation t\n@attribute a numeric\n@attribute b {x,y}\n@data\n1,x\n?,y\n3,?\n"
    df = load_dataset(arff, "t.arff")
    assert np.isnan(df.loc[1, "a"])
    assert df["b"].isna().sum() == 1


def test_unsupported_extension():
    with pytest.raises(ValueError):
        load_dataset(b"abc", "data.xlsx")
