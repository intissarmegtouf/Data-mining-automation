"""Dataset loading for CSV and ARFF files."""
from __future__ import annotations

import io

import pandas as pd
from scipy.io import arff

SUPPORTED_EXTENSIONS = ("csv", "arff")


def load_dataset(content: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded file (raw bytes) into a DataFrame."""
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if name.endswith(".arff"):
        return _load_arff(content)
    raise ValueError(f"Unsupported file type: {filename!r} (use .csv or .arff)")


def _load_arff(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="replace")
    records, _ = arff.loadarff(io.StringIO(text))
    df = pd.DataFrame(records)

    # scipy returns nominal/string attributes as bytes: decode them,
    # and turn the ARFF missing-value marker "?" into NaN.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
            df[col] = df[col].replace("?", pd.NA)
    return df
