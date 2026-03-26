"""Utilities for loading, inspecting, and cleaning the SDSS dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
    "class",
    "snr_r",
    "extinction_r",
]

NUMERIC_COLUMNS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
    "snr_r",
    "extinction_r",
]

CATEGORICAL_COLUMNS = ["class"]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the dataset from disk and validate the expected schema."""
    dataset_path = Path(file_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)
    _validate_columns(dataframe)
    return dataframe


def inspect_data(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a compact inspection summary for EDA and logging."""
    summary: dict[str, Any] = {
        "shape": {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
        },
        "columns": list(dataframe.columns),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "null_values": {column: int(value) for column, value in dataframe.isnull().sum().items()},
        "duplicated_rows": int(dataframe.duplicated().sum()),
    }

    if "class" in dataframe.columns:
        summary["class_distribution"] = {
            label: int(count)
            for label, count in dataframe["class"].value_counts(dropna=False).items()
        }

    return summary


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the initial cleaning required for the next project phases."""
    cleaned = dataframe.copy()

    cleaned.columns = [column.strip() for column in cleaned.columns]
    _validate_columns(cleaned)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    cleaned = cleaned.drop_duplicates().dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    return cleaned


def _validate_columns(dataframe: pd.DataFrame) -> None:
    """Ensure the dataset contains the columns required by the project."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

