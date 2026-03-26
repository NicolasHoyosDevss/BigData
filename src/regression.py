"""Linear regression utilities for the SDSS dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import NUMERIC_COLUMNS


TARGET_COLUMN = "redshift"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def prepare_regression_data(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned dataset into features and target for regression."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column not found: {target_column}")

    feature_columns = [column for column in NUMERIC_COLUMNS if column != target_column]
    features = dataframe[feature_columns].copy()
    target = dataframe[target_column].copy()
    return features, target


def build_regression_pipeline() -> Pipeline:
    """Create a scaled linear regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def evaluate_regressor(
    model: Pipeline,
    features_test: pd.DataFrame,
    target_test: pd.Series,
) -> dict[str, Any]:
    """Compute the metrics required for the regression stage."""
    predictions = model.predict(features_test)
    return {
        "mse": float(mean_squared_error(target_test, predictions)),
        "r2": float(r2_score(target_test, predictions)),
        "test_samples": int(len(target_test)),
    }


def run_linear_regression(
    dataframe: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train and evaluate a linear regression model on the cleaned dataset."""
    features, target = prepare_regression_data(dataframe)

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_regression_pipeline()
    model.fit(features_train, target_train)

    metrics = evaluate_regressor(model, features_test, target_test)
    return {
        "model_name": "LinearRegression",
        "notes": [
            "This model is a baseline linear regressor for redshift prediction.",
            "Negative predictions may appear because linear regression is not bounded to non-negative outputs.",
        ],
        "features": list(features.columns),
        "target": TARGET_COLUMN,
        "train_samples": int(len(target_train)),
        "test_samples": int(len(target_test)),
        "metrics": metrics,
        "plot_data": {
            "actual_values": [float(value) for value in target_test.tolist()],
            "predicted_values": [float(value) for value in model.predict(features_test).tolist()],
        },
    }
