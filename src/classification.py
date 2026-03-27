"""KNN classification utilities for the SDSS dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import NUMERIC_COLUMNS


TARGET_COLUMN = "class"
DEFAULT_NEIGHBORS = 5
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42


def prepare_classification_data(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column not found: {target_column}")

    feature_columns = [
        column for column in NUMERIC_COLUMNS if column in dataframe.columns
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns available for classification.")

    features = dataframe[feature_columns].copy()
    target = dataframe[target_column].copy()
    return features, target


def build_knn_pipeline(n_neighbors: int = DEFAULT_NEIGHBORS) -> Pipeline:
    """Create a scaled KNN classifier pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def evaluate_classifier(
    model: Pipeline,
    features_test: pd.DataFrame,
    target_test: pd.Series,
) -> dict[str, Any]:
    predictions = model.predict(features_test)
    labels = sorted(target_test.unique().tolist())

    return {
        "accuracy": float(accuracy_score(target_test, predictions)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            target_test,
            predictions,
            labels=labels,
        ).tolist(),
        "test_samples": int(len(target_test)),
    }


def run_knn_classification(
    dataframe: pd.DataFrame,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train and evaluate a KNN classifier using the cleaned dataset."""
    features, target = prepare_classification_data(dataframe)

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    model = build_knn_pipeline(n_neighbors=n_neighbors)
    model.fit(features_train, target_train)

    metrics = evaluate_classifier(model, features_test, target_test)
    return {
        "model_name": "KNeighborsClassifier",
        "n_neighbors": n_neighbors,
        "features": list(features.columns),
        "target": TARGET_COLUMN,
        "train_samples": int(len(target_train)),
        "test_samples": int(len(target_test)),
        "metrics": metrics,
    }
