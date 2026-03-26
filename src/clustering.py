"""KMeans clustering utilities for the SDSS dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import NUMERIC_COLUMNS


DEFAULT_CLUSTERS = 3
DEFAULT_RANDOM_STATE = 42
TARGET_COLUMN = "class"


def prepare_clustering_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split the cleaned dataset into clustering features and optional real labels."""
    feature_columns = [column for column in NUMERIC_COLUMNS if column in dataframe.columns]
    if not feature_columns:
        raise ValueError("No numeric feature columns available for clustering.")

    features = dataframe[feature_columns].copy()
    labels = dataframe[TARGET_COLUMN].copy() if TARGET_COLUMN in dataframe.columns else None
    return features, labels


def build_kmeans_pipeline(
    n_clusters: int = DEFAULT_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Create a scaled KMeans clustering pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clusterer",
                KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state),
            ),
        ]
    )


def summarize_cluster_vs_class(
    cluster_assignments: pd.Series,
    labels: pd.Series | None,
) -> dict[str, dict[str, int]]:
    """Build a simple comparison table between clusters and real classes."""
    if labels is None:
        return {}

    comparison = pd.DataFrame({"cluster": cluster_assignments, "class": labels})
    counts = pd.crosstab(comparison["cluster"], comparison["class"])

    return {
        f"cluster_{int(cluster_id)}": {class_name: int(count) for class_name, count in row.items()}
        for cluster_id, row in counts.iterrows()
    }


def project_clusters(
    features: pd.DataFrame,
    cluster_assignments: pd.Series,
    labels: pd.Series | None = None,
) -> list[dict[str, float | int | str]]:
    """Project clustering features to 2D to support later visualizations."""
    projected = PCA(n_components=2).fit_transform(features)
    projection_data: dict[str, Any] = {
        "component_1": projected[:, 0],
        "component_2": projected[:, 1],
        "cluster": cluster_assignments,
    }
    if labels is not None:
        projection_data["class"] = labels.values

    projection_frame = pd.DataFrame(
        {
            key: value for key, value in projection_data.items()
        }
    )
    return projection_frame.to_dict(orient="records")


def run_kmeans_clustering(
    dataframe: pd.DataFrame,
    n_clusters: int = DEFAULT_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train KMeans and prepare summary data for the clustering stage."""
    features, labels = prepare_clustering_data(dataframe)
    model = build_kmeans_pipeline(n_clusters=n_clusters, random_state=random_state)

    cluster_assignments = pd.Series(model.fit_predict(features), name="cluster")
    scaled_features = model.named_steps["scaler"].transform(features)

    metrics = {
        "silhouette_score": float(silhouette_score(scaled_features, cluster_assignments)),
        "cluster_sizes": {
            f"cluster_{int(cluster_id)}": int(count)
            for cluster_id, count in cluster_assignments.value_counts().sort_index().items()
        },
    }

    return {
        "model_name": "KMeans",
        "n_clusters": n_clusters,
        "features": list(features.columns),
        "metrics": metrics,
        "cluster_vs_class": summarize_cluster_vs_class(cluster_assignments, labels),
        "plot_data": {
            "projection_2d": project_clusters(features, cluster_assignments, labels),
        },
    }
