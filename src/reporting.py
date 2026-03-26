"""Utilities for persisting model metrics and visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_pipeline_outputs(report: dict[str, Any], output_dir: str | Path) -> None:
    """Persist the current pipeline results as files inside outputs/."""
    output_path = Path(output_dir)
    metrics_path = output_path / "metrics"
    plots_path = output_path / "plots"

    metrics_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    compact_report = build_compact_pipeline_report(report)

    _save_json(metrics_path / "pipeline_report.json", compact_report)
    _save_summary_txt(metrics_path / "summary.txt", compact_report)

    if "classification" in report:
        _save_json(
            metrics_path / "classification_metrics.json",
            build_classification_metrics_output(report["classification"]),
        )
        _plot_confusion_matrix(report["classification"], plots_path / "classification_confusion_matrix.png")

    if "regression" in report:
        _save_json(
            metrics_path / "regression_metrics.json",
            build_regression_metrics_output(report["regression"]),
        )
        _save_json(
            metrics_path / "regression_plot_data.json",
            report["regression"]["plot_data"],
        )
        _plot_regression_predictions(report["regression"], plots_path / "regression_actual_vs_predicted.png")

    if "clustering" in report:
        _save_json(
            metrics_path / "clustering_metrics.json",
            build_clustering_metrics_output(report["clustering"]),
        )
        _save_json(
            metrics_path / "clustering_plot_data.json",
            report["clustering"]["plot_data"],
        )
        _plot_cluster_projection(report["clustering"], plots_path / "clustering_projection.png")
        _plot_cluster_vs_class(report["clustering"], plots_path / "clustering_vs_class.png")


def build_compact_pipeline_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep only the summary information required for the main pipeline report."""
    compact_report = {
        "phase": report.get("phase"),
        "dataset_path": report.get("dataset_path"),
        "raw_data": report.get("raw_data"),
        "cleaned_data": report.get("cleaned_data"),
    }

    if "classification" in report:
        compact_report["classification"] = build_classification_metrics_output(report["classification"])

    if "regression" in report:
        compact_report["regression"] = build_regression_metrics_output(report["regression"])

    if "clustering" in report:
        compact_report["clustering"] = build_clustering_metrics_output(report["clustering"])

    return compact_report


def build_classification_metrics_output(classification_report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact classification output containing only evaluation data."""
    return {
        "model_name": classification_report["model_name"],
        "n_neighbors": classification_report["n_neighbors"],
        "features": classification_report["features"],
        "target": classification_report["target"],
        "train_samples": classification_report["train_samples"],
        "test_samples": classification_report["test_samples"],
        "metrics": classification_report["metrics"],
    }


def build_regression_metrics_output(regression_report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact regression output containing only evaluation data."""
    return {
        "model_name": regression_report["model_name"],
        "notes": regression_report.get("notes", []),
        "features": regression_report["features"],
        "target": regression_report["target"],
        "train_samples": regression_report["train_samples"],
        "test_samples": regression_report["test_samples"],
        "metrics": regression_report["metrics"],
    }


def build_clustering_metrics_output(clustering_report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact clustering output containing only evaluation data."""
    return {
        "model_name": clustering_report["model_name"],
        "n_clusters": clustering_report["n_clusters"],
        "features": clustering_report["features"],
        "metrics": clustering_report["metrics"],
        "cluster_vs_class": clustering_report["cluster_vs_class"],
    }


def _save_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Serialize a Python dictionary to JSON."""
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_summary_txt(file_path: Path, report: dict[str, Any]) -> None:
    """Create a plain text overview of the current pipeline results."""
    lines = [
        f"Phase: {report.get('phase', 'unknown')}",
        f"Dataset: {report.get('dataset_path', 'unknown')}",
        f"Raw rows: {report.get('raw_data', {}).get('shape', {}).get('rows', 'n/a')}",
        f"Cleaned rows: {report.get('cleaned_data', {}).get('shape', {}).get('rows', 'n/a')}",
    ]

    if "classification" in report:
        accuracy = report["classification"]["metrics"]["accuracy"]
        lines.append(f"Classification accuracy: {accuracy:.4f}")

    if "regression" in report:
        mse = report["regression"]["metrics"]["mse"]
        r2 = report["regression"]["metrics"]["r2"]
        lines.append(f"Regression MSE: {mse:.4f}")
        lines.append(f"Regression R2: {r2:.4f}")

    if "clustering" in report:
        silhouette = report["clustering"]["metrics"]["silhouette_score"]
        lines.append(f"Clustering silhouette score: {silhouette:.4f}")

    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_confusion_matrix(classification_report: dict[str, Any], file_path: Path) -> None:
    """Save a confusion matrix heatmap."""
    labels = classification_report["metrics"]["labels"]
    matrix = np.array(classification_report["metrics"]["confusion_matrix"])

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title("KNN Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(col_index, row_index, str(matrix[row_index, col_index]), ha="center", va="center")

    figure.tight_layout()
    figure.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_regression_predictions(regression_report: dict[str, Any], file_path: Path) -> None:
    """Save an actual-vs-predicted scatter plot for regression."""
    plot_data = regression_report["plot_data"]
    actual = np.array(plot_data["actual_values"])
    predicted = np.array(plot_data["predicted_values"])

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(actual, predicted, alpha=0.7, edgecolors="none")

    min_value = float(min(actual.min(), predicted.min()))
    max_value = float(max(actual.max(), predicted.max()))
    axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")

    axis.set_title("Linear Regression: Actual vs Predicted")
    axis.set_xlabel("Actual redshift")
    axis.set_ylabel("Predicted redshift")

    figure.tight_layout()
    figure.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_cluster_projection(clustering_report: dict[str, Any], file_path: Path) -> None:
    """Save a 2D projection of clustering assignments."""
    projection = pd.DataFrame(clustering_report["plot_data"]["projection_2d"])

    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        projection["component_1"],
        projection["component_2"],
        c=projection["cluster"],
        cmap="tab10",
        alpha=0.75,
        edgecolors="none",
    )
    legend = axis.legend(*scatter.legend_elements(), title="Cluster")
    axis.add_artist(legend)

    axis.set_title("KMeans Cluster Projection")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")

    figure.tight_layout()
    figure.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_cluster_vs_class(clustering_report: dict[str, Any], file_path: Path) -> None:
    """Save a heatmap that compares discovered clusters against real classes."""
    comparison = clustering_report.get("cluster_vs_class", {})
    if not comparison:
        return

    frame = pd.DataFrame.from_dict(comparison, orient="index").fillna(0)
    matrix = frame.to_numpy()

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Greens")
    figure.colorbar(image, ax=axis)

    axis.set_title("Clusters vs Real Classes")
    axis.set_xlabel("Class")
    axis.set_ylabel("Cluster")
    axis.set_xticks(range(len(frame.columns)))
    axis.set_yticks(range(len(frame.index)))
    axis.set_xticklabels(frame.columns, rotation=45, ha="right")
    axis.set_yticklabels(frame.index)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(col_index, row_index, int(matrix[row_index, col_index]), ha="center", va="center")

    figure.tight_layout()
    figure.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
