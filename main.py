"""Entry point for the current SDSS machine learning project phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.classification import run_knn_classification
from src.clustering import run_kmeans_clustering
from src.preprocessing import clean_data, inspect_data, load_data
from src.regression import run_linear_regression
from src.reporting import build_compact_pipeline_report, save_pipeline_outputs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the current project phase."""
    parser = argparse.ArgumentParser(
        description="Run the current SDSS project workflow."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("sdss_sample.csv"),
        help="Path to the CSV dataset file.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Directory reserved for generated artifacts in later phases.",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip the KNN classification stage.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip the linear regression stage.",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip the KMeans clustering stage.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the current workflow without advancing future phases."""
    args = parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)

    raw_data = load_data(args.data)
    raw_summary = inspect_data(raw_data)

    cleaned_data = clean_data(raw_data)
    cleaned_summary = inspect_data(cleaned_data)

    report = {
        "phase": "phase_5",
        "dataset_path": str(args.data),
        "raw_data": raw_summary,
        "cleaned_data": cleaned_summary,
    }

    if not args.skip_classification:
        report["classification"] = run_knn_classification(cleaned_data)

    if not args.skip_regression:
        report["regression"] = run_linear_regression(cleaned_data)

    if not args.skip_clustering:
        report["clustering"] = run_kmeans_clustering(cleaned_data)

    save_pipeline_outputs(report, args.outputs)
    print(json.dumps(build_compact_pipeline_report(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
