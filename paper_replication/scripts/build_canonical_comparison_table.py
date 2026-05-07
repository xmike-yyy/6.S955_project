#!/usr/bin/env python3
"""Build the report-facing 9-row comparison table.

Run from ``paper_replication/`` after ``scripts/collect_results.py`` has produced
``results/comparison_table.csv``:

    python scripts/build_canonical_comparison_table.py

The default results directory resolves to ``paper_replication/results`` based on
this script's location. If ``--results-dir`` is provided as a relative path, it
is resolved relative to the current working directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_EXPERIMENT_IDS = ("experiment_1", "experiment_2", "experiment_3")
CANONICAL_METHODS = ("svm", "cnn", "mfcc_cnn")
METHOD_ORDER = {method: index for index, method in enumerate(CANONICAL_METHODS)}


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir(),
        help="Directory containing comparison_table.csv. Defaults to paper_replication/results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    source_path = results_dir / "comparison_table.csv"
    output_path = results_dir / "comparison_table_canonical.csv"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source comparison table: {source_path}")

    source = pd.read_csv(source_path)
    canonical = source[
        source["experiment_id"].isin(EXPECTED_EXPERIMENT_IDS)
        & source["method"].isin(CANONICAL_METHODS)
    ].copy()

    existing_pairs = set(zip(canonical["experiment_id"], canonical["method"]))
    for experiment_id in EXPECTED_EXPERIMENT_IDS:
        for method in CANONICAL_METHODS:
            if (experiment_id, method) not in existing_pairs:
                print(f"WARNING missing expected row: experiment_id={experiment_id} method={method}")

    if not canonical.empty:
        canonical["_experiment_order"] = canonical["experiment_id"].map(
            {experiment_id: index for index, experiment_id in enumerate(EXPECTED_EXPERIMENT_IDS)}
        )
        canonical["_method_order"] = canonical["method"].map(METHOD_ORDER)
        canonical = canonical.sort_values(["_experiment_order", "_method_order"]).drop(
            columns=["_experiment_order", "_method_order"]
        )

    canonical.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(canonical)} rows)")


if __name__ == "__main__":
    main()
