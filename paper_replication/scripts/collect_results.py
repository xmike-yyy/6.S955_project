#!/usr/bin/env python3
"""Aggregate canonical `summary.json` files under `results/experiment_*/**/` into comparison tables.

Only the three canonical methods appear in `comparison_table.csv` / `.json`:
`svm`, `cnn`, and `mfcc_cnn`. Any other `method` value found in a discovered
`summary.json` is skipped defensively, so stale or experimental MFCC variants no longer
contaminate the comparison output.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

CANONICAL_METHODS = frozenset({"svm", "cnn", "mfcc_cnn"})

# Stable table order: guitar-paper baselines first, then notebook-aligned MFCC CNN.
_METHOD_SORT = {"svm": 0, "cnn": 1, "mfcc_cnn": 2}


def numeric_or_nan(value):
    if value is None:
        return float("nan")
    return value


def main() -> None:
    replication_root = Path(__file__).resolve().parents[1]
    rows = []
    for summary_path in sorted((replication_root / "results").glob("experiment_*/**/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("method") not in CANONICAL_METHODS:
            continue
        target = summary.get("paper_target", {})
        reproduced_accuracy = numeric_or_nan(summary.get("accuracy_mean_pct"))
        reproduced_f1 = numeric_or_nan(summary.get("f1_mean_pct"))
        paper_accuracy = numeric_or_nan(target.get("accuracy_pct"))
        paper_f1 = numeric_or_nan(target.get("f1_pct"))
        row = {
            "experiment_id": summary.get("experiment_id"),
            "method": summary.get("method"),
            "status": summary.get("status", "completed"),
            "split_schema": summary.get("split_schema"),
            "num_folds": summary.get("num_folds"),
            "num_folds_completed": summary.get("num_folds_completed", summary.get("num_folds")),
            "paper_accuracy_pct": paper_accuracy,
            "reproduced_accuracy_pct": reproduced_accuracy,
            "reproduced_accuracy_ci95_low_pct": numeric_or_nan(summary.get("accuracy_ci95_low_pct")),
            "reproduced_accuracy_ci95_high_pct": numeric_or_nan(summary.get("accuracy_ci95_high_pct")),
            "accuracy_delta_pct_points": reproduced_accuracy - paper_accuracy,
            "paper_f1_pct": paper_f1,
            "reproduced_f1_pct": reproduced_f1,
            "reproduced_f1_ci95_low_pct": numeric_or_nan(summary.get("f1_ci95_low_pct")),
            "reproduced_f1_ci95_high_pct": numeric_or_nan(summary.get("f1_ci95_high_pct")),
            "f1_delta_pct_points": reproduced_f1 - paper_f1,
            "paper_f1_std_pct": target.get("f1_std_pct"),
            "reproduced_f1_std_pct": summary.get("f1_std_pct"),
            "summary_path": str(summary_path),
        }
        tolerance = target.get("f1_std_pct")
        if tolerance is None or math.isnan(row["f1_delta_pct_points"]):
            row["within_paper_f1_std"] = None
        else:
            row["within_paper_f1_std"] = abs(row["f1_delta_pct_points"]) <= tolerance
        rows.append(row)

    out_dir = replication_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        df["_method_order"] = df["method"].map(lambda m: _METHOD_SORT.get(str(m), 99))
        df = df.sort_values(["experiment_id", "_method_order", "method"]).drop(columns="_method_order")
    else:
        df = pd.DataFrame()
    df.to_csv(out_dir / "comparison_table.csv", index=False)
    (out_dir / "comparison_table.json").write_text(df.to_json(orient="records", indent=2))
    print(df.to_string(index=False) if not df.empty else "No summary files found yet.")


if __name__ == "__main__":
    main()
