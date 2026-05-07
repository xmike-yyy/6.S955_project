#!/usr/bin/env python3
"""Render report-facing confusion-matrix heatmaps.

Run from ``paper_replication/``:

    python scripts/plot_confusion_matrices.py

By default, this script looks in ``paper_replication/results`` based on this
file's location and plots ``svm``, ``cnn``, and ``mfcc_cnn`` only. A
relative ``--results-dir`` argument is resolved relative to the current working
directory. Pass ``--all`` to plot every method directory containing an
``aggregated_confusion_matrix.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHODS = ("svm", "cnn", "mfcc_cnn")


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir(),
        help="Directory containing experiment_* result subdirectories. Defaults to paper_replication/results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Method directories to plot. Ignored when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every method directory containing aggregated_confusion_matrix.csv.",
    )
    return parser.parse_args()


def candidate_method_dirs(results_dir: Path, methods: Iterable[str], plot_all: bool) -> list[Path]:
    experiment_dirs = sorted(path for path in results_dir.glob("experiment_*") if path.is_dir())
    if plot_all:
        return [
            method_dir
            for experiment_dir in experiment_dirs
            for method_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir())
            if (method_dir / "aggregated_confusion_matrix.csv").exists()
        ]

    return [experiment_dir / method for experiment_dir in experiment_dirs for method in methods]


def matrix_values(matrix: pd.DataFrame) -> np.ndarray:
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)


def annotate_cells(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    normalized: bool,
    threshold: float,
) -> None:
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            label = f"{value * 100:.2f}%" if normalized else f"{int(round(value))}"
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, label, ha="center", va="center", color=color, fontsize=7)


def save_heatmap(
    matrix: pd.DataFrame,
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
    normalized: bool,
) -> None:
    labels_x = [str(label) for label in matrix.columns]
    labels_y = [str(label) for label in matrix.index]
    size = max(7.0, 0.62 * max(len(labels_x), len(labels_y)))
    fig, ax = plt.subplots(figsize=(size + 1.3, size))
    image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels_x)), labels=labels_x, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels_y)), labels=labels_y)
    threshold = float(np.nanmax(values) * 0.5) if values.size else 0.0
    annotate_cells(ax, values, normalized=normalized, threshold=threshold)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_matrix(csv_path: Path) -> None:
    matrix = pd.read_csv(csv_path, index_col=0)
    counts = matrix_values(matrix)
    method = csv_path.parent.name
    title_prefix = f"{csv_path.parent.parent.name} {method}"

    raw_path = csv_path.parent / f"{method}_confusion.png"
    save_heatmap(matrix, counts, raw_path, title=f"{title_prefix} confusion matrix", normalized=False)

    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
    normalized_path = csv_path.parent / f"{method}_confusion_normalized.png"
    save_heatmap(
        matrix,
        normalized,
        normalized_path,
        title=f"{title_prefix} row-normalized confusion matrix",
        normalized=True,
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    for method_dir in candidate_method_dirs(results_dir, args.methods, args.all):
        csv_path = method_dir / "aggregated_confusion_matrix.csv"
        if not csv_path.exists():
            print(f"Skipping missing {csv_path}")
            continue
        plot_matrix(csv_path)


if __name__ == "__main__":
    main()
