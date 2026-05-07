#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def fmt(value) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2f}"


def fmt_status(value) -> str:
    if pd.isna(value):
        return "NA"
    return str(value)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    comparison_path = root / "results" / "comparison_table.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        df = df[df["method"].isin({"svm", "cnn", "mfcc_cnn"})]
    else:
        df = pd.DataFrame()

    lines = [
        "# Replication Report",
        "",
        "Generated from `paper_replication/results`.",
        "",
        "## Scope",
        "",
        "This replication attempts the three paper experiments using the local original `guitar_style_dataset` repository and the already-downloaded dataset. Each experiment runs **SVM**, **mel-spectrogram CNN**, and the notebook-aligned **MFCC CNN** (`mfcc_cnn`, Alar-style architecture on guitar audio) on the **same official splits** (Table 3 baselines for SVM and mel CNN). The only MFCC variant reported is `mfcc_cnn`; non-aligned baselines and `n_fft` sweeps are not included. MFCC summaries use the guitar-paper **CNN** targets in `paper_target` for delta columns so Stage 3 is comparable to the published mel-CNN numbers, not a separate paper table.",
        "",
        "## Environment Outcome",
        "",
        "- Docker is the preferred reproducibility target, and a Dockerfile is provided in `environment/`.",
        "- Docker was not installed on the execution host, so runs used the local `.venv` fallback.",
        "- The local fallback uses Python 3.11 and modern Torch because the original CNN pins include `torch==1.9.0`, which is not compatible with the available host Python.",
        "- Compatibility shims were limited to runtime issues: ignore the removed `verbose` keyword in `ReduceLROnPlateau`, skip a diagnostic histogram plot that crashed in headless macOS execution, and force Torch dataloaders to `num_workers=0` after shared-memory worker launch failed.",
        "",
        "## Results",
        "",
        "When available, `comparison_table.csv` includes t-based 95% CI columns for reproduced MFCC metrics; the compact table below reports means and deltas.",
        "",
    ]

    if df.empty:
        lines.append("No completed experiment summaries were found yet.")
    else:
        lines.extend(
            [
                "| Experiment | Method | Status | Folds | Paper Acc | Reproduced Acc | Acc Delta | Paper F1 | Reproduced F1 | F1 Delta | Within Paper F1 Std |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in df.to_dict(orient="records"):
            folds = f"{int(row['num_folds_completed'])}/{int(row['num_folds'])}" if not pd.isna(row.get("num_folds_completed")) else "NA"
            lines.append(
                "| {experiment_id} | {method} | {status} | {folds} | {paper_acc} | {rep_acc} | {acc_delta} | {paper_f1} | {rep_f1} | {f1_delta} | {within} |".format(
                    experiment_id=row["experiment_id"],
                    method=row["method"],
                    status=row.get("status", "completed"),
                    folds=folds,
                    paper_acc=fmt(row["paper_accuracy_pct"]),
                    rep_acc=fmt(row["reproduced_accuracy_pct"]),
                    acc_delta=fmt(row["accuracy_delta_pct_points"]),
                    paper_f1=fmt(row["paper_f1_pct"]),
                    rep_f1=fmt(row["reproduced_f1_pct"]),
                    f1_delta=fmt(row["f1_delta_pct_points"]),
                    within=fmt_status(row["within_paper_f1_std"]),
                )
            )

        completed = df[df["status"] == "completed"]
        all_within = completed["within_paper_f1_std"].fillna(False).all() if not completed.empty else False
        any_within = df["within_paper_f1_std"].fillna(False).any()
        methods_present = set(df["method"].astype(str).str.strip())
        has_mfcc = "mfcc_cnn" in methods_present
        # Six rows = 3 experiments × (SVM + CNN). Nine rows = same plus MFCC CNN per experiment.
        expected_completed = 9 if has_mfcc else 6
        if all_within and len(completed) == expected_completed:
            verdict = "fully replicated"
        elif any_within:
            verdict = "partially replicated"
        else:
            verdict = "not replicated"
        lines.extend(
            [
                "",
                f"Overall verdict: **{verdict}**.",
                "",
                f"(Verdict expects **{expected_completed}** completed rows when MFCC summaries are present in the table, otherwise **6**.)",
            ]
        )

    lines.extend(
        [
            "",
            "## Matched Exactly",
            "",
            "- Local dataset archive is reused, not redownloaded.",
            "- Official split JSON files are used for the three paper experiments.",
            "- SVM grid values match the paper and original repo.",
            "- CNN architecture and training defaults come from `deep-audio-features==0.2.18`, the original repo dependency.",
            "",
            "## Approximations And Deviations",
            "",
            "- The paper PDF path requested by the user was not present inside `6.S955_project`; the actual local PDF used is `../../Guitar_dataset.pdf` from `paper_replication`.",
            "- Docker execution was not possible because the host has no `docker` binary.",
            "- Local Python 3.11 required Torch 2.9 rather than the original CNN requirement `torch==1.9.0`.",
            "- The original SVM code path extracts one long-term averaged feature vector per WAV file, while the paper prose describes 1-second segment-level vectors.",
            "- CNN F1 uses the original wrapper formula, the harmonic mean of macro precision and macro recall; sklearn macro-F1 is also saved in fold metrics.",
            "- CNN full reproduction was blocked by CPU runtime in this interactive session: experiment 1 fold 0 completed, fold 1 was interrupted at epoch 15, and experiments 2-3 were left as full-fidelity rerun commands.",
            "",
            "## Confidence",
            "",
            "Confidence is highest for data/split parity, SVM code mapping, and the completed SVM metrics. Confidence is lower for exact CNN numerical reproduction because the host could not run the original Docker-like Python/Torch stack and the full 11-fold CPU run exceeded the interactive budget.",
            "",
        ]
    )

    report_path = root / "replication_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
