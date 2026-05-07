#!/usr/bin/env python3
"""Run the Stage 3 notebook-aligned MFCC CNN replication experiments.

This CLI mirrors `scripts/run_cnn_original.py`: it reads the same YAML configs,
uses the same official split JSON files, calls the librosa-based Stage 3
segmenter, and writes fold CSVs plus `summary.json` under
`paper_replication/results/<experiment_id>/mfcc_cnn/`. The notebook-aligned
variant is the only MFCC method this runner supports; non-aligned baselines and
n_fft sweeps have been removed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLICATION_ROOT))

from mfcc_cnn import CLASSES, CLASS_TO_IDX, train_and_evaluate_fold  # noqa: E402
from mfcc_cnn.segmenter import prepare_dirs_with_librosa_resample  # noqa: E402
from run_cnn_original import import_original_wrapper, resolve_path  # noqa: E402


def format_kernel_list(values: list[int] | tuple[int, ...]) -> str:
    """Format kernel lists compactly for fold logs and notes.

    Args:
        values: Kernel-size values.

    Returns:
        A compact list string such as `[3,3,2]`.
    """

    return "[" + ",".join(str(int(value)) for value in values) + "]"


def format_seed_list(values: list[int] | tuple[int, ...]) -> str:
    """Format seed lists compactly for fold logs and notes."""

    return "[" + ",".join(str(int(value)) for value in values) + "]"


def mean_std(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Return population mean/std for a numeric vector."""

    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def ci95_pct(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Return a t-based 95% CI for the mean, expressed as percentage points."""

    array = np.asarray(values, dtype=float)
    if array.size <= 1:
        return float("nan"), float("nan")
    center = float(array.mean())
    sem = float(stats.sem(array, ddof=0))
    half_width = float(stats.t.ppf(0.975, array.size - 1) * sem)
    return 100.0 * (center - half_width), 100.0 * (center + half_width)


def ensure_writable_runtime_caches(replication_root: Path) -> None:
    """Provide writable cache directories for imported audio/plotting packages.

    Args:
        replication_root: Path to the `paper_replication` directory.

    Returns:
        None.
    """

    cache_root = replication_root / ".cache"
    defaults = {
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for env_name, path in defaults.items():
        os.environ.setdefault(env_name, str(path))
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch like the mel CNN runner.

    Args:
        seed: Integer seed from the YAML config.

    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_mfcc_cnn(cfg: dict, replication_root: Path) -> dict:
    """Run all folds for one notebook-aligned MFCC CNN experiment config.

    Args:
        cfg: Parsed YAML configuration.
        replication_root: Path to the `paper_replication` directory.

    Returns:
        Summary dictionary written to `summary.json`.

    Raises:
        FileNotFoundError: If configured dataset or split paths are missing.
        ValueError: If prepared data, labels, or training settings are invalid.
    """

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    ensure_writable_runtime_caches(replication_root)
    if "mfcc_cnn" not in cfg or "sample_rate_hz" not in cfg["mfcc_cnn"]:
        raise ValueError("Missing required config key: mfcc_cnn.sample_rate_hz")
    mfcc_cfg = cfg["mfcc_cnn"]

    mfcc_sample_rate_hz = int(mfcc_cfg["sample_rate_hz"])
    effective_n_fft = int(mfcc_cfg.get("n_fft", 4096))
    method = "mfcc_cnn"

    mfcc_train_cfg = dict(cfg["cnn"])
    segment_for_stage3 = float(mfcc_cfg.get("segment_seconds", cfg["cnn"]["segment_seconds"]))
    if segment_for_stage3 <= 0:
        raise ValueError("mfcc_cnn.segment_seconds must be positive.")
    if "seeds" not in mfcc_cfg:
        raise ValueError("Missing required config key: mfcc_cnn.seeds")
    raw_seeds = mfcc_cfg["seeds"]
    if not isinstance(raw_seeds, list) or len(raw_seeds) == 0:
        raise ValueError("mfcc_cnn.seeds must be a non-empty list.")
    aligned_seeds = [int(value) for value in raw_seeds]
    aligned_training = dict(mfcc_cfg.get("training", {}))
    mfcc_train_cfg.update(aligned_training)
    mfcc_train_cfg["segment_seconds"] = segment_for_stage3
    mfcc_train_cfg["sample_rate_hz"] = mfcc_sample_rate_hz
    mfcc_train_cfg["n_fft"] = effective_n_fft

    for key in (
        "n_mfcc",
        "hop_length",
        "center",
        "use_cmvn",
        "conv_kernels",
        "pool_kernels",
        "conv_padding",
    ):
        if key not in mfcc_cfg:
            raise ValueError(f"Missing required config key: mfcc_cnn.{key}")
        mfcc_train_cfg[key] = mfcc_cfg[key]
    for key in ("init_scheme", "bn_keras_defaults", "optimizer_eps"):
        if key not in mfcc_cfg:
            raise ValueError(f"Missing required config key: mfcc_cnn.{key}")
    conv_kernels = [int(value) for value in mfcc_train_cfg["conv_kernels"]]
    pool_kernels = [int(value) for value in mfcc_train_cfg["pool_kernels"]]
    conv_padding = str(mfcc_train_cfg["conv_padding"])
    init_scheme = str(mfcc_cfg["init_scheme"])
    bn_keras_defaults = mfcc_cfg["bn_keras_defaults"]
    optimizer_eps = mfcc_cfg["optimizer_eps"]
    if len(conv_kernels) != 3:
        raise ValueError("mfcc_cnn.conv_kernels must contain exactly three values.")
    if len(pool_kernels) != 3:
        raise ValueError("mfcc_cnn.pool_kernels must contain exactly three values.")
    if conv_padding not in {"valid", "same"}:
        raise ValueError("mfcc_cnn.conv_padding must be 'valid' or 'same'.")
    if init_scheme not in {"keras_glorot", "pytorch_default"}:
        raise ValueError("mfcc_cnn.init_scheme must be 'keras_glorot' or 'pytorch_default'.")
    if not isinstance(bn_keras_defaults, bool):
        raise ValueError("mfcc_cnn.bn_keras_defaults must be a boolean.")
    if isinstance(optimizer_eps, bool):
        raise ValueError("mfcc_cnn.optimizer_eps must be a float.")
    optimizer_eps = float(optimizer_eps)
    if optimizer_eps <= 0.0:
        raise ValueError("mfcc_cnn.optimizer_eps must be positive.")
    mfcc_train_cfg["conv_kernels"] = conv_kernels
    mfcc_train_cfg["pool_kernels"] = pool_kernels
    mfcc_train_cfg["conv_padding"] = conv_padding
    mfcc_train_cfg["init_scheme"] = init_scheme
    mfcc_train_cfg["bn_keras_defaults"] = bool(bn_keras_defaults)
    mfcc_train_cfg["optimizer_eps"] = optimizer_eps

    original_repo_root = resolve_path(replication_root, cfg["original_repo_root"])
    wav_root = resolve_path(replication_root, cfg["wav_root"])
    split_json = resolve_path(replication_root, cfg["split_json"])
    if not split_json.exists():
        raise FileNotFoundError(f"Missing split JSON: {split_json}")
    if not wav_root.exists():
        raise FileNotFoundError(f"Missing WAV root: {wav_root}")

    out_dir = replication_root / "results" / cfg["experiment_id"] / method
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = replication_root / "work" / "mfcc_cnn_segments" / cfg["experiment_id"]
    work_root.mkdir(parents=True, exist_ok=True)
    pkl_dir = replication_root / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    crawl_directory, _deep_audio_training, _prepare_dirs, _validate_on_test = import_original_wrapper(original_repo_root)

    splits = json.loads(split_json.read_text())
    songs = crawl_directory(str(wav_root), ".wav")
    metrics_rows = []
    seed_metrics_rows = []
    prediction_frames = []
    fold_train_seconds: dict[str, list[float]] = {}
    model_params_for_report: int | None = None
    aggregate_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    started = time.time()

    print(f"Torch version: {torch.__version__}")
    print(f"MFCC CNN experiment: {cfg['experiment_id']}")
    print(f"MFCC CNN method: {method}")
    print(f"Effective MFCC n_fft: {effective_n_fft}")
    print(f"Prepared segment work root: {work_root}")

    for fold_name, payload in splits.items():
        architecture_log = (
            f"conv_kernels={format_kernel_list(mfcc_train_cfg['conv_kernels'])} "
            f"pool_kernels={format_kernel_list(mfcc_train_cfg['pool_kernels'])} "
            f"conv_padding={mfcc_train_cfg['conv_padding']} "
        )
        parity_log = (
            f"init={mfcc_train_cfg['init_scheme']} "
            f"bn_keras={bool(mfcc_train_cfg['bn_keras_defaults'])} "
            f"adam_eps={float(mfcc_train_cfg['optimizer_eps'])} "
        )
        audio_backend_log = "audio_backend=librosa "
        seeds_log = f"seeds={format_seed_list(aligned_seeds)} "
        print(f"\n================ MFCC CNN {cfg['experiment_id']} {fold_name} ================")
        print(
            "Variant config: "
            f"SR={mfcc_sample_rate_hz} "
            f"seg={segment_for_stage3:g}s "
            f"n_mfcc={int(mfcc_train_cfg.get('n_mfcc', 40))} "
            f"n_fft={effective_n_fft} "
            f"hop={int(mfcc_train_cfg.get('hop_length', 512))} "
            f"center={bool(mfcc_train_cfg.get('center', False))} "
            f"CMVN={'on' if bool(mfcc_train_cfg.get('use_cmvn', True)) else 'off'} "
            f"{architecture_log}"
            f"{parity_log}"
            f"{audio_backend_log}"
            f"{seeds_log}"
            f"optimizer={mfcc_train_cfg.get('optimizer', 'AdamW')} "
            f"lr={float(mfcc_train_cfg['learning_rate'])} "
            f"batch={int(mfcc_train_cfg['batch_size'])} "
            f"epochs={int(mfcc_train_cfg['epochs'])} "
            f"early_stopping={bool(mfcc_train_cfg.get('early_stopping', False))}"
        )
        train_set = set(payload["train"])
        test_set = set(payload["test"])
        train_wavs = []
        test_wavs = []
        for song in songs:
            base = os.path.basename(song)
            if base in train_set:
                train_wavs.append(base)
            elif base in test_set:
                test_wavs.append(base)

        fold_seed_rows = []
        fold_true_names_for_report = []
        fold_pred_names_for_report = []
        fold_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

        for run_seed in aligned_seeds:
            print(f"--- seed={int(run_seed)} ---")
            seed_everything(int(run_seed))

            fold_work = work_root / fold_name
            if fold_work.exists():
                shutil.rmtree(fold_work)
            fold_work.mkdir(parents=True, exist_ok=True)

            print(f"Preparing segmented directories in {fold_work}")
            print(f"Train files: {len(train_wavs)}")
            print(f"Test files: {len(test_wavs)}")
            prepare_dirs_with_librosa_resample(
                str(wav_root),
                train_wavs,
                test_wavs,
                str(fold_work),
                segment_for_stage3,
                False,
                mfcc_sample_rate_hz,
            )

            # Aligned runs are large in the course-workspace disk budget; the
            # mirrored artifacts are the CSV/JSON result files, so checkpoints
            # are not persisted.
            model_path = None

            fold_result = train_and_evaluate_fold(
                fold_work=fold_work,
                cnn_cfg=mfcc_train_cfg,
                seed=int(run_seed),
                model_path=model_path,
            )

            y_true_names = fold_result.y_true_names
            y_pred_names = fold_result.y_pred_names
            y_true = np.asarray(fold_result.y_true_indices, dtype=int)
            y_pred = np.asarray(fold_result.y_pred_indices, dtype=int)
            cm = confusion_matrix(y_true_names, y_pred_names, labels=CLASSES)
            fold_cm += cm

            accuracy = accuracy_score(y_true_names, y_pred_names)
            precision_macro = precision_score(y_true_names, y_pred_names, average="macro", zero_division=0, labels=CLASSES)
            recall_macro = recall_score(y_true_names, y_pred_names, average="macro", zero_division=0, labels=CLASSES)
            repo_f1 = 0.0 if (precision_macro + recall_macro) == 0 else 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)
            sklearn_macro_f1 = f1_score(y_true_names, y_pred_names, average="macro", zero_division=0, labels=CLASSES)
            model_params_for_report = int(fold_result.model_params)

            print(f"Accuracy: {accuracy:.6f}")
            print(f"Repo F1 from macro precision/recall: {repo_f1:.6f}")
            print(f"Sklearn macro F1: {sklearn_macro_f1:.6f}")
            print(classification_report(y_true_names, y_pred_names, labels=CLASSES, zero_division=0))

            common_row = {
                "fold": fold_name,
                "n_train_files": int(len(train_wavs)),
                "n_test_files": int(len(test_wavs)),
                "n_train_segments": int(fold_result.n_train_segments),
                "n_val_segments": int(fold_result.n_val_segments),
                "n_test_segments": int(fold_result.n_test_segments),
                "n_test_files_aggregated": int(fold_result.n_test_files),
                "best_epoch": int(fold_result.best_epoch),
                "best_val_loss": fold_result.best_val_loss,
                "mfcc_fixed_frames": int(fold_result.mfcc_config.fixed_frames),
                "mfcc_n_mfcc": int(fold_result.mfcc_config.n_mfcc),
                "mfcc_n_fft": int(fold_result.mfcc_config.n_fft),
                "mfcc_hop_length": int(fold_result.mfcc_config.hop_length),
                "mfcc_center": bool(fold_result.mfcc_config.center),
                "mfcc_use_cmvn": bool(fold_result.mfcc_config.use_cmvn),
                "pool_kernel": "",
                "conv_kernels": format_kernel_list(mfcc_train_cfg["conv_kernels"]),
                "pool_kernels": format_kernel_list(mfcc_train_cfg["pool_kernels"]),
                "conv_padding": str(mfcc_train_cfg["conv_padding"]),
                "model_flat_dim": int(fold_result.flat_dim),
            }
            run_metrics = {
                "accuracy": float(accuracy),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "repo_f1_from_macro_pr": float(repo_f1),
                "sklearn_macro_f1": float(sklearn_macro_f1),
                "train_seconds": float(fold_result.train_seconds),
                "model_params": int(fold_result.model_params),
                "model_path": "",
            }
            fold_seed_rows.append({"seed": int(run_seed), **common_row, **run_metrics})
            seed_metrics_rows.append(
                {
                    "experiment_id": cfg["experiment_id"],
                    "fold": fold_name,
                    "seed": int(run_seed),
                    "accuracy": float(accuracy),
                    "repo_f1": float(repo_f1),
                    "sklearn_macro_f1": float(sklearn_macro_f1),
                    "train_seconds": float(fold_result.train_seconds),
                    "model_params": int(fold_result.model_params),
                }
            )
            pred_df = pd.DataFrame(
                {
                    "fold": fold_name,
                    "seed": int(run_seed),
                    "true_class": y_true_names,
                    "pred_class": y_pred_names,
                    "label": y_true,
                    "pred_label": y_pred,
                }
            )
            fold_true_names_for_report.extend(y_true_names)
            fold_pred_names_for_report.extend(y_pred_names)

            prediction_frames.append(pred_df)
            shutil.rmtree(fold_work / "train", ignore_errors=True)
            shutil.rmtree(fold_work / "test", ignore_errors=True)

        if not fold_seed_rows:
            raise ValueError(f"No seed runs completed for fold {fold_name}.")
        fold_seed_df = pd.DataFrame(fold_seed_rows)
        accuracy_mean, accuracy_std = mean_std(fold_seed_df["accuracy"].to_numpy(dtype=float))
        repo_f1_mean, repo_f1_std = mean_std(fold_seed_df["repo_f1_from_macro_pr"].to_numpy(dtype=float))
        sklearn_f1_mean, sklearn_f1_std = mean_std(fold_seed_df["sklearn_macro_f1"].to_numpy(dtype=float))
        train_seconds_mean, train_seconds_std = mean_std(fold_seed_df["train_seconds"].to_numpy(dtype=float))
        best_epoch_mean, best_epoch_std = mean_std(fold_seed_df["best_epoch"].to_numpy(dtype=float))
        best_val_loss_values = pd.to_numeric(fold_seed_df["best_val_loss"], errors="coerce").to_numpy(dtype=float)
        best_val_loss_mean = float(np.nanmean(best_val_loss_values))
        best_val_loss_std = float(np.nanstd(best_val_loss_values))
        first_row = fold_seed_rows[0]
        metrics_rows.append(
            {
                "fold": fold_name,
                "n_seeds": int(len(aligned_seeds)),
                "seeds": format_seed_list(aligned_seeds),
                "n_train_files": first_row["n_train_files"],
                "n_test_files": first_row["n_test_files"],
                "n_train_segments": first_row["n_train_segments"],
                "n_val_segments": first_row["n_val_segments"],
                "n_test_segments": first_row["n_test_segments"],
                "n_test_files_aggregated": first_row["n_test_files_aggregated"],
                "best_epoch_mean": best_epoch_mean,
                "best_epoch_std": best_epoch_std,
                "best_val_loss_mean": best_val_loss_mean,
                "best_val_loss_std": best_val_loss_std,
                "mfcc_fixed_frames": first_row["mfcc_fixed_frames"],
                "mfcc_n_mfcc": first_row["mfcc_n_mfcc"],
                "mfcc_n_fft": first_row["mfcc_n_fft"],
                "mfcc_hop_length": first_row["mfcc_hop_length"],
                "mfcc_center": first_row["mfcc_center"],
                "mfcc_use_cmvn": first_row["mfcc_use_cmvn"],
                "pool_kernel": "",
                "conv_kernels": first_row["conv_kernels"],
                "pool_kernels": first_row["pool_kernels"],
                "conv_padding": first_row["conv_padding"],
                "model_flat_dim": first_row["model_flat_dim"],
                "model_params": first_row["model_params"],
                "train_seconds_mean": train_seconds_mean,
                "train_seconds_std": train_seconds_std,
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "repo_f1_mean": repo_f1_mean,
                "repo_f1_std": repo_f1_std,
                "sklearn_macro_f1_mean": sklearn_f1_mean,
                "sklearn_macro_f1_std": sklearn_f1_std,
                "confusion_matrix_note": "summed_over_seeds_for_visualization",
                "model_path": "",
            }
        )
        aggregate_cm += fold_cm
        fold_train_seconds[fold_name] = [float(value) for value in fold_seed_df["train_seconds"].tolist()]
        pd.DataFrame(
            classification_report(
                fold_true_names_for_report,
                fold_pred_names_for_report,
                labels=CLASSES,
                output_dict=True,
                zero_division=0,
            )
        ).T.to_csv(out_dir / f"{fold_name}_classification_report.csv")
        pd.DataFrame(fold_cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / f"{fold_name}_confusion_matrix.csv")

    metrics_df = pd.DataFrame(metrics_rows)
    pred_all = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    pred_all.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(aggregate_cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "aggregated_confusion_matrix.csv")
    np.save(out_dir / "aggregated_confusion_matrix.npy", aggregate_cm)

    seed_metrics_df = pd.DataFrame(
        seed_metrics_rows,
        columns=[
            "experiment_id",
            "fold",
            "seed",
            "accuracy",
            "repo_f1",
            "sklearn_macro_f1",
            "train_seconds",
            "model_params",
        ],
    )
    seed_metrics_df.to_csv(out_dir / "seed_metrics.csv", index=False)
    train_seconds_values = seed_metrics_df["train_seconds"].to_numpy(dtype=float)
    model_params = int(model_params_for_report or seed_metrics_df["model_params"].iloc[0])
    compute_report = {
        "method": method,
        "model_params": model_params,
        "fold_train_seconds": fold_train_seconds,
        "mean_train_seconds_per_fold_seed": float(train_seconds_values.mean()),
        "total_train_seconds": float(train_seconds_values.sum()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "seeds": [int(value) for value in aligned_seeds],
    }
    (out_dir / "compute_report.json").write_text(json.dumps(compute_report, indent=2))
    accuracy_vector = metrics_df["accuracy_mean"].to_numpy(dtype=float)
    f1_vector = metrics_df["repo_f1_mean"].to_numpy(dtype=float)
    sklearn_f1_vector = metrics_df["sklearn_macro_f1_mean"].to_numpy(dtype=float)

    accuracy_mean, accuracy_std = mean_std(accuracy_vector)
    f1_mean, f1_std = mean_std(f1_vector)
    sklearn_f1_mean, sklearn_f1_std = mean_std(sklearn_f1_vector)
    accuracy_ci95_low_pct, accuracy_ci95_high_pct = ci95_pct(accuracy_vector)
    f1_ci95_low_pct, f1_ci95_high_pct = ci95_pct(f1_vector)

    architecture_note = (
        "Notebook-strict aligned architecture uses "
        f"conv_kernels={format_kernel_list(mfcc_train_cfg['conv_kernels'])} with "
        f"conv_padding={mfcc_train_cfg['conv_padding']} and "
        f"pool_kernels={format_kernel_list(mfcc_train_cfg['pool_kernels'])} "
        "with Keras same stride-2 pooling."
    )
    audio_note = (
        f"Aligned Stage 3 MFCC audio is resampled to {mfcc_sample_rate_hz} Hz with "
        "librosa.load(..., mono=True), matching the notebook resampling path; ffmpeg is used "
        "only for time segmentation of already-resampled FLOAT WAV audio."
    )

    summary = {
        "experiment_id": cfg["experiment_id"],
        "method": method,
        "split_schema": cfg["split_schema"],
        "num_folds": int(len(metrics_df)),
        "accuracy_mean_pct": float(100.0 * accuracy_mean),
        "accuracy_std_pct": float(100.0 * accuracy_std),
        "f1_mean_pct": float(100.0 * f1_mean),
        "f1_std_pct": float(100.0 * f1_std),
        "sklearn_macro_f1_mean_pct": float(100.0 * sklearn_f1_mean),
        "sklearn_macro_f1_std_pct": float(100.0 * sklearn_f1_std),
        "runtime_seconds": float(time.time() - started),
        # MFCC is a new input front-end, but the delta columns compare against
        # the guitar paper's CNN target because it is the CNN-family baseline.
        "paper_target": cfg["paper_targets"]["cnn"],
        "notes": [
            (
                f"MFCC inputs use {int(mfcc_train_cfg.get('n_mfcc', 40))} coefficients, "
                f"n_fft={effective_n_fft}, hop_length={int(mfcc_train_cfg.get('hop_length', 512))}, "
                f"center={bool(mfcc_train_cfg.get('center', False))}, "
                f"and {'per-clip CMVN' if bool(mfcc_train_cfg.get('use_cmvn', True)) else 'no CMVN'}."
            ),
            audio_note,
            architecture_note,
            (
                f"Optimizer is {mfcc_train_cfg.get('optimizer', 'AdamW')} with "
                f"lr={float(mfcc_train_cfg['learning_rate'])}, batch_size={int(mfcc_train_cfg['batch_size'])}, "
                f"epochs={int(mfcc_train_cfg['epochs'])}, and early_stopping={bool(mfcc_train_cfg.get('early_stopping', False))}."
            ),
            (
                f"Training data comes from Stage 3 {float(mfcc_train_cfg['segment_seconds']):g}-second clips; "
                f"test WAVs are evaluated as {float(mfcc_train_cfg['segment_seconds']):g}-second MFCC windows "
                "and aggregated per file."
            ),
            "Architecture follows the Alar et al. Proposed CNN pattern: Conv/MaxPool/BatchNorm blocks with 32, 64, and 128 channels.",
            "Dense(128) is followed by Dropout(0.5), matching the author Proposed notebook rather than the Benchmark dropout.",
            "The final layer is linear and has a 9-class guitar head.",
            "Primary F1 is the repo definition: harmonic mean of macro precision and macro recall.",
        ],
        "seed_count": int(len(aligned_seeds)),
        "n_seeds_per_fold": int(len(aligned_seeds)),
        "accuracy_ci95_low_pct": float(accuracy_ci95_low_pct),
        "accuracy_ci95_high_pct": float(accuracy_ci95_high_pct),
        "f1_ci95_low_pct": float(f1_ci95_low_pct),
        "f1_ci95_high_pct": float(f1_ci95_high_pct),
    }
    summary["notes"].insert(
        3,
        (
            f"Keras-default parity uses init_scheme={mfcc_train_cfg['init_scheme']}, "
            f"bn_keras_defaults={bool(mfcc_train_cfg['bn_keras_defaults'])}, "
            f"optimizer_eps={float(mfcc_train_cfg['optimizer_eps'])}."
        ),
    )
    summary["notes"].insert(
        4,
        (
            f"Aligned metrics use seeds={format_seed_list(aligned_seeds)}: metrics are averaged across seeds "
            "within each fold, then summarized across fold means with t-based 95% CIs."
        ),
    )
    summary["notes"].insert(
        5,
        "Training wall-clock in compute_report.json is measured around fit_model only; confusion matrices are summed over seeds and folds for visualization.",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Parse CLI arguments and run one configured aligned MFCC CNN experiment.

    Args:
        None.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    replication_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(replication_root, args.config)
    cfg = yaml.safe_load(config_path.read_text())
    run_mfcc_cnn(cfg, replication_root)


if __name__ == "__main__":
    main()
