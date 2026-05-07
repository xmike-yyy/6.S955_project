#!/usr/bin/env python3
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


CLASSES = [
    "alternate picking",
    "legato",
    "tapping",
    "sweep picking",
    "vibrato",
    "hammer on",
    "pull off",
    "slide",
    "bend",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def resolve_path(replication_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (replication_root / path).resolve()


def install_torch_scheduler_compat() -> None:
    """deep-audio-features 0.2.18 passes verbose= to ReduceLROnPlateau.

    PyTorch 2.9 removed that keyword. This shim ignores the keyword while
    preserving all other scheduler behavior.
    """

    original_cls = torch.optim.lr_scheduler.ReduceLROnPlateau

    class CompatReduceLROnPlateau(original_cls):
        def __init__(
            self,
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=False,
        ):
            del verbose
            super().__init__(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                eps=eps,
            )

    torch.optim.lr_scheduler.ReduceLROnPlateau = CompatReduceLROnPlateau


def import_original_wrapper(original_repo_root: Path):
    install_torch_scheduler_compat()
    wrapper_dir = original_repo_root / "deep_audio_features_wrapper"
    sys.path.insert(0, str(wrapper_dir))
    sys.path.insert(0, str(original_repo_root))
    from deep_audio_utils import crawl_directory, deep_audio_training, prepare_dirs, validate_on_test
    from deep_audio_features.bin import basic_test as btest
    from deep_audio_features.bin import basic_training as bt
    from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset
    from torch.utils.data import DataLoader as TorchDataLoader

    def skip_histogram_plot(self, spec_sizes, labels):
        del self, spec_sizes, labels
        print("--> Skipping diagnostic spectrogram-size histogram plot in headless local runtime.")

    def single_process_dataloader(*args, **kwargs):
        if kwargs.get("num_workers", 0) != 0:
            print("--> Forcing DataLoader num_workers=0 because torch shared-memory workers are unavailable.")
        kwargs["num_workers"] = 0
        return TorchDataLoader(*args, **kwargs)

    FeatureExtractorDataset.plot_hist = skip_histogram_plot
    bt.DataLoader = single_process_dataloader
    btest.DataLoader = single_process_dataloader

    return crawl_directory, deep_audio_training, prepare_dirs, validate_on_test


def run_cnn(cfg: dict, replication_root: Path) -> dict:
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    original_repo_root = resolve_path(replication_root, cfg["original_repo_root"])
    wav_root = resolve_path(replication_root, cfg["wav_root"])
    split_json = resolve_path(replication_root, cfg["split_json"])
    out_dir = replication_root / "results" / cfg["experiment_id"] / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = replication_root / "work" / "cnn_segments" / cfg["experiment_id"]
    work_root.mkdir(parents=True, exist_ok=True)
    pkl_dir = replication_root / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    crawl_directory, deep_audio_training, prepare_dirs, validate_on_test = import_original_wrapper(original_repo_root)

    splits = json.loads(split_json.read_text())
    songs = crawl_directory(str(wav_root), extension=".wav")
    metrics_rows = []
    prediction_frames = []
    aggregate_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    started = time.time()

    print(f"Torch version: {torch.__version__}")
    print(f"Device seen by deep_audio_features: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Working directory for pkl outputs: {Path.cwd()}")

    for fold_name, payload in splits.items():
        print(f"\n================ CNN {cfg['experiment_id']} {fold_name} ================")
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

        fold_work = work_root / fold_name
        if fold_work.exists():
            shutil.rmtree(fold_work)
        fold_work.mkdir(parents=True, exist_ok=True)

        print(f"Preparing segmented directories in {fold_work}")
        print(f"Train files: {len(train_wavs)}")
        print(f"Test files: {len(test_wavs)}")
        prepare_dirs(str(wav_root), train_wavs, test_wavs, str(fold_work), int(cfg["cnn"]["segment_seconds"]), False)

        model_name = f"{cfg['experiment_id']}_{fold_name}_cnn"
        model_path = pkl_dir / f"{model_name}.pt"
        if model_path.exists():
            model_path.unlink()

        print("Training starts with original deep_audio_features wrapper")
        deep_audio_training(str(fold_work), model_name)

        y_true_names, y_pred_names = validate_on_test(str(fold_work), str(model_path), False)
        y_true = np.array([CLASS_TO_IDX[name] for name in y_true_names], dtype=int)
        y_pred = np.array([CLASS_TO_IDX[name] for name in y_pred_names], dtype=int)
        cm = confusion_matrix(y_true_names, y_pred_names, labels=CLASSES)
        aggregate_cm += cm

        accuracy = accuracy_score(y_true_names, y_pred_names)
        precision_macro = precision_score(y_true_names, y_pred_names, average="macro", zero_division=0, labels=CLASSES)
        recall_macro = recall_score(y_true_names, y_pred_names, average="macro", zero_division=0, labels=CLASSES)
        repo_f1 = 0.0 if (precision_macro + recall_macro) == 0 else 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)
        sklearn_macro_f1 = f1_score(y_true_names, y_pred_names, average="macro", zero_division=0, labels=CLASSES)

        print(f"Accuracy: {accuracy:.6f}")
        print(f"Repo F1 from macro precision/recall: {repo_f1:.6f}")
        print(f"Sklearn macro F1: {sklearn_macro_f1:.6f}")
        print(classification_report(y_true_names, y_pred_names, labels=CLASSES, zero_division=0))

        metrics_rows.append(
            {
                "fold": fold_name,
                "n_train_files": int(len(train_wavs)),
                "n_test_files": int(len(test_wavs)),
                "accuracy": float(accuracy),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "repo_f1_from_macro_pr": float(repo_f1),
                "sklearn_macro_f1": float(sklearn_macro_f1),
                "model_path": str(model_path),
            }
        )
        pred_df = pd.DataFrame(
            {
                "fold": fold_name,
                "true_class": y_true_names,
                "pred_class": y_pred_names,
                "label": y_true,
                "pred_label": y_pred,
            }
        )
        prediction_frames.append(pred_df)
        pd.DataFrame(
            classification_report(y_true_names, y_pred_names, labels=CLASSES, output_dict=True, zero_division=0)
        ).T.to_csv(out_dir / f"{fold_name}_classification_report.csv")
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / f"{fold_name}_confusion_matrix.csv")

        shutil.rmtree(fold_work / "train", ignore_errors=True)
        shutil.rmtree(fold_work / "test", ignore_errors=True)

    metrics_df = pd.DataFrame(metrics_rows)
    pred_all = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    pred_all.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(aggregate_cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "aggregated_confusion_matrix.csv")
    np.save(out_dir / "aggregated_confusion_matrix.npy", aggregate_cm)

    summary = {
        "experiment_id": cfg["experiment_id"],
        "method": "cnn",
        "split_schema": cfg["split_schema"],
        "num_folds": int(len(metrics_df)),
        "accuracy_mean_pct": float(100.0 * metrics_df["accuracy"].mean()),
        "accuracy_std_pct": float(100.0 * metrics_df["accuracy"].std(ddof=0)),
        "f1_mean_pct": float(100.0 * metrics_df["repo_f1_from_macro_pr"].mean()),
        "f1_std_pct": float(100.0 * metrics_df["repo_f1_from_macro_pr"].std(ddof=0)),
        "sklearn_macro_f1_mean_pct": float(100.0 * metrics_df["sklearn_macro_f1"].mean()),
        "sklearn_macro_f1_std_pct": float(100.0 * metrics_df["sklearn_macro_f1"].std(ddof=0)),
        "runtime_seconds": float(time.time() - started),
        "paper_target": cfg["paper_targets"]["cnn"],
        "notes": [
            "Uses original repository deep_audio_features_wrapper logic.",
            "Applies a local PyTorch 2.9 compatibility shim for ReduceLROnPlateau(verbose=...).",
            "Primary F1 matches original wrapper formula: harmonic mean of macro precision and macro recall.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    replication_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(replication_root, args.config)
    cfg = yaml.safe_load(config_path.read_text())
    run_cnn(cfg, replication_root)


if __name__ == "__main__":
    main()
