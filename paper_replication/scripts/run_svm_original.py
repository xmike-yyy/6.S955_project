#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyAudioAnalysis import MidTermFeatures as aF
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_MAPPING = {
    "alternate picking": 0,
    "legato": 1,
    "tapping": 2,
    "sweep picking": 3,
    "vibrato": 4,
    "hammer on": 5,
    "pull off": 6,
    "slide": 7,
    "bend": 8,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
LABEL_ORDER = sorted(IDX_TO_CLASS)
CLASS_NAMES = [IDX_TO_CLASS[i] for i in LABEL_ORDER]


def resolve_path(replication_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (replication_root / path).resolve()


def load_or_extract_features(cfg: dict, replication_root: Path, wav_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    feature_dir = replication_root / "work" / "svm_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / "X_original_file_level.npy"
    meta_path = feature_dir / "file_index_original_file_level.csv"

    if feature_path.exists() and meta_path.exists():
        print(f"Loading cached SVM features from {feature_path}")
        return np.load(feature_path), pd.read_csv(meta_path)

    class_dirs = [p for p in sorted(wav_root.iterdir()) if p.is_dir()]
    print("Extracting original-repo SVM features with pyAudioAnalysis")
    print("Class directories:")
    for path in class_dirs:
        print(f"  {path}")

    svm_cfg = cfg["svm"]
    features, class_names, file_names = aF.multiple_directory_feature_extraction(
        [str(p) for p in class_dirs],
        float(svm_cfg["mid_window_seconds"]),
        float(svm_cfg["mid_step_seconds"]),
        float(svm_cfg["short_window_seconds"]),
        float(svm_cfg["short_step_seconds"]),
        compute_beat=False,
    )

    rows = []
    flat_features = []
    feat_idx = 0
    for class_dir, class_features, class_file_names in zip(class_dirs, features, file_names):
        class_name = class_dir.name
        label = CLASS_MAPPING[class_name]
        for local_idx, file_path in enumerate(class_file_names):
            flat_features.append(class_features[local_idx])
            rows.append(
                {
                    "feat_idx": feat_idx,
                    "file_name": Path(file_path).name,
                    "file_path": str(file_path),
                    "class_name": class_name,
                    "label": label,
                }
            )
            feat_idx += 1

    X = np.asarray(flat_features, dtype=np.float32)
    meta_df = pd.DataFrame(rows)
    np.save(feature_path, X)
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved features: {feature_path} shape={X.shape}")
    print(f"Saved feature index: {meta_path}")
    return X, meta_df


def run_svm(cfg: dict, replication_root: Path) -> dict:
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    wav_root = resolve_path(replication_root, cfg["wav_root"])
    split_json = resolve_path(replication_root, cfg["split_json"])
    out_dir = replication_root / "results" / cfg["experiment_id"] / "svm"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, meta_df = load_or_extract_features(cfg, replication_root, wav_root)
    split_obj = json.loads(split_json.read_text())
    svm_cfg = cfg["svm"]
    param_grid = {
        "C": svm_cfg["grid_c"],
        "gamma": svm_cfg["grid_gamma"],
        "kernel": [svm_cfg["kernel"]],
    }

    metrics_rows = []
    prediction_frames = []
    aggregate_cm = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=int)
    started = time.time()

    for fold_name, payload in split_obj.items():
        print(f"\n================ SVM {cfg['experiment_id']} {fold_name} ================")
        train_files = set(payload["train"])
        test_files = set(payload["test"])
        train_df = meta_df[meta_df["file_name"].isin(train_files)].sample(frac=1, random_state=seed)
        test_df = meta_df[meta_df["file_name"].isin(test_files)].sample(frac=1, random_state=seed)

        X_train = X[train_df["feat_idx"].to_numpy()]
        y_train = train_df["label"].to_numpy()
        X_test = X[test_df["feat_idx"].to_numpy()]
        y_test = test_df["label"].to_numpy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        grid_search = GridSearchCV(
            SVC(),
            param_grid=param_grid,
            scoring=make_scorer(f1_score, average="macro"),
            cv=int(svm_cfg["inner_cv"]),
            n_jobs=int(svm_cfg.get("n_jobs", 1)),
        )
        grid_search.fit(X_train_scaled, y_train)
        y_pred = grid_search.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
        aggregate_cm += cm

        print(f"Best params: {grid_search.best_params_}")
        print(f"Accuracy: {acc:.6f}")
        print(f"Macro F1: {f1_macro:.6f}")
        print(classification_report(y_test, y_pred, labels=LABEL_ORDER, target_names=CLASS_NAMES, zero_division=0))

        metrics_rows.append(
            {
                "fold": fold_name,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "accuracy": float(acc),
                "f1_macro": float(f1_macro),
                "best_C": grid_search.best_params_["C"],
                "best_gamma": grid_search.best_params_["gamma"],
                "best_cv_f1_macro": float(grid_search.best_score_),
            }
        )
        pred_df = test_df[["file_name", "class_name", "label"]].copy()
        pred_df["fold"] = fold_name
        pred_df["pred_label"] = y_pred
        pred_df["pred_class"] = pred_df["pred_label"].map(IDX_TO_CLASS)
        prediction_frames.append(pred_df)

        pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                labels=LABEL_ORDER,
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
        ).T.to_csv(out_dir / f"{fold_name}_classification_report.csv")
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / f"{fold_name}_confusion_matrix.csv")

    metrics_df = pd.DataFrame(metrics_rows)
    pred_all = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    pred_all.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(aggregate_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "aggregated_confusion_matrix.csv")
    np.save(out_dir / "aggregated_confusion_matrix.npy", aggregate_cm)

    summary = {
        "experiment_id": cfg["experiment_id"],
        "method": "svm",
        "split_schema": cfg["split_schema"],
        "num_folds": int(len(metrics_df)),
        "accuracy_mean_pct": float(100.0 * metrics_df["accuracy"].mean()),
        "accuracy_std_pct": float(100.0 * metrics_df["accuracy"].std(ddof=0)),
        "f1_mean_pct": float(100.0 * metrics_df["f1_macro"].mean()),
        "f1_std_pct": float(100.0 * metrics_df["f1_macro"].std(ddof=0)),
        "feature_matrix_shape": list(X.shape),
        "feature_granularity": cfg["svm"]["feature_granularity"],
        "runtime_seconds": float(time.time() - started),
        "paper_target": cfg["paper_targets"]["svm"],
        "notes": [
            "Uses the original repository pyAudioAnalysis directory_feature_extraction path.",
            "That original helper long-term-averages each WAV into one feature vector, despite the paper prose describing 1-second segment vectors.",
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
    run_svm(cfg, replication_root)


if __name__ == "__main__":
    main()

