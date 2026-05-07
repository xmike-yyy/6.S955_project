#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

import yaml


CLASS_NAMES = {
    "0": "alternate picking",
    "1": "legato",
    "2": "tapping",
    "3": "sweep picking",
    "4": "vibrato",
    "5": "hammer on",
    "6": "pull off",
    "7": "slide",
    "8": "bend",
}


def resolve_path(replication_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (replication_root / path).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_wavs(wav_root: Path) -> dict:
    wav_paths = sorted(wav_root.glob("*/*.wav"))
    class_counts = Counter(p.parent.name for p in wav_paths)
    sample_rates = Counter()
    channels = Counter()
    total_segments = Counter()
    durations = {}

    for path in wav_paths:
        with contextlib.closing(wave.open(str(path), "rb")) as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            dur = wf.getnframes() / float(sr)
        sample_rates[str(sr)] += 1
        channels[str(ch)] += 1
        total_segments[path.parent.name] += int(dur // 1)
        durations.setdefault(path.parent.name, []).append(dur)

    duration_summary = {
        cls: {
            "mean_seconds": sum(vals) / len(vals),
            "min_seconds": min(vals),
            "max_seconds": max(vals),
        }
        for cls, vals in durations.items()
    }
    return {
        "wav_file_count": len(wav_paths),
        "class_counts": dict(sorted(class_counts.items())),
        "sample_rates": dict(sorted(sample_rates.items())),
        "channels": dict(sorted(channels.items())),
        "whole_second_segments_by_class": dict(sorted(total_segments.items())),
        "duration_summary_by_class": duration_summary,
    }


def inspect_split(split_json: Path) -> dict:
    split_obj = json.loads(split_json.read_text())
    fold_rows = []
    all_test_counts = Counter()
    for fold, payload in split_obj.items():
        train = set(payload["train"])
        test = set(payload["test"])
        all_test_counts.update(test)
        fold_rows.append(
            {
                "fold": fold,
                "train_files": len(train),
                "test_files": len(test),
                "train_test_overlap": len(train & test),
            }
        )
    return {
        "split_sha256": sha256(split_json),
        "num_folds": len(split_obj),
        "folds": fold_rows,
        "unique_test_files": len(all_test_counts),
        "repeated_test_files": sum(1 for count in all_test_counts.values() if count > 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    replication_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(replication_root, args.config)
    cfg = yaml.safe_load(config_path.read_text())

    dataset_root = resolve_path(replication_root, cfg["dataset_root"])
    wav_root = resolve_path(replication_root, cfg["wav_root"])
    split_json = resolve_path(replication_root, cfg["split_json"])
    paper_pdf = resolve_path(replication_root, cfg["paper_pdf"])
    archive_path = replication_root.parent / "guitar_style_dataset-v1.0.0.zip"

    required = {
        "dataset_root": dataset_root,
        "wav_root": wav_root,
        "split_json": split_json,
        "paper_pdf": paper_pdf,
    }
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")

    result = {
        "experiment_id": cfg["experiment_id"],
        "config_path": str(config_path),
        "paper_pdf": str(paper_pdf),
        "paper_pdf_sha256": sha256(paper_pdf),
        "dataset_root": str(dataset_root),
        "wav_root": str(wav_root),
        "split_json": str(split_json),
        "split": inspect_split(split_json),
        "wav_inventory": inspect_wavs(wav_root),
    }
    if archive_path.exists():
        result["archive_path"] = str(archive_path)
        result["archive_md5"] = md5(archive_path)

    out_dir = replication_root / "results" / cfg["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "input_validation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved input validation to {out_path}")


if __name__ == "__main__":
    main()

