"""PyTorch datasets for MFCC CNN fold work directories.

The Stage 3 runner uses the same `prepare_dirs` output as
`scripts/run_cnn_original.py`: class-named fixed-duration training WAVs and trimmed
test WAVs under `paper_replication/work/mfcc_cnn_segments_aligned/...`. This module
turns those files into MFCC tensors while preserving the guitar-paper class
order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .features import MfccConfig, count_complete_segments, load_mfcc_feature, segment_num_samples


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

_CLASS_RE = re.compile(r"class_(\d+)_")


@dataclass(frozen=True)
class SegmentRecord:
    """One MFCC segment to read from a fold work directory.

    Args:
        path: WAV file path.
        label: Integer class index in `CLASSES` order.
        source_name: Source test filename used for file-level aggregation.
        segment_index: Zero-based segment number within the source file.
        offset_samples: Offset used when slicing a trimmed test file.
    """

    path: Path
    label: int
    source_name: str
    segment_index: int
    offset_samples: int = 0


def parse_class_index(path: Path) -> int:
    """Parse the `class_<i>_` label prefix from a prepared WAV filename.

    Args:
        path: Segment or trimmed WAV path.

    Returns:
        Integer class index.

    Raises:
        ValueError: If the filename does not contain a valid guitar class index.
    """

    match = _CLASS_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse class_<i>_ prefix from {path}.")
    class_index = int(match.group(1))
    if class_index < 0 or class_index >= len(CLASSES):
        raise ValueError(f"Class index {class_index} is outside the 9 guitar classes.")
    return class_index


def collect_train_records(train_dir: Path) -> list[SegmentRecord]:
    """Collect training segment records from class subdirectories.

    Args:
        train_dir: `prepare_dirs` training directory.

    Returns:
        Sorted list of training segment records.

    Raises:
        FileNotFoundError: If the train directory is missing.
        ValueError: If no training WAV segments are found.
    """

    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    records = []
    for path in sorted(train_dir.rglob("*.wav")):
        label = parse_class_index(path)
        records.append(
            SegmentRecord(
                path=path,
                label=label,
                source_name=path.name,
                segment_index=0,
                offset_samples=0,
            )
        )

    if not records:
        raise ValueError(f"No training WAV segments found under {train_dir}.")
    return records


def collect_test_records(test_dir: Path, config: MfccConfig) -> list[SegmentRecord]:
    """Collect configured-duration test segment records from trimmed test WAV files.

    Args:
        test_dir: `prepare_dirs` test directory.
        config: MFCC extraction configuration.

    Returns:
        Segment records that share `source_name` for later file-level voting.

    Raises:
        FileNotFoundError: If the test directory is missing.
        ValueError: If no test WAV files are found.
    """

    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    records = []
    samples_per_segment = segment_num_samples(config)
    for path in sorted(test_dir.glob("*.wav")):
        label = parse_class_index(path)
        n_segments = count_complete_segments(path, config)
        for segment_index in range(n_segments):
            records.append(
                SegmentRecord(
                    path=path,
                    label=label,
                    source_name=path.name,
                    segment_index=segment_index,
                    offset_samples=segment_index * samples_per_segment,
                )
            )

    if not records:
        raise ValueError(f"No test WAV records found under {test_dir}.")
    return records


class MfccSegmentDataset(Dataset):
    """Dataset that serves cached MFCC tensors to PyTorch.

    Args:
        records: Segment records collected from a fold work directory.
        config: MFCC extraction configuration.
        cache: Whether to cache extracted MFCC tensors in memory.

    Raises:
        ValueError: If `records` is empty.
    """

    def __init__(self, records: Sequence[SegmentRecord], config: MfccConfig, cache: bool = True) -> None:
        """Initialize the dataset.

        Args:
            records: Segment records collected from a fold work directory.
            config: MFCC extraction configuration.
            cache: Whether extracted tensors should be reused across epochs.

        Raises:
            ValueError: If no records are provided.
        """

        if not records:
            raise ValueError("MfccSegmentDataset requires at least one record.")
        self.records = list(records)
        self.config = config
        self.cache = bool(cache)
        self._cache: dict[int, torch.Tensor] = {}

    @classmethod
    def from_train_dir(cls, train_dir: Path, config: MfccConfig) -> "MfccSegmentDataset":
        """Build a dataset from prepared training directories.

        Args:
            train_dir: `prepare_dirs` training directory.
            config: MFCC extraction configuration.

        Returns:
            Dataset over prepared fixed-duration training WAVs.

        Raises:
            FileNotFoundError: If `train_dir` is missing.
            ValueError: If no segment WAVs are found.
        """

        return cls(collect_train_records(train_dir), config=config)

    @classmethod
    def from_test_dir(cls, test_dir: Path, config: MfccConfig) -> "MfccSegmentDataset":
        """Build a dataset from prepared test WAVs.

        Args:
            test_dir: `prepare_dirs` test directory.
            config: MFCC extraction configuration.

        Returns:
            Dataset over configured-duration slices from trimmed test WAVs.

        Raises:
            FileNotFoundError: If `test_dir` is missing.
            ValueError: If no test records are found.
        """

        return cls(collect_test_records(test_dir, config=config), config=config)

    @property
    def labels(self) -> list[int]:
        """Return integer labels for all records.

        Returns:
            Labels in dataset order.
        """

        return [record.label for record in self.records]

    def __len__(self) -> int:
        """Return the number of segment records.

        Returns:
            Dataset length.
        """

        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """Return one MFCC segment, label, and source filename.

        Args:
            index: Dataset index.

        Returns:
            Tuple of `(feature_tensor, label_index, source_name)`.

        Raises:
            IndexError: If the index is outside the dataset.
            FileNotFoundError: If the underlying WAV file is missing.
            ValueError: If MFCC extraction fails.
        """

        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        if self.cache and index in self._cache:
            feature = self._cache[index]
        else:
            record = self.records[index]
            feature = torch.from_numpy(
                load_mfcc_feature(
                    record.path,
                    config=self.config,
                    offset_samples=record.offset_samples,
                )
            )
            if self.cache:
                self._cache[index] = feature

        record = self.records[index]
        return feature, int(record.label), record.source_name
