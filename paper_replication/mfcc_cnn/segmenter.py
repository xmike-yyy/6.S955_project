"""Stage 3 WAV segmentation that preserves MFCC bandwidth.

The original guitar repository's `prepare_dirs` helper always invokes ffmpeg
with `-ar 8000`, which is correct for the mel CNN replication but throws away
high-frequency content before the Stage 3 MFCC front-end can use its configured
sample rate. This module mirrors that helper's fold directory layout and filename
patterns while using the Stage 3 target sample rate, so `dataset.py` and
`train_eval.py` can remain unchanged.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf

from .dataset import CLASSES


CLASS_MAPPING = {idx: name for idx, name in enumerate(CLASSES)}


def get_label(filename: str) -> str:
    """Extract the class index from a guitar dataset WAV filename.

    Args:
        filename: Source WAV filename containing a `class_<i>_` prefix.

    Returns:
        Class index as a string.

    Raises:
        ValueError: If the filename does not contain a class prefix.
    """

    marker = "class_"
    idx = filename.rfind(marker)
    if idx < 0:
        raise ValueError(f"Could not parse class marker from {filename}.")
    idx += len(marker)
    return filename[idx]


def run_ffmpeg(args: Sequence[str], wav_file: str, action: str) -> bool:
    """Run one quiet ffmpeg command and report failures like the original helper.

    Args:
        args: ffmpeg command arguments.
        wav_file: Source WAV filename for diagnostic output.
        action: Short description of the attempted operation.

    Returns:
        `True` when ffmpeg succeeds, otherwise `False`.
    """

    try:
        subprocess.check_call(list(args))
    except subprocess.CalledProcessError as err:
        print(f"An error occurred while {action} for {wav_file}.\nError: {err}")
        return False
    return True


def prepare_dirs_with_librosa_resample(
    input_dir: str,
    train_wavs: Sequence[str],
    test_wavs: Sequence[str],
    output_path: str,
    segment_size: float,
    test_seg: bool,
    sample_rate_hz: int,
) -> None:
    """Create Stage 3 train/test directories using librosa for resampling.

    Args:
        input_dir: Root of the source guitar WAV tree.
        train_wavs: Source basenames assigned to training.
        test_wavs: Source basenames assigned to testing.
        output_path: Fold work directory to populate.
        segment_size: Segment length in seconds.
        test_seg: Whether to segment test files instead of writing trimmed files.
        sample_rate_hz: Stage 3 output sample rate, usually 48000.

    Returns:
        None.

    Raises:
        FileExistsError: If the caller did not create a fresh fold work directory.
        ValueError: If the segment size or sample rate is invalid.
    """

    if segment_size <= 0:
        raise ValueError("segment_size must be positive.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    train_set = set(train_wavs)
    test_set = set(test_wavs)
    train_path = Path(output_path) / "train"
    test_path = Path(output_path) / "test"
    temp_wav_path = Path(output_path) / "temp.wav"
    train_path.mkdir()
    test_path.mkdir()

    train_path_by_label = {}
    for label, guitar_technique in CLASS_MAPPING.items():
        class_path = train_path / guitar_technique
        class_path.mkdir()
        train_path_by_label[label] = class_path

    for subdir, _dirs, files in os.walk(input_dir):
        for wav_file in files:
            if not wav_file.endswith(".wav"):
                continue

            wav_path = Path(subdir) / wav_file
            wav_name = wav_file.removesuffix(".wav")
            if wav_file in train_set:
                label = int(get_label(wav_file))
                out_path = train_path_by_label[label]
                split_role = "train"
            elif wav_file in test_set:
                out_path = test_path
                split_role = "test"
            else:
                print(f"File {wav_file} does not belong to train nor test set. \nSkipping {wav_file}.")
                continue

            audio, _ = librosa.load(str(wav_path), sr=int(sample_rate_hz), mono=True)
            duration = len(audio) / float(sample_rate_hz)
            end = int((duration // segment_size) * segment_size)
            if end <= 0:
                print(f"Skipping {wav_file}; duration is shorter than one full segment.")
                continue

            n_trim = int(round(end * sample_rate_hz))
            trimmed = audio[:n_trim].astype(np.float32)

            if split_role == "train" or test_seg:
                sf.write(str(temp_wav_path), trimmed, sample_rate_hz, subtype="FLOAT")
                segment_ok = run_ffmpeg(
                    [
                        "ffmpeg",
                        "-i",
                        str(temp_wav_path),
                        "-f",
                        "segment",
                        "-segment_time",
                        str(segment_size),
                        "-ac",
                        "1",
                        "-loglevel",
                        "quiet",
                        str(out_path / f"{wav_name}_{segment_size}_%03d.wav"),
                    ],
                    wav_file,
                    "segmenting the Stage 3 WAV",
                )
                temp_wav_path.unlink(missing_ok=True)
                if not segment_ok:
                    continue
            else:
                test_wav_path = out_path / f"{wav_name}_trimmed.wav"
                sf.write(str(test_wav_path), trimmed, sample_rate_hz, subtype="FLOAT")
