"""MFCC feature extraction for the Stage 3 replication CNN.

This module converts the `paper_replication` harness' fixed-duration WAV segments into
Mel-frequency cepstral coefficient (MFCC) tensors. Stage 3 uses its own
48,000 Hz MFCC sample rate while the mel CNN and SVM paths keep their original
sample-rate settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class MfccConfig:
    """Configuration for locked MFCC extraction.

    Args:
        sample_rate_hz: Target sample rate for every segment.
        segment_seconds: Segment duration passed to `prepare_dirs`.
        n_mfcc: Number of MFCC coefficients per frame.
        n_fft: Short-time Fourier transform (STFT) window length in samples.
        hop_length: Hop length between STFT frames in samples.
        center: Whether librosa pads the waveform before STFT framing.
        use_cmvn: Whether to apply per-clip cepstral mean and variance
            normalization.
        fixed_frames: Fixed MFCC time width `T` used by the CNN.
        cmvn_epsilon: Numerical stabilizer for cepstral mean and variance
            normalization (CMVN).
    """

    sample_rate_hz: int
    segment_seconds: float
    n_mfcc: int = 40
    n_fft: int = 4096
    hop_length: int = 512
    center: bool = False
    use_cmvn: bool = True
    fixed_frames: int = 86
    cmvn_epsilon: float = 1e-8


def expected_num_frames(
    sample_rate_hz: int,
    segment_seconds: float,
    n_fft: int,
    hop_length: int,
    center: bool,
) -> int:
    """Compute the fixed MFCC time width for the configured framing mode.

    With `center=False`, librosa frames a 1-second segment as
    `T = 1 + (n_samples - n_fft) // hop_length` when the segment has at
    least one full STFT window. At 48,000 Hz with `hop_length=512`, the sweep
    values give `n_fft=4096 -> T=86` (about 85 ms windows),
    `n_fft=2048 -> T=90` (about 43 ms), and `n_fft=1024 -> T=92`
    (about 21 ms). This `T` is the reference width used throughout the fold.
    With `center=True`, librosa pads before framing, giving
    `T = 1 + n_samples // hop_length`; for the notebook-aligned 48,000 Hz,
    0.5-second, `hop_length=1024` setup, `T = 1 + 24000 // 1024 = 24`.
    With `n_mfcc=26`, the aligned tensor is `(1, 26, 24)`. The
    notebook-strict aligned model then applies valid conv kernels `(3, 3, 2)`
    and Keras-same stride-2 pool kernels `(3, 3, 2)`, reducing the feature map
    to 128 channels over a `2 x 2` grid; `model.py` derives the exact flattened
    dimension with a dummy forward pass.

    Args:
        sample_rate_hz: Segment sample rate in hertz.
        segment_seconds: Segment duration in seconds.
        n_fft: STFT window length in samples.
        hop_length: STFT hop length in samples.
        center: Whether librosa center padding is enabled.

    Returns:
        Number of MFCC frames expected for each fixed-length segment.

    Raises:
        ValueError: If any timing or framing argument is not positive.
    """

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive.")
    if n_fft <= 0:
        raise ValueError("n_fft must be positive.")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive.")

    n_samples = int(round(sample_rate_hz * segment_seconds))
    if n_samples <= 0:
        raise ValueError("segment duration produced zero samples.")

    if center:
        return 1 + n_samples // hop_length
    if n_samples < n_fft:
        return 1
    return 1 + (n_samples - n_fft) // hop_length


def build_mfcc_config(
    sample_rate_hz: int,
    segment_seconds: float,
    n_fft: int = 4096,
    n_mfcc: int = 40,
    hop_length: int = 512,
    center: bool = False,
    use_cmvn: bool = True,
) -> MfccConfig:
    """Create the locked MFCC configuration used by the runner.

    Args:
        sample_rate_hz: Target audio sample rate from `cfg["mfcc_cnn"]`.
        segment_seconds: Segment length from `cfg["cnn"]`.
        n_fft: STFT window length from `cfg["mfcc_cnn"].get("n_fft", 4096)`.
        n_mfcc: Number of MFCC coefficients.
        hop_length: STFT hop length in samples.
        center: Whether librosa center padding is enabled.
        use_cmvn: Whether to apply per-clip CMVN.

    Returns:
        MFCC configuration with the fixed time width derived from STFT framing.

    Raises:
        ValueError: If the sample rate or segment length is invalid.
    """

    fixed_frames = expected_num_frames(
        sample_rate_hz=sample_rate_hz,
        segment_seconds=segment_seconds,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        center=bool(center),
    )
    return MfccConfig(
        sample_rate_hz=int(sample_rate_hz),
        segment_seconds=float(segment_seconds),
        n_mfcc=int(n_mfcc),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        center=bool(center),
        use_cmvn=bool(use_cmvn),
        fixed_frames=int(fixed_frames),
    )


def segment_num_samples(config: MfccConfig) -> int:
    """Return the waveform sample count for one configured segment.

    Args:
        config: MFCC extraction configuration.

    Returns:
        Number of samples in one segment.
    """

    return int(round(config.sample_rate_hz * config.segment_seconds))


def load_resampled_audio(path: Path, sample_rate_hz: int) -> np.ndarray:
    """Load a mono WAV file and resample it for guitar-CNN parity.

    Args:
        path: WAV file path.
        sample_rate_hz: Target sample rate, usually 48,000 from `mfcc_cnn`.

    Returns:
        Mono waveform as `float32`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If librosa returns an empty waveform.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing WAV file: {path}")

    # The Alar notebook used its own audio front-end. Stage 3 now resamples to
    # 48,000 Hz so the tested n_fft values retain enough frames for the three
    # pooling blocks, without touching the mel CNN or SVM sample rates.
    audio, _ = librosa.load(path, sr=int(sample_rate_hz), mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Empty WAV file after loading: {path}")
    return audio


def slice_fixed_segment(audio: np.ndarray, offset_samples: int, config: MfccConfig) -> np.ndarray:
    """Slice one fixed-width segment from a waveform.

    Args:
        audio: Mono waveform.
        offset_samples: Starting sample index of the segment.
        config: MFCC extraction configuration.

    Returns:
        One waveform segment, right-padded with zeros only when needed.

    Raises:
        ValueError: If `audio` is not one-dimensional or the offset is invalid.
    """

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}.")
    if offset_samples < 0:
        raise ValueError("offset_samples must be non-negative.")

    length = segment_num_samples(config)
    chunk = audio[offset_samples : offset_samples + length]
    if chunk.size == 0:
        raise ValueError("Segment offset is beyond the end of the waveform.")

    # `prepare_dirs` writes fixed-duration chunks. This tiny right-pad handles
    # rare decoder or rounding edge cases before MFCC framing.
    if chunk.size < length:
        chunk = np.pad(chunk, (0, length - chunk.size), mode="constant")
    elif chunk.size > length:
        chunk = chunk[:length]
    return np.asarray(chunk, dtype=np.float32)


def mfcc_from_segment(segment: np.ndarray, config: MfccConfig) -> np.ndarray:
    """Convert one waveform segment into a CNN-ready MFCC tensor.

    Args:
        segment: Mono waveform segment.
        config: MFCC extraction configuration.

    Returns:
        Tensor-shaped NumPy array with shape `(1, n_mfcc, T)`.

    Raises:
        ValueError: If the input segment is not one-dimensional or produces an
            unexpected number of MFCC frames.
    """

    if segment.ndim != 1:
        raise ValueError(f"Expected one-dimensional segment, got {segment.shape}.")

    if segment.size < config.n_fft:
        segment = np.pad(segment, (0, config.n_fft - segment.size), mode="constant")

    # `center=False` avoids librosa reflection padding at segment edges for the
    # baseline. The notebook-aligned variant uses center=True; at 48,000 Hz,
    # 0.5 seconds, and hop=1024, librosa deterministically gives T=24, so the
    # padding branch below should not trigger in normal operation.
    mfcc = librosa.feature.mfcc(
        y=np.asarray(segment, dtype=np.float32),
        sr=config.sample_rate_hz,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        center=config.center,
    ).astype(np.float32)

    if config.use_cmvn:
        # CMVN means cepstral mean and variance normalization. We normalize each
        # coefficient over time within the clip before any padding so padded
        # zeros do not influence the clip statistics.
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True)
        mfcc = (mfcc - mean) / (std + config.cmvn_epsilon)

    if mfcc.shape[1] > config.fixed_frames:
        raise ValueError(
            f"MFCC produced {mfcc.shape[1]} frames, expected at most {config.fixed_frames}."
        )

    if mfcc.shape[1] < config.fixed_frames:
        pad_width = config.fixed_frames - mfcc.shape[1]
        # Minimal right zero-padding fixes rare off-by-one widths while keeping
        # the frequency axis and extracted coefficients unchanged.
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

    return mfcc[np.newaxis, :, :].astype(np.float32)


def load_mfcc_feature(path: Path, config: MfccConfig, offset_samples: int = 0) -> np.ndarray:
    """Load a WAV path and return one MFCC feature segment.

    Args:
        path: WAV path from a fold work directory.
        config: MFCC extraction configuration.
        offset_samples: Segment start offset in samples for trimmed test files.

    Returns:
        CNN input feature with shape `(1, n_mfcc, T)`.

    Raises:
        FileNotFoundError: If the WAV path is missing.
        ValueError: If audio loading, slicing, or MFCC framing fails.
    """

    audio = load_resampled_audio(Path(path), config.sample_rate_hz)
    segment = slice_fixed_segment(audio, offset_samples, config)
    return mfcc_from_segment(segment, config)


def count_complete_segments(path: Path, config: MfccConfig) -> int:
    """Count configured-duration MFCC chunks inside a prepared test WAV.

    Args:
        path: Trimmed test WAV produced by `prepare_dirs`.
        config: MFCC extraction configuration.

    Returns:
        Number of complete segment windows. A short non-empty file returns one
        padded segment so the evaluator can still produce a file-level label.

    Raises:
        FileNotFoundError: If the WAV path is missing.
        ValueError: If the WAV file is empty.
    """

    audio = load_resampled_audio(Path(path), config.sample_rate_hz)
    samples_per_segment = segment_num_samples(config)
    full_segments = int(audio.size // samples_per_segment)
    if full_segments == 0:
        return 1
    return full_segments
