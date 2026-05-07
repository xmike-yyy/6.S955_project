"""Training and evaluation utilities for the MFCC CNN replication.

The functions here mirror the fold-level protocol in `scripts/run_cnn_original.py`
while replacing the original mel-spectrogram package with a local PyTorch model
that consumes MFCC plus cepstral mean and variance normalization (CMVN) inputs.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import CLASSES, MfccSegmentDataset
from .features import MfccConfig, build_mfcc_config
from .model import AlarMfccCNN


@dataclass
class FoldResult:
    """File-level predictions and training metadata for one fold.

    Args:
        y_true_names: True class names after file-level aggregation.
        y_pred_names: Predicted class names after file-level aggregation.
        y_true_indices: True class indices.
        y_pred_indices: Predicted class indices.
        n_train_segments: Number of training segments.
        n_val_segments: Number of validation segments.
        n_test_segments: Number of test segments evaluated.
        n_test_files: Number of test files after aggregation.
        best_epoch: Epoch whose weights were used for testing.
        best_val_loss: Best validation loss, or `None` if no validation split.
        mfcc_config: MFCC extraction configuration used for the fold.
        flat_dim: Flattened feature dimension before Dense(128).
        train_seconds: Wall-clock seconds spent in `fit_model` only.
        model_params: Total trainable and non-trainable model parameters.
    """

    y_true_names: list[str]
    y_pred_names: list[str]
    y_true_indices: list[int]
    y_pred_indices: list[int]
    n_train_segments: int
    n_val_segments: int
    n_test_segments: int
    n_test_files: int
    best_epoch: int
    best_val_loss: float | None
    mfcc_config: MfccConfig
    flat_dim: int
    train_seconds: float
    model_params: int


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for fold reproducibility.

    Args:
        seed: Integer random seed from the YAML config.

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


def choose_device() -> torch.device:
    """Choose the best available PyTorch device.

    Returns:
        CUDA device when available; otherwise CPU.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_train_validation(
    dataset: MfccSegmentDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset, Subset | None]:
    """Split training segments into train and validation subsets.

    Args:
        dataset: Full training segment dataset.
        validation_fraction: Fraction from `cfg["cnn"]` used for validation.
        seed: Random seed for deterministic splitting.

    Returns:
        Tuple of `(train_subset, validation_subset_or_none)`.

    Raises:
        ValueError: If the validation fraction is outside `[0, 1)`.
    """

    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_split_from_training_segments must be in [0, 1).")

    indices = np.arange(len(dataset))
    if validation_fraction == 0.0 or len(indices) < 2:
        return Subset(dataset, indices.tolist()), None

    labels = np.asarray(dataset.labels)
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        val_size = max(1, int(round(len(shuffled) * validation_fraction)))
        val_idx = shuffled[:val_size]
        train_idx = shuffled[val_size:]

    return Subset(dataset, sorted(map(int, train_idx))), Subset(dataset, sorted(map(int, val_idx)))


def make_loader(dataset: Any, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """Create a deterministic, single-process DataLoader.

    Args:
        dataset: PyTorch dataset or subset.
        batch_size: Batch size from `cfg["cnn"]`.
        shuffle: Whether to shuffle examples.
        seed: Random seed for the loader generator.

    Returns:
        Configured DataLoader.

    Raises:
        ValueError: If the batch size is not positive.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one supervised training epoch.

    Args:
        model: MFCC CNN.
        loader: Training DataLoader.
        criterion: Cross-entropy loss.
        optimizer: AdamW or Adam optimizer.
        device: PyTorch device.

    Returns:
        Tuple `(mean_loss, accuracy)`.
    """

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for features, labels, _sources in loader:
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_seen += batch_size

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def evaluate_segments(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate segment-level loss and accuracy.

    Args:
        model: MFCC CNN.
        loader: Validation DataLoader.
        criterion: Cross-entropy loss.
        device: PyTorch device.

    Returns:
        Tuple `(mean_loss, accuracy)`.
    """

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for features, labels, _sources in loader:
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            logits = model(features)
            loss = criterion(logits, labels)
            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += batch_size

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cnn_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[int, float | None, dict[str, torch.Tensor]]:
    """Train a model with AdamW or Adam and optional early stopping.

    Args:
        model: MFCC CNN.
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        cnn_cfg: `cnn:` section augmented with Stage 3 `sample_rate_hz` and
            optional `n_fft` from the YAML config.
        device: PyTorch device.

    Returns:
        Tuple `(best_epoch, best_val_loss, best_state_dict)`.

    Raises:
        ValueError: If the optimizer name or epoch settings are invalid.
    """

    optimizer_name = str(cnn_cfg.get("optimizer", "AdamW"))
    optimizer_key = optimizer_name.lower()
    if optimizer_key not in {"adamw", "adam"}:
        raise ValueError(f"Unsupported optimizer for MFCC CNN: {optimizer_name}")
    epochs = int(cnn_cfg["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    validation_every = int(cnn_cfg.get("validation_every_epochs", 1))
    if validation_every <= 0:
        raise ValueError("validation_every_epochs must be positive.")

    if optimizer_key == "adam":
        adam_kwargs = {"lr": float(cnn_cfg["learning_rate"])}
        if "optimizer_eps" in cnn_cfg:
            adam_kwargs["eps"] = float(cnn_cfg["optimizer_eps"])
        optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cnn_cfg["learning_rate"]),
            weight_decay=float(cnn_cfg["weight_decay"]),
        )
    criterion = nn.CrossEntropyLoss()
    early_stopping = bool(cnn_cfg.get("early_stopping", False)) and val_loader is not None
    patience = int(cnn_cfg.get("early_stopping_patience", 0))
    best_epoch = 0
    best_val_loss: float | None = None
    best_state = copy.deepcopy(model.state_dict())
    validations_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_training_epoch(model, train_loader, criterion, optimizer, device)
        should_validate = val_loader is not None and (epoch % validation_every == 0 or epoch == epochs)
        if should_validate:
            val_loss, val_acc = evaluate_segments(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            if best_val_loss is None or val_loss < best_val_loss - 1e-8:
                best_val_loss = float(val_loss)
                if early_stopping:
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    validations_without_improvement = 0
            else:
                if early_stopping:
                    validations_without_improvement += 1
                    if validations_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch} after {patience} stale validations.")
                        break
            if not early_stopping:
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        elif val_loader is None or not early_stopping:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    return best_epoch, best_val_loss, best_state


def aggregate_test_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    classes: Sequence[str],
) -> tuple[list[int], list[int], list[str]]:
    """Aggregate segment predictions into one label per test file.

    Args:
        model: Trained MFCC CNN.
        test_loader: Test DataLoader over configured-duration MFCC segments.
        device: PyTorch device.
        classes: Class-name sequence.

    Returns:
        Tuple `(true_indices, predicted_indices, source_names)`.

    Raises:
        ValueError: If a source file is seen with inconsistent labels.
    """

    num_classes = len(classes)
    model.eval()
    buckets: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for features, labels, sources in test_loader:
            features = features.to(device=device, dtype=torch.float32)
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            for source, true_label, pred_label, probs in zip(sources, labels_np, predictions, probabilities):
                bucket = buckets.setdefault(
                    str(source),
                    {"true": int(true_label), "preds": [], "probs": []},
                )
                if bucket["true"] != int(true_label):
                    raise ValueError(f"Inconsistent labels for source file {source}.")
                bucket["preds"].append(int(pred_label))
                bucket["probs"].append(np.asarray(probs, dtype=np.float64))

    true_indices = []
    predicted_indices = []
    source_names = []
    for source in sorted(buckets):
        bucket = buckets[source]
        preds = np.asarray(bucket["preds"], dtype=int)
        mean_probs = np.vstack(bucket["probs"]).mean(axis=0)
        counts = np.bincount(preds, minlength=num_classes)
        # Match the mel CNN wrapper's test-time rule: first sort by segment vote
        # count, then use mean posterior probability to break ties.
        predicted = max(range(num_classes), key=lambda idx: (counts[idx], mean_probs[idx]))
        true_indices.append(int(bucket["true"]))
        predicted_indices.append(int(predicted))
        source_names.append(source)

    return true_indices, predicted_indices, source_names


def train_and_evaluate_fold(
    fold_work: Path,
    cnn_cfg: dict[str, Any],
    seed: int,
    model_path: Path | None = None,
) -> FoldResult:
    """Train and evaluate the MFCC CNN for one prepared fold.

    Args:
        fold_work: Fold work directory containing `train/` and `test/`.
        cnn_cfg: `cnn:` section augmented with Stage 3 MFCC/model fields.
        seed: Fold seed.
        model_path: Optional path where the best state dict should be saved.

    Returns:
        FoldResult with file-level predictions and training metadata.

    Raises:
        FileNotFoundError: If prepared fold directories are missing.
        ValueError: If dataset construction or training configuration fails.
    """

    seed_everything(seed)
    config = build_mfcc_config(
        sample_rate_hz=int(cnn_cfg["sample_rate_hz"]),
        segment_seconds=float(cnn_cfg["segment_seconds"]),
        n_fft=int(cnn_cfg.get("n_fft", 4096)),
        n_mfcc=int(cnn_cfg.get("n_mfcc", 40)),
        hop_length=int(cnn_cfg.get("hop_length", 512)),
        center=bool(cnn_cfg.get("center", False)),
        use_cmvn=bool(cnn_cfg.get("use_cmvn", True)),
    )
    train_dataset = MfccSegmentDataset.from_train_dir(Path(fold_work) / "train", config=config)
    test_dataset = MfccSegmentDataset.from_test_dir(Path(fold_work) / "test", config=config)

    # The validation fraction is borrowed directly from the mel CNN YAML block.
    train_subset, val_subset = split_train_validation(
        train_dataset,
        validation_fraction=float(cnn_cfg["validation_split_from_training_segments"]),
        seed=seed,
    )

    batch_size = int(cnn_cfg["batch_size"])
    train_loader = make_loader(train_subset, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = None if val_subset is None else make_loader(val_subset, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = make_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)

    device = choose_device()
    if all(key in cnn_cfg for key in ("conv_kernels", "pool_kernels", "conv_padding")):
        model_kwargs: dict[str, Any] = {
            "conv_kernels": tuple(int(value) for value in cnn_cfg["conv_kernels"]),
            "pool_kernels": tuple(int(value) for value in cnn_cfg["pool_kernels"]),
            "conv_padding": str(cnn_cfg["conv_padding"]),
        }
    else:
        model_kwargs = {"pool_kernel": int(cnn_cfg.get("pool_kernel", 2))}
    if "init_scheme" in cnn_cfg:
        model_kwargs["init_scheme"] = str(cnn_cfg["init_scheme"])
    if "bn_keras_defaults" in cnn_cfg:
        model_kwargs["bn_keras_defaults"] = bool(cnn_cfg["bn_keras_defaults"])
    model = AlarMfccCNN(
        input_shape=(1, config.n_mfcc, config.fixed_frames),
        num_classes=len(CLASSES),
        **model_kwargs,
    ).to(device)
    model_params = int(sum(parameter.numel() for parameter in model.parameters()))
    print(f"Effective MFCC n_fft: {config.n_fft}")
    print(f"Effective MFCC hop_length: {config.hop_length}")
    print(f"Effective MFCC center: {config.center}")
    print(f"Effective MFCC CMVN: {'on' if config.use_cmvn else 'off'}")
    print(f"MFCC input shape: (1, {config.n_mfcc}, {config.fixed_frames})")
    if "conv_kernels" in model_kwargs:
        print(f"MFCC CNN conv_kernels: {list(model_kwargs['conv_kernels'])}")
        print(f"MFCC CNN pool_kernels: {list(model_kwargs['pool_kernels'])}")
        print(f"MFCC CNN conv_padding: {model_kwargs['conv_padding']}")
    else:
        print(f"MFCC CNN pool_kernel: {int(cnn_cfg.get('pool_kernel', 2))}")
    print(f"MFCC CNN flat_dim: {int(model.flat_dim)}")
    print(f"Device for MFCC CNN: {device}")
    print(f"Train segments: {len(train_subset)} | Val segments: {0 if val_subset is None else len(val_subset)}")
    print(f"Test segments: {len(test_dataset)}")

    train_started = time.time()
    best_epoch, best_val_loss, best_state = fit_model(model, train_loader, val_loader, cnn_cfg, device)
    train_seconds = float(time.time() - train_started)
    model.load_state_dict(best_state)
    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)

    y_true_indices, y_pred_indices, _sources = aggregate_test_predictions(model, test_loader, device, CLASSES)
    y_true_names = [CLASSES[idx] for idx in y_true_indices]
    y_pred_names = [CLASSES[idx] for idx in y_pred_indices]

    return FoldResult(
        y_true_names=y_true_names,
        y_pred_names=y_pred_names,
        y_true_indices=y_true_indices,
        y_pred_indices=y_pred_indices,
        n_train_segments=len(train_subset),
        n_val_segments=0 if val_subset is None else len(val_subset),
        n_test_segments=len(test_dataset),
        n_test_files=len(y_true_indices),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        mfcc_config=config,
        flat_dim=int(model.flat_dim),
        train_seconds=train_seconds,
        model_params=model_params,
    )
