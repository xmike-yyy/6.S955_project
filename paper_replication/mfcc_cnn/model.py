"""Alar-style PyTorch CNN for MFCC inputs.

This module ports the documented Alar et al. Sequential CNN pattern into
PyTorch for the guitar replication harness: three Conv-MaxPool-BatchNorm blocks,
Flatten, Dense(128), Dropout(0.5), and a 9-class linear output head.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _keras_glorot_init(module: nn.Module) -> None:
    """Initialize trainable layers with Keras Dense/Conv2D defaults."""

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class AlarMfccCNN(nn.Module):
    """Convolutional neural network for `(1, n_mfcc, T)` MFCC inputs.

    Args:
        input_shape: Shape of one input tensor, excluding batch.
        num_classes: Number of guitar classes. The replication task uses 9.
        dropout_p: Dropout probability after the 128-unit dense layer.
        pool_kernel: Legacy max-pooling kernel size for all three blocks.
        conv_kernels: Optional three convolution kernel sizes.
        pool_kernels: Optional three max-pooling kernel sizes.
        conv_padding: Optional convolution padding mode, `valid` or `same`,
            used with explicit kernel lists.
        init_scheme: Weight initialization scheme. `pytorch_default` leaves
            PyTorch module defaults untouched; `keras_glorot` applies Glorot
            uniform weights and zero biases to Conv2d and Linear layers after
            the full model is built.
        bn_keras_defaults: If true, BatchNorm2d layers use `momentum=0.01` and
            `eps=1e-3` to mirror Keras `BatchNormalization(momentum=0.99,
            epsilon=1e-3)`. Keras momentum 0.99 maps to PyTorch momentum 0.01
            because the frameworks use opposite update conventions.

    Raises:
        ValueError: If the shape, class count, or dropout probability is invalid.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int = 9,
        dropout_p: float = 0.5,
        pool_kernel: int | None = None,
        conv_kernels: tuple[int, int, int] | list[int] | None = None,
        pool_kernels: tuple[int, int, int] | list[int] | None = None,
        conv_padding: str | None = None,
        init_scheme: str = "pytorch_default",
        bn_keras_defaults: bool = False,
    ) -> None:
        """Initialize the Alar-style CNN.

        Args:
            input_shape: Shape of one input tensor, excluding batch.
            num_classes: Number of output logits.
            dropout_p: Dropout probability after Dense(128).
            pool_kernel: Legacy max-pooling kernel size.
            conv_kernels: Optional explicit convolution kernel sizes.
            pool_kernels: Optional explicit max-pooling kernel sizes.
            conv_padding: Optional convolution padding mode.
            init_scheme: `pytorch_default` or `keras_glorot`.
            bn_keras_defaults: Whether BatchNorm2d should use Keras-equivalent
                momentum/epsilon defaults. Keras momentum 0.99 maps to PyTorch
                momentum 0.01.

        Raises:
            ValueError: If model arguments are invalid.
        """

        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (channels, n_mfcc, frames).")
        if input_shape[0] != 1:
            raise ValueError("MFCC CNN expects a single input channel.")
        if num_classes != 9:
            raise ValueError("Stage 3 MFCC CNN must use the 9 guitar classes.")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1).")
        init_scheme = str(init_scheme)
        if init_scheme not in {"pytorch_default", "keras_glorot"}:
            raise ValueError("init_scheme must be 'pytorch_default' or 'keras_glorot'.")
        bn_keras_defaults = bool(bn_keras_defaults)
        use_explicit_kernels = conv_kernels is not None or pool_kernels is not None or conv_padding is not None
        if use_explicit_kernels:
            conv_kernel_list = self._validate_kernel_list(conv_kernels, "conv_kernels")
            pool_kernel_list = self._validate_kernel_list(pool_kernels, "pool_kernels")
            if conv_padding not in {"valid", "same"}:
                raise ValueError("conv_padding must be 'valid' or 'same' when explicit kernels are used.")
            self.feature_extractor = self._make_explicit_feature_extractor(
                conv_kernel_list,
                pool_kernel_list,
                str(conv_padding),
                bn_keras_defaults,
            )
        else:
            legacy_pool_kernel = 2 if pool_kernel is None else int(pool_kernel)
            if legacy_pool_kernel <= 0:
                raise ValueError("pool_kernel must be positive.")
            # Keep baseline 2x2 pooling behavior unchanged. Legacy aligned runs
            # could also pass one pool_kernel for every block.
            pool_padding = 0 if legacy_pool_kernel == 2 else legacy_pool_kernel // 2
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=legacy_pool_kernel, stride=2, padding=pool_padding),
                self._make_batch_norm(32, bn_keras_defaults),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=legacy_pool_kernel, stride=2, padding=pool_padding),
                self._make_batch_norm(64, bn_keras_defaults),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=legacy_pool_kernel, stride=2, padding=pool_padding),
                self._make_batch_norm(128, bn_keras_defaults),
            )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = int(self.feature_extractor(dummy).flatten(start_dim=1).shape[1])

        if flat_dim <= 0:
            raise ValueError("Input shape collapses before the Dense(128) layer.")
        if (
            use_explicit_kernels
            and tuple(conv_kernel_list) == (3, 3, 2)
            and tuple(pool_kernel_list) == (3, 3, 2)
            and conv_padding == "valid"
            and input_shape == (1, 26, 24)
            and flat_dim != 512
        ):
            raise ValueError(f"Notebook-strict aligned model expected flat_dim=512, got {flat_dim}.")
        self.flat_dim = flat_dim

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            # The author Proposed notebook uses Dropout(0.5) here; this is not
            # the 0.3 dropout from the separate Benchmark notebook.
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

        if init_scheme == "keras_glorot":
            self.apply(_keras_glorot_init)

    @staticmethod
    def _validate_kernel_list(values: tuple[int, int, int] | list[int] | None, name: str) -> list[int]:
        """Validate one explicit three-block kernel list.

        Args:
            values: Kernel-size values from the runner config.
            name: Human-readable config key for errors.

        Returns:
            Three positive integer kernel sizes.

        Raises:
            ValueError: If the list is missing, has the wrong length, or has a
                non-positive value.
        """

        if values is None:
            raise ValueError(f"{name} is required when explicit kernels are used.")
        kernels = [int(value) for value in values]
        if len(kernels) != 3:
            raise ValueError(f"{name} must contain exactly three values.")
        if any(value <= 0 for value in kernels):
            raise ValueError(f"{name} values must be positive.")
        return kernels

    @staticmethod
    def _make_batch_norm(channels: int, bn_keras_defaults: bool) -> nn.BatchNorm2d:
        """Build BatchNorm2d with PyTorch or Keras-equivalent defaults."""

        if bn_keras_defaults:
            return nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)
        return nn.BatchNorm2d(channels)

    @staticmethod
    def _make_explicit_feature_extractor(
        conv_kernels: list[int],
        pool_kernels: list[int],
        conv_padding: str,
        bn_keras_defaults: bool,
    ) -> nn.Sequential:
        """Build the notebook-strict explicit-kernel feature extractor.

        Args:
            conv_kernels: Three convolution kernel sizes.
            pool_kernels: Three Keras-style max-pooling kernel sizes.
            conv_padding: Convolution padding mode.
            bn_keras_defaults: Whether BatchNorm2d should use Keras-equivalent
                momentum/epsilon defaults.

        Returns:
            Sequential Conv/ReLU/MaxPool/BatchNorm feature extractor.
        """

        channels = [(1, 32), (32, 64), (64, 128)]
        layers: list[nn.Module] = []
        for (in_channels, out_channels), conv_kernel, pool_kernel in zip(channels, conv_kernels, pool_kernels):
            conv_pad = 0 if conv_padding == "valid" else conv_kernel // 2
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, padding=conv_pad),
                    nn.ReLU(inplace=True),
                    KerasSameMaxPool2d(kernel_size=pool_kernel, stride=2),
                    AlarMfccCNN._make_batch_norm(out_channels, bn_keras_defaults),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of MFCC tensors.

        Args:
            inputs: Tensor with shape `(batch, 1, n_mfcc, T)`.

        Returns:
            Logits with shape `(batch, 9)`.
        """

        features = self.feature_extractor(inputs)
        return self.classifier(features)


class KerasSameMaxPool2d(nn.Module):
    """MaxPool2d with Keras `padding="same"` output sizing.

    Args:
        kernel_size: Square pooling kernel size.
        stride: Pooling stride.
    """

    def __init__(self, kernel_size: int, stride: int = 2) -> None:
        """Initialize Keras-style max pooling.

        Args:
            kernel_size: Square pooling kernel size.
            stride: Pooling stride.

        Raises:
            ValueError: If either argument is not positive.
        """

        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pool a tensor using Keras `same` padding dimensions.

        Args:
            inputs: Tensor with shape `(batch, channels, height, width)`.

        Returns:
            Max-pooled tensor.
        """

        height = int(inputs.shape[-2])
        width = int(inputs.shape[-1])
        out_height = (height + self.stride - 1) // self.stride
        out_width = (width + self.stride - 1) // self.stride
        pad_height = max((out_height - 1) * self.stride + self.kernel_size - height, 0)
        pad_width = max((out_width - 1) * self.stride + self.kernel_size - width, 0)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        if pad_height or pad_width:
            inputs = F.pad(inputs, (pad_left, pad_right, pad_top, pad_bottom), value=float("-inf"))
        return F.max_pool2d(inputs, kernel_size=self.kernel_size, stride=self.stride)
