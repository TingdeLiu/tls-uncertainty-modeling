"""Regression PointNet model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class RePN(nn.Module):
    """Multi-scale point-neighbourhood regression network.

    Parameters mirror the historical notebook while removing its hard-coded
    fully connected input size.
    """

    def __init__(
        self,
        samples_per_scale: Sequence[int] = (16, 32, 128),
        mlp_channels: Sequence[Sequence[int]] = (
            (32, 32, 64),
            (64, 64, 128),
            (64, 96, 128),
        ),
        *,
        physical_feature_count: int = 4,
        hidden_channels: Sequence[int] = (216, 216, 110, 110),
    ):
        super().__init__()
        if len(samples_per_scale) != len(mlp_channels):
            raise ValueError("one MLP definition is required for each scale")
        if not samples_per_scale or any(value < 1 for value in samples_per_scale):
            raise ValueError("samples_per_scale must contain positive integers")
        if physical_feature_count < 0:
            raise ValueError("physical_feature_count must be non-negative")
        if any(not channels for channels in mlp_channels):
            raise ValueError("each scale must contain at least one MLP channel")

        self.samples_per_scale = tuple(samples_per_scale)
        self.physical_feature_count = physical_feature_count
        self.scale_blocks = nn.ModuleList()

        for channels in mlp_channels:
            layers: list[nn.Module] = []
            input_channels = 3
            for output_channels in channels:
                layers.extend(
                    (
                        nn.Conv1d(input_channels, output_channels, kernel_size=1),
                        nn.BatchNorm1d(output_channels),
                        nn.ReLU(),
                    )
                )
                input_channels = output_channels
            self.scale_blocks.append(nn.Sequential(*layers))

        regressor_input = sum(channels[-1] for channels in mlp_channels)
        regressor_input += physical_feature_count
        regressor_layers: list[nn.Module] = []
        for output_channels in hidden_channels:
            regressor_layers.extend(
                (nn.Linear(regressor_input, output_channels), nn.ReLU())
            )
            regressor_input = output_channels
        regressor_layers.append(nn.Linear(regressor_input, 1))
        self.regressor = nn.Sequential(*regressor_layers)

    def forward(
        self,
        neighbourhoods: torch.Tensor,
        physical_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one range residual per neighbourhood."""
        if neighbourhoods.ndim != 3 or neighbourhoods.shape[-1] != 3:
            raise ValueError("neighbourhoods must have shape (B, K, 3)")
        if physical_features.ndim != 2:
            raise ValueError("physical_features must have shape (B, F)")
        if neighbourhoods.shape[0] != physical_features.shape[0]:
            raise ValueError("batch dimensions must match")
        if physical_features.shape[1] != self.physical_feature_count:
            raise ValueError(
                f"expected {self.physical_feature_count} physical features"
            )
        if neighbourhoods.shape[1] < max(self.samples_per_scale):
            raise ValueError(
                "neighbourhood size must be at least the largest configured scale"
            )

        xyz_channels = neighbourhoods.transpose(1, 2)
        pooled_features = []
        for sample_count, block in zip(
            self.samples_per_scale, self.scale_blocks, strict=True
        ):
            scale_features = block(xyz_channels[:, :, :sample_count])
            pooled_features.append(F.adaptive_max_pool1d(scale_features, 1).squeeze(-1))

        combined = torch.cat((*pooled_features, physical_features), dim=1)
        return self.regressor(combined)
