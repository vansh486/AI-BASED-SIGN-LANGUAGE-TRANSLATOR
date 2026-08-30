"""Feature encoder for flattened MediaPipe landmarks.

This module deliberately does not claim to be an ST-GCN. A true spatial-temporal graph
convolution needs an explicit joint graph and graph-convolution layers, neither of which
is present in the current isolated-sign baseline.
"""

from __future__ import annotations

import torch
from torch import nn


class LandmarkFeatureEncoder(nn.Module):
    """Project one flattened landmark frame into the model embedding space."""

    def __init__(self, input_dim: int = 258, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class STGCN(LandmarkFeatureEncoder):
    """Compatibility wrapper for older imports; it is not a graph-convolution model."""

    def __init__(
        self, in_channels: int = 258, num_classes: int = 96, dropout: float = 0.1
    ) -> None:
        super().__init__(input_dim=in_channels, hidden_dim=num_classes, dropout=dropout)
