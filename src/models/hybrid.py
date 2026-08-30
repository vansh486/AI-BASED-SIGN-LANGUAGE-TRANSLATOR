"""Current production baseline for isolated-sign classification."""

from __future__ import annotations

import torch
from torch import nn

from src.models.stgcn import LandmarkFeatureEncoder
from src.models.transformer import TemporalTransformer


class AttentionPooling(nn.Module):
    """Learn which frames in an isolated sign carry the most useful information."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.score(features).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class IsolatedSignClassifier(nn.Module):
    """Classify one fixed-duration isolated sign from 258-value landmark frames.

    This is a feature encoder plus temporal Transformer, not an ST-GCN. The model name
    reflects what is actually implemented and evaluated in Phase 0.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 258,
        sequence_length: int = 90,
        hidden_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 192,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two classes are required for classification.")

        self.model_config = {
            "num_classes": num_classes,
            "input_dim": input_dim,
            "sequence_length": sequence_length,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feedforward_dim": feedforward_dim,
            "dropout": dropout,
        }
        self.feature_encoder = LandmarkFeatureEncoder(input_dim, hidden_dim, dropout)
        self.temporal_encoder = TemporalTransformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            max_sequence_length=sequence_length,
        )
        self.pooling = AttentionPooling(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, landmarks: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if landmarks.ndim != 3:
            raise ValueError(
                f"Expected [batch, frames, features], received {tuple(landmarks.shape)}."
            )
        if landmarks.size(-1) != self.model_config["input_dim"]:
            raise ValueError(
                f"Expected {self.model_config['input_dim']} features per frame, "
                f"received {landmarks.size(-1)}."
            )
        if landmarks.size(1) != self.model_config["sequence_length"]:
            raise ValueError(
                f"Expected {self.model_config['sequence_length']} frames, "
                f"received {landmarks.size(1)}."
            )

        features = self.feature_encoder(landmarks)
        features = self.temporal_encoder(features, padding_mask)
        pooled = self.pooling(features, padding_mask)
        return self.classifier(pooled)


# Compatibility alias for old scripts. New code should use IsolatedSignClassifier.
HybridModel = IsolatedSignClassifier
