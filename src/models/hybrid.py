"""Isolated-sign classifier architectures."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.stgcn import LandmarkFeatureEncoder, STGCN, landmarks_to_graph
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
    """Original frame-wise landmark encoder and Transformer baseline."""

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


class STGCNTransformerClassifier(nn.Module):
    """Classify an isolated sign with joint-graph convolution and a temporal Transformer."""

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
        graph_num_blocks: int = 3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two classes are required for classification.")
        if input_dim != 258:
            raise ValueError("The MediaPipe Holistic graph model requires 258 input features.")
        if sequence_length <= 0 or hidden_dim <= 0 or num_heads <= 0 or graph_num_blocks <= 0:
            raise ValueError("Sequence length, model dimensions, and graph blocks must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.model_config = {
            "num_classes": num_classes,
            "input_dim": input_dim,
            "sequence_length": sequence_length,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feedforward_dim": feedforward_dim,
            "dropout": dropout,
            "graph_num_blocks": graph_num_blocks,
        }
        self.graph_encoder = STGCN(
            hidden_dim=hidden_dim, dropout=dropout, num_blocks=graph_num_blocks
        )
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

        graph = landmarks_to_graph(landmarks)
        features = self.graph_encoder(graph)
        features = self.temporal_encoder(features, padding_mask)
        pooled = self.pooling(features, padding_mask)
        return self.classifier(pooled)


def build_classifier(model_type: str, model_config: dict[str, Any]) -> nn.Module:
    """Create a supported classifier from its checkpoint architecture name."""

    classifiers: dict[str, type[nn.Module]] = {
        "IsolatedSignClassifier": IsolatedSignClassifier,
        "STGCNTransformerClassifier": STGCNTransformerClassifier,
    }
    try:
        classifier = classifiers[model_type]
    except KeyError as error:
        raise ValueError(f"Unsupported classifier architecture: {model_type}") from error
    return classifier(**model_config)


HybridModel = IsolatedSignClassifier
