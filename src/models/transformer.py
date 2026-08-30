"""Transformer components used by the isolated-sign classifier."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    """Add deterministic temporal positions to landmark-frame embeddings."""

    def __init__(self, d_model: int, max_sequence_length: int) -> None:
        super().__init__()
        position = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_sequence_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {features.size(1)} exceeds configured maximum "
                f"{self.encoding.size(1)}."
            )
        return features + self.encoding[:, : features.size(1)].to(features.dtype)


class TemporalTransformer(nn.Module):
    """Encode the temporal order of a fixed-length landmark sequence."""

    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
        max_sequence_length: int = 90,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.position_encoding = SinusoidalPositionEncoding(d_model, max_sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.position_encoding(features)
        encoded = self.transformer_encoder(features, src_key_padding_mask=padding_mask)
        return self.output_norm(encoded)


# Compatibility alias for the original module name.
CustomTransformer = TemporalTransformer
