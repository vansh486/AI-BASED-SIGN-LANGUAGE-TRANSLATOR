"""Spatial-temporal graph convolution for MediaPipe Holistic landmarks."""

from __future__ import annotations

import torch
from torch import nn


POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
NODE_COUNT = POSE_LANDMARK_COUNT + 2 * HAND_LANDMARK_COUNT
INPUT_FEATURE_COUNT = 4
FLAT_FEATURE_COUNT = 258

# MediaPipe Pose landmark connections, using its 33-point landmark ordering.
POSE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


def build_landmark_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return directed graph edges and symmetric degree-normalized edge weights."""

    undirected_edges = list(POSE_EDGES)
    for hand_offset in (POSE_LANDMARK_COUNT, POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT):
        for finger_start in (1, 5, 9, 13, 17):
            undirected_edges.append((hand_offset, hand_offset + finger_start))
            undirected_edges.extend(
                (hand_offset + joint, hand_offset + joint + 1)
                for joint in range(finger_start, finger_start + 3)
            )

    # Connect each hand's wrist node to the corresponding pose wrist.
    undirected_edges.extend(
        (
            (15, POSE_LANDMARK_COUNT),
            (16, POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT),
        )
    )

    adjacency = torch.eye(NODE_COUNT, dtype=torch.float32)
    for first, second in undirected_edges:
        adjacency[first, second] = 1.0
        adjacency[second, first] = 1.0

    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inverse_sqrt_degree = degree.rsqrt()
    normalized = inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]
    source, target = torch.nonzero(adjacency, as_tuple=True)
    edge_weights = normalized[source, target]
    return source, target, edge_weights


def landmarks_to_graph(landmarks: torch.Tensor) -> torch.Tensor:
    """Convert [batch, frames, 258] features to [batch, 4, frames, 75] nodes."""

    if landmarks.ndim != 3 or landmarks.size(-1) != FLAT_FEATURE_COUNT:
        raise ValueError(
            f"Expected landmark input [batch, frames, {FLAT_FEATURE_COUNT}], "
            f"received {tuple(landmarks.shape)}."
        )

    batch_size, frame_count, _ = landmarks.shape
    pose = landmarks[..., : POSE_LANDMARK_COUNT * INPUT_FEATURE_COUNT].reshape(
        batch_size, frame_count, POSE_LANDMARK_COUNT, INPUT_FEATURE_COUNT
    )
    hands = landmarks[..., POSE_LANDMARK_COUNT * INPUT_FEATURE_COUNT :].reshape(
        batch_size, frame_count, 2, HAND_LANDMARK_COUNT, 3
    )
    hand_presence = torch.any(hands != 0, dim=-1, keepdim=True).to(landmarks.dtype)
    hand_nodes = torch.cat((hands, hand_presence), dim=-1)
    nodes = torch.cat((pose, hand_nodes.flatten(start_dim=2).reshape(
        batch_size, frame_count, 2 * HAND_LANDMARK_COUNT, INPUT_FEATURE_COUNT
    )), dim=2)
    return nodes.permute(0, 3, 1, 2).contiguous()


class SpatialTemporalGraphBlock(nn.Module):
    """Aggregate neighboring joints, then learn their temporal movement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_weights: torch.Tensor,
        dropout: float,
        temporal_kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.register_buffer("edge_source", source)
        self.register_buffer("edge_target", target)
        self.register_buffer("edge_weights", edge_weights)
        self.edge_importance = nn.Parameter(torch.ones_like(edge_weights))

        self.spatial_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.temporal_convolution = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1),
            padding=(temporal_kernel_size // 2, 0),
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        edge_scale = self.edge_weights * self.edge_importance
        messages = features.index_select(-1, self.edge_source)
        messages = messages * edge_scale.view(1, 1, 1, -1)
        aggregated = torch.zeros_like(features).index_add(-1, self.edge_target, messages)
        projected = self.spatial_projection(aggregated)
        temporal = self.temporal_convolution(projected)
        temporal = self.normalization(temporal)
        return self.activation(self.dropout(temporal) + self.residual(features))


class STGCN(nn.Module):
    """Encode a sequence of 75 body/hand joints with graph and temporal convolutions."""

    def __init__(
        self,
        in_channels: int = INPUT_FEATURE_COUNT,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != INPUT_FEATURE_COUNT:
            raise ValueError(f"ST-GCN expects {INPUT_FEATURE_COUNT} channels per joint.")
        if hidden_dim <= 0 or num_blocks <= 0:
            raise ValueError("hidden_dim and num_blocks must be greater than zero.")

        source, target, edge_weights = build_landmark_graph()
        channels = [in_channels, *([hidden_dim] * num_blocks)]
        self.blocks = nn.Sequential(
            *(
                SpatialTemporalGraphBlock(
                    channels[index],
                    channels[index + 1],
                    source,
                    target,
                    edge_weights,
                    dropout,
                )
                for index in range(num_blocks)
            )
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 4 or nodes.size(1) != INPUT_FEATURE_COUNT or nodes.size(3) != NODE_COUNT:
            raise ValueError(
                f"Expected graph input [batch, {INPUT_FEATURE_COUNT}, frames, {NODE_COUNT}], "
                f"received {tuple(nodes.shape)}."
            )
        features = self.blocks(nodes)
        frame_features = features.mean(dim=-1).transpose(1, 2)
        return self.output_norm(frame_features)


class LandmarkFeatureEncoder(nn.Module):
    """Simple frame-wise encoder retained for the original baseline classifier."""

    def __init__(self, input_dim: int = FLAT_FEATURE_COUNT, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
