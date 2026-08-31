"""Landmark-space augmentation for isolated-sign training sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import get_worker_info

from src.data.loader import FEATURE_DIM


POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_FEATURES = POSE_LANDMARKS * 4
HAND_FEATURES = HAND_LANDMARKS * 3


@dataclass(frozen=True)
class LandmarkAugmentationConfig:
    """Conservative augmentation settings for MediaPipe pose-and-hand coordinates."""

    probability: float = 0.9
    rotation_degrees: float = 8.0
    scale_min: float = 0.92
    scale_max: float = 1.08
    noise_std: float = 0.003
    frame_drop_probability: float = 0.08
    temporal_rate_min: float = 0.85
    temporal_rate_max: float = 1.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in the range [0, 1].")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees cannot be negative.")
        if not 0 < self.scale_min <= self.scale_max:
            raise ValueError("scale_min must be positive and no greater than scale_max.")
        if self.noise_std < 0:
            raise ValueError("noise_std cannot be negative.")
        if not 0.0 <= self.frame_drop_probability < 1.0:
            raise ValueError("frame_drop_probability must be in the range [0, 1).")
        if not 0 < self.temporal_rate_min <= self.temporal_rate_max:
            raise ValueError(
                "temporal_rate_min must be positive and no greater than temporal_rate_max."
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class LandmarkAugmenter:
    """Apply reproducible geometric and temporal jitter to a landmark sequence.

    The augmenter is intended for already-resampled training sequences. It never changes
    pose visibility values and never turns absent hand landmarks (all-zero coordinates)
    into artificial hand detections.
    """

    def __init__(self, config: LandmarkAugmentationConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self._generator: np.random.Generator | None = None
        self._worker_id: int | None = None

    def _rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        if self._generator is None or self._worker_id != worker_id:
            self._generator = np.random.default_rng(self.seed + worker_id)
            self._worker_id = worker_id
        return self._generator

    @staticmethod
    def _validate(sequence: np.ndarray) -> None:
        if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected landmark sequence with shape [frames, {FEATURE_DIM}], "
                f"received {sequence.shape}."
            )
        if len(sequence) < 2:
            raise ValueError("At least two frames are required for augmentation.")
        if not np.isfinite(sequence).all():
            raise ValueError("Landmark sequence contains NaN or infinite values.")

    @staticmethod
    def _blocks(sequence: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        pose = sequence[:, :POSE_FEATURES].reshape(-1, POSE_LANDMARKS, 4)
        left_hand = sequence[:, POSE_FEATURES : POSE_FEATURES + HAND_FEATURES].reshape(
            -1, HAND_LANDMARKS, 3
        )
        right_hand = sequence[:, POSE_FEATURES + HAND_FEATURES :].reshape(
            -1, HAND_LANDMARKS, 3
        )
        pose_mask = pose[..., 3] > 0.0
        left_hand_mask = np.any(left_hand != 0.0, axis=-1)
        right_hand_mask = np.any(right_hand != 0.0, axis=-1)
        return (
            (pose[..., :3], pose_mask),
            (left_hand, left_hand_mask),
            (right_hand, right_hand_mask),
        )

    @staticmethod
    def _anchor(sequence: np.ndarray) -> np.ndarray:
        pose = sequence[:, :POSE_FEATURES].reshape(-1, POSE_LANDMARKS, 4)
        shoulder_coordinates = pose[:, (11, 12), :2]
        shoulder_visible = pose[:, (11, 12), 3] > 0.0
        if shoulder_visible.any():
            return shoulder_coordinates[shoulder_visible].mean(axis=0)

        coordinates = [block[mask, :2] for block, mask in LandmarkAugmenter._blocks(sequence) if mask.any()]
        if coordinates:
            return np.concatenate(coordinates, axis=0).mean(axis=0)
        return np.array([0.5, 0.5], dtype=np.float32)

    def _geometric_jitter(self, sequence: np.ndarray, rng: np.random.Generator) -> None:
        angle = np.deg2rad(rng.uniform(-self.config.rotation_degrees, self.config.rotation_degrees))
        scale = rng.uniform(self.config.scale_min, self.config.scale_max)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32
        )
        anchor = self._anchor(sequence)

        for block, mask in self._blocks(sequence):
            if not mask.any():
                continue
            values = block[mask]
            values[:, :2] = ((values[:, :2] - anchor) @ rotation.T) * scale + anchor
            values[:, 2] *= scale
            if self.config.noise_std:
                values += rng.normal(0.0, self.config.noise_std, size=values.shape).astype(np.float32)
            block[mask] = values

    def _drop_frames(self, sequence: np.ndarray, rng: np.random.Generator) -> None:
        if self.config.frame_drop_probability == 0.0:
            return
        candidate_indices = np.flatnonzero(rng.random(len(sequence)) < self.config.frame_drop_probability)
        # Replacing a dropped frame with its neighboring observation simulates capture
        # skips without changing the fixed model input length or landmark schema.
        for index in candidate_indices:
            source_index = index - 1 if index > 0 else 1
            sequence[index] = sequence[source_index]

    def _time_warp(self, sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.config.temporal_rate_min == self.config.temporal_rate_max == 1.0:
            return sequence
        rate = rng.uniform(self.config.temporal_rate_min, self.config.temporal_rate_max)
        source_positions = np.arange(len(sequence), dtype=np.float32)
        center = (len(sequence) - 1) / 2.0
        warped_positions = np.clip(
            center + (source_positions - center) * rate, 0.0, len(sequence) - 1
        )
        warped = np.empty_like(sequence)
        for feature_index in range(FEATURE_DIM):
            warped[:, feature_index] = np.interp(
                warped_positions, source_positions, sequence[:, feature_index]
            )

        # Visibility and missing-hand state are categorical signals, not coordinates.
        # Restore them with nearest-neighbor sampling after interpolation so a temporal
        # warp cannot invent a partially visible body or hand.
        nearest_indices = np.rint(warped_positions).astype(np.intp)
        source_pose = sequence[:, :POSE_FEATURES].reshape(-1, POSE_LANDMARKS, 4)
        warped_pose = warped[:, :POSE_FEATURES].reshape(-1, POSE_LANDMARKS, 4)
        warped_pose[..., 3] = source_pose[nearest_indices, :, 3]
        warped_pose[warped_pose[..., 3] <= 0.0, :3] = 0.0
        for start in (POSE_FEATURES, POSE_FEATURES + HAND_FEATURES):
            source_hand = sequence[:, start : start + HAND_FEATURES].reshape(-1, HAND_LANDMARKS, 3)
            warped_hand = warped[:, start : start + HAND_FEATURES].reshape(-1, HAND_LANDMARKS, 3)
            hand_present = np.any(source_hand[nearest_indices] != 0.0, axis=-1)
            warped_hand[~hand_present] = 0.0
        return warped

    def __call__(self, sequence: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        is_tensor = isinstance(sequence, torch.Tensor)
        source = sequence.detach().cpu().numpy() if is_tensor else np.asarray(sequence)
        array = np.array(source, dtype=np.float32, copy=True)
        self._validate(array)

        if self._rng().random() >= self.config.probability:
            result = array
        else:
            rng = self._rng()
            self._geometric_jitter(array, rng)
            self._drop_frames(array, rng)
            result = self._time_warp(array, rng)

        if is_tensor:
            return torch.as_tensor(result, dtype=sequence.dtype, device=sequence.device)
        return result
