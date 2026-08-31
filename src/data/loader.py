"""Dataset and split utilities for isolated-sign landmark sequences."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


FEATURE_DIM = 258
DEFAULT_SEQUENCE_LENGTH = 90


@dataclass(frozen=True)
class SampleRecord:
    """A validated processed sequence and its class label."""

    path: Path
    label: int
    class_name: str


def _validate_sequence(sequence: np.ndarray, source: Path | str) -> None:
    if sequence.ndim != 2:
        raise ValueError(f"{source} must be a 2D array, got shape {sequence.shape}.")
    if sequence.shape[0] == 0:
        raise ValueError(f"{source} contains no frames.")
    if sequence.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{source} has {sequence.shape[1]} features per frame; expected {FEATURE_DIM}."
        )
    if not np.isfinite(sequence).all():
        raise ValueError(f"{source} contains NaN or infinite landmark values.")


def resample_sequence(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    """Resample a sequence across time without discarding its end frames.

    Isolated signs are recorded at slightly different frame rates and durations. Linear
    temporal resampling preserves the full movement while producing the fixed input
    length needed by the classifier.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")

    array = np.asarray(sequence, dtype=np.float32)
    _validate_sequence(array, "sequence")
    if len(array) == sequence_length:
        return np.ascontiguousarray(array)

    source_positions = np.arange(len(array), dtype=np.float32)
    target_positions = np.linspace(0, len(array) - 1, sequence_length, dtype=np.float32)
    resampled = np.empty((sequence_length, array.shape[1]), dtype=np.float32)
    for feature_index in range(array.shape[1]):
        resampled[:, feature_index] = np.interp(
            target_positions, source_positions, array[:, feature_index]
        )
    return resampled


class ISLSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load validated MediaPipe landmark sequences from class-named directories."""

    def __init__(
        self,
        data_dir: str | Path = "data/processed",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Processed data directory does not exist: {self.data_dir.resolve()}"
            )
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")

        self.class_names = sorted(
            directory.name for directory in self.data_dir.iterdir() if directory.is_dir()
        )
        if not self.class_names:
            raise ValueError(f"No class directories found in {self.data_dir.resolve()}.")

        self.label_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.samples: list[SampleRecord] = []
        self.labels: list[int] = []

        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            files = sorted(class_dir.glob("*.npy"))
            if not files:
                raise ValueError(f"Class '{class_name}' has no .npy files in {class_dir}.")

            for file_path in files:
                # Validate the complete dataset before training begins. This avoids a
                # failure halfway through an expensive training run.
                sequence = np.load(file_path, allow_pickle=False)
                _validate_sequence(sequence, file_path)
                label = self.label_to_idx[class_name]
                self.samples.append(SampleRecord(file_path, label, class_name))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.samples[index]
        sequence = np.load(record.path, allow_pickle=False)
        sequence = resample_sequence(sequence, self.sequence_length)
        features = torch.from_numpy(np.ascontiguousarray(sequence))
        label = torch.tensor(record.label, dtype=torch.long)
        return features, label

    def summary(self) -> dict[str, int]:
        """Return sample counts keyed by class name."""

        return {
            class_name: sum(record.class_name == class_name for record in self.samples)
            for class_name in self.class_names
        }


class FeatureTransformDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Apply a feature transform to one split without touching validation or test data."""

    def __init__(
        self,
        dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, label = self.dataset[index]
        transformed = self.transform(features)
        if transformed.shape != features.shape:
            raise ValueError(
                f"Feature transform changed sequence shape from {tuple(features.shape)} "
                f"to {tuple(transformed.shape)}."
            )
        if not torch.isfinite(transformed).all():
            raise ValueError("Feature transform produced NaN or infinite values.")
        return transformed, label


def make_stratified_splits(
    dataset: ISLSequenceDataset,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset[ISLSequenceDataset], Subset[ISLSequenceDataset], Subset[ISLSequenceDataset]]:
    """Create deterministic, class-balanced train/validation/test subsets.

    Each class requires at least three samples so all three splits contain that class.
    This is intentional: a split that silently omits a class gives misleading metrics.
    """

    fractions = train_fraction + validation_fraction + test_fraction
    if not np.isclose(fractions, 1.0):
        raise ValueError("train, validation, and test fractions must add up to 1.0.")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be greater than zero.")

    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset.labels):
        indices_by_label[label].append(index)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for label, indices in sorted(indices_by_label.items()):
        class_name = dataset.class_names[label]
        if len(indices) < 3:
            raise ValueError(
                f"Class '{class_name}' has only {len(indices)} samples. "
                "At least 3 samples per class are required for train/validation/test splits."
            )

        shuffled = list(rng.permutation(indices))
        test_count = max(1, round(len(indices) * test_fraction))
        validation_count = max(1, round(len(indices) * validation_fraction))
        train_count = len(indices) - test_count - validation_count
        if train_count < 1:
            raise ValueError(
                f"Class '{class_name}' cannot populate all splits with the configured fractions."
            )

        test_indices.extend(shuffled[:test_count])
        validation_indices.extend(shuffled[test_count : test_count + validation_count])
        train_indices.extend(shuffled[test_count + validation_count :])

    # Shuffle split order while retaining deterministic membership.
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)
    return (
        Subset(dataset, train_indices),
        Subset(dataset, validation_indices),
        Subset(dataset, test_indices),
    )


# Kept as a compatibility name for existing notebooks and scripts.
CustomISLDataset = ISLSequenceDataset
