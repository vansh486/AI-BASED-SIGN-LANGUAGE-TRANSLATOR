"""Versioned, atomic checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_CHECKPOINT_KEYS = {
    "schema_version",
    "model_type",
    "model_config",
    "state_dict",
    "class_names",
    "sequence_length",
    "feature_dim",
}


def atomic_torch_save(payload: dict[str, Any], destination: str | Path) -> None:
    """Write a checkpoint atomically so interrupted saves never corrupt the model."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".tmp", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            torch.save(payload, temporary_file)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load and validate a checkpoint created by this project."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path.resolve()}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary created by the Phase 0 trainer.")

    missing = REQUIRED_CHECKPOINT_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint is incompatible with the Phase 0 inference pipeline; "
            f"missing keys: {', '.join(sorted(missing))}. Retrain the model."
        )
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {checkpoint['schema_version']}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    if checkpoint["model_type"] != "IsolatedSignClassifier":
        raise ValueError(f"Unsupported model type: {checkpoint['model_type']}")
    if checkpoint["model_config"]["num_classes"] != len(checkpoint["class_names"]):
        raise ValueError("Checkpoint class_names do not match model_config.num_classes.")
    return checkpoint
