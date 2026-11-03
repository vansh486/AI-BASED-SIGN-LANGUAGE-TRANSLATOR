"""Versioned, atomic checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_SCHEMA_VERSION = 1
SUPPORTED_MODEL_TYPES = {"IsolatedSignClassifier", "STGCNTransformerClassifier"}
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary created by this project's trainer.")

    missing = REQUIRED_CHECKPOINT_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint is incompatible with the inference pipeline; "
            f"missing keys: {', '.join(sorted(missing))}. Retrain the model."
        )
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {checkpoint['schema_version']}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    model_type = checkpoint["model_type"]
    if not isinstance(model_type, str) or model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    if not isinstance(checkpoint["model_config"], dict):
        raise ValueError("Checkpoint model_config must be a dictionary.")
    class_names = checkpoint["class_names"]
    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(isinstance(class_name, str) and class_name for class_name in class_names)
    ):
        raise ValueError("Checkpoint class_names must be a non-empty list.")
    model_config = checkpoint["model_config"]
    if model_config.get("num_classes") != len(class_names):
        raise ValueError("Checkpoint class_names do not match model_config.num_classes.")
    if model_config.get("sequence_length") != checkpoint["sequence_length"]:
        raise ValueError("Checkpoint sequence_length does not match model_config.")
    if model_config.get("input_dim") != checkpoint["feature_dim"]:
        raise ValueError("Checkpoint feature_dim does not match model_config.input_dim.")
    if (
        not isinstance(checkpoint["sequence_length"], int)
        or checkpoint["sequence_length"] <= 0
        or not isinstance(checkpoint["feature_dim"], int)
        or checkpoint["feature_dim"] <= 0
    ):
        raise ValueError("Checkpoint sequence_length and feature_dim must be positive integers.")
    return checkpoint
