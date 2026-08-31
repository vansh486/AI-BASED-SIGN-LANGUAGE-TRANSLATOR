"""Reproducible training entry point for the isolated-sign baseline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.data.augment import LandmarkAugmentationConfig, LandmarkAugmenter
from src.data.loader import (
    DEFAULT_SEQUENCE_LENGTH,
    FEATURE_DIM,
    FeatureTransformDataset,
    ISLSequenceDataset,
    make_stratified_splits,
)
from src.models.hybrid import IsolatedSignClassifier
from src.training.evaluate import evaluate_model
from src.utils.checkpoints import CHECKPOINT_SCHEMA_VERSION, atomic_torch_save, load_checkpoint
from src.utils.io import atomic_json_save


def set_global_seed(seed: int) -> None:
    """Make data splits and training behavior repeatable on the same platform."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0

    for features, labels in dataloader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Cannot train on an empty dataloader.")
    return total_loss / sample_count


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be greater than zero.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative.")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label_smoothing must be in the range [0, 1).")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in the range [0, 1).")
    if args.patience <= 0 or args.min_delta < 0 or args.num_workers < 0:
        raise ValueError("patience must be positive; min_delta and num_workers cannot be negative.")

    augmentation_config = LandmarkAugmentationConfig(
        probability=0.0 if args.disable_augmentation else args.augmentation_probability,
        rotation_degrees=args.rotation_degrees,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        noise_std=args.noise_std,
        frame_drop_probability=args.frame_drop_probability,
        temporal_rate_min=args.temporal_rate_min,
        temporal_rate_max=args.temporal_rate_max,
    )

    set_global_seed(args.seed)
    device = resolve_device(args.device)
    dataset = ISLSequenceDataset(args.data_dir, sequence_length=args.sequence_length)
    train_set, validation_set, test_set = make_stratified_splits(
        dataset,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    if len(dataset.class_names) < 2:
        raise ValueError("At least two class directories are required for training.")

    augmented_train_set = FeatureTransformDataset(
        train_set, LandmarkAugmenter(augmentation_config, seed=args.seed)
    )
    train_loader = make_dataloader(augmented_train_set, args.batch_size, True, args.num_workers, device)
    validation_loader = make_dataloader(validation_set, args.batch_size, False, args.num_workers, device)
    test_loader = make_dataloader(test_set, args.batch_size, False, args.num_workers, device)

    model = IsolatedSignClassifier(
        num_classes=len(dataset.class_names),
        input_dim=FEATURE_DIM,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 3)
    )

    checkpoint_path = Path(args.checkpoint)
    print(
        f"Training {len(dataset.class_names)} classes on {device}: "
        f"train={len(train_set)}, validation={len(validation_set)}, test={len(test_set)}"
    )
    print(f"Class distribution: {dataset.summary()}")
    print(f"Training augmentation: {augmentation_config.to_dict()}")

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        training_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = evaluate_model(
            model, validation_loader, criterion, device, dataset.class_names
        )
        scheduler.step(validation_metrics["loss"])
        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": training_loss,
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={training_loss:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_accuracy={validation_metrics['accuracy']:.3f} | "
            f"val_macro_f1={validation_metrics['macro_f1']:.3f} | lr={learning_rate:.2e}"
        )

        if validation_metrics["loss"] < best_validation_loss - args.min_delta:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "model_type": "IsolatedSignClassifier",
                    "model_config": model.model_config,
                    "state_dict": model.state_dict(),
                    "class_names": dataset.class_names,
                    "sequence_length": args.sequence_length,
                    "feature_dim": FEATURE_DIM,
                    "augmentation": augmentation_config.to_dict(),
                    "training": {
                        "seed": args.seed,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_validation_loss,
                        "data_split": {
                            "train": len(train_set),
                            "validation": len(validation_set),
                            "test": len(test_set),
                        },
                        "class_distribution": dataset.summary(),
                    },
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best checkpoint was epoch {best_epoch}.")
                break

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_model(model, test_loader, criterion, device, dataset.class_names)
    report = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "validation": checkpoint["validation_metrics"],
        "test": test_metrics,
        "history": history,
    }
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    atomic_json_save(report, metrics_path)
    print(
        f"Best model: {checkpoint_path} (epoch {best_epoch})\n"
        f"Test accuracy={test_metrics['accuracy']:.3f}, "
        f"test macro F1={test_metrics['macro_f1']:.3f}\n"
        f"Metrics report: {metrics_path}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the isolated-sign classifier.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="models/isolated_sign_classifier.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=192)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--augmentation-probability", type=float, default=0.9)
    parser.add_argument("--rotation-degrees", type=float, default=8.0)
    parser.add_argument("--scale-min", type=float, default=0.92)
    parser.add_argument("--scale-max", type=float, default=1.08)
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--frame-drop-probability", type=float, default=0.08)
    parser.add_argument("--temporal-rate-min", type=float, default=0.85)
    parser.add_argument("--temporal-rate-max", type=float, default=1.15)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
