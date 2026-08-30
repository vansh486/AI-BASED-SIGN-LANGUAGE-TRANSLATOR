"""Evaluation utilities with no dependency on scikit-learn."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> dict[str, Any]:
    """Calculate accuracy, per-class precision/recall/F1, and confusion matrix."""

    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length.")
    if not targets:
        raise ValueError("Cannot calculate metrics for an empty result set.")

    class_count = len(class_names)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, prediction in zip(targets, predictions, strict=True):
        if not 0 <= target < class_count or not 0 <= prediction < class_count:
            raise ValueError("Targets and predictions must be valid class indices.")
        confusion[target][prediction] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for index, class_name in enumerate(class_names):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(class_count)) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        support = sum(confusion[index])
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)

    correct = sum(confusion[index][index] for index in range(class_count))
    return {
        "accuracy": correct / len(targets),
        "macro_f1": sum(f1_scores) / class_count,
        "samples": len(targets),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate a classifier and return JSON-serializable metrics."""

    model.eval()
    total_loss = 0.0
    sample_count = 0
    targets: list[int] = []
    predictions: list[int] = []

    for features, labels in dataloader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")
    metrics = classification_metrics(targets, predictions, class_names)
    metrics["loss"] = total_loss / sample_count
    return metrics
