"""Webcam inference for a Phase 0 isolated-sign classifier checkpoint."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from src.data.landmarks import LandmarkExtractor
from src.models.hybrid import IsolatedSignClassifier
from src.training.train import resolve_device
from src.utils.checkpoints import load_checkpoint


NO_SIGN_NAMES = {"background", "no_sign", "no sign", "none"}


class ProbabilitySmoother:
    """Average recent model probabilities to reduce live prediction flicker."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero.")
        self._probabilities: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        self._probabilities.append(probabilities)
        return np.mean(np.stack(self._probabilities), axis=0)


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[IsolatedSignClassifier, dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = IsolatedSignClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def is_no_sign(label: str) -> bool:
    return label.strip().lower().replace("-", "_") in NO_SIGN_NAMES


def run_realtime_inference(args: argparse.Namespace) -> None:
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold must be in the range (0, 1].")
    if args.inference_stride <= 0:
        raise ValueError("inference_stride must be greater than zero.")

    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    class_names: Sequence[str] = checkpoint["class_names"]
    sequence_length = checkpoint["sequence_length"]
    feature_dim = checkpoint["feature_dim"]
    if feature_dim != 258:
        raise ValueError(f"Unsupported landmark feature dimension: {feature_dim}")

    extractor = LandmarkExtractor()
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        extractor.close()
        raise RuntimeError(f"Cannot open camera index {args.camera_index}.")

    frames: deque[np.ndarray] = deque(maxlen=sequence_length)
    smoother = ProbabilitySmoother(args.smoothing_window)
    prediction_text = "Collecting frames..."
    frame_count = 0

    print("Starting real-time isolated-sign recognition. Press 'q' to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera frame capture failed.")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = extractor.holistic.process(rgb_frame)
            keypoints = extractor.extract_keypoints(results)
            frames.append(keypoints)
            frame_count += 1

            if len(frames) == sequence_length and frame_count % args.inference_stride == 0:
                sequence = np.asarray(frames, dtype=np.float32)
                input_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
                with torch.inference_mode():
                    probabilities = torch.softmax(model(input_tensor), dim=1).cpu().numpy()[0]
                smoothed = smoother.update(probabilities)
                predicted_index = int(np.argmax(smoothed))
                confidence = float(smoothed[predicted_index])
                predicted_label = class_names[predicted_index]

                if is_no_sign(predicted_label) or confidence < args.threshold:
                    prediction_text = f"No sign detected ({confidence:.0%})"
                else:
                    prediction_text = f"{predicted_label} ({confidence:.0%})"

            cv2.putText(
                frame,
                f"Prediction: {prediction_text}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Frames: {len(frames)}/{sequence_length}",
                (30, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Sign Language Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run webcam inference for an isolated-sign model.")
    parser.add_argument("--checkpoint", default="models/isolated_sign_classifier.pt")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--inference-stride", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run_realtime_inference(parse_args())
