"""Model loading and isolated-sign prediction services for the API."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import cv2
import numpy as np
import torch

from src.data.landmarks import FEATURE_DIM, LandmarkExtractor
from src.data.loader import resample_sequence, validate_sequence
from src.inference.predict import load_model


MAX_JPEG_BYTES = 1_000_000
MAX_IMAGE_PIXELS = 2_073_600
MAX_VIDEO_FRAMES = 500
MIN_LIVE_FRAMES = 20
LIVE_WINDOW_FRAMES = 20


class InvalidMediaError(ValueError):
    """Raised when an uploaded image or video cannot be processed."""


class MediaTooLargeError(ValueError):
    """Raised when an uploaded image or video exceeds API limits."""


class RecognitionEngine:
    """Load one trained checkpoint and reuse it for all API predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path = "models/stgcn_transformer.pt",
        requested_device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if requested_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")

        self.model, self.checkpoint = load_model(self.checkpoint_path, self.device)
        if self.checkpoint["feature_dim"] != FEATURE_DIM:
            raise ValueError(
                f"Checkpoint has {self.checkpoint['feature_dim']} features; "
                f"this API requires {FEATURE_DIM}."
            )
        self.class_names: list[str] = self.checkpoint["class_names"]
        self.sequence_length: int = self.checkpoint["sequence_length"]
        self._prediction_lock = threading.Lock()

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": "ST-GCN + Transformer",
            "model_type": self.checkpoint["model_type"],
            "architecture": "ST-GCN + temporal Transformer",
            "checkpoint": self.checkpoint_path.name,
            "device": str(self.device),
            "sequence_length": self.sequence_length,
            "feature_dim": FEATURE_DIM,
            "supported_labels": self.class_names,
            "features": {
                "live_sign_recognition": True,
                "video_prediction": True,
                "speech_to_sign": False,
                "english_translation": False,
                "avatar_motion": False,
            },
        }

    def predict_landmarks(self, sequence: np.ndarray) -> dict[str, Any]:
        """Predict the top classes from one variable-length landmark sequence."""

        validate_sequence(sequence, "landmark sequence")
        if not np.any(sequence[:, 132:] != 0.0):
            return {
                "recognized": False,
                "gloss": None,
                "confidence": 0.0,
                "top_predictions": [],
                "message": (
                    "No hand landmarks were detected. Move closer, improve the lighting, "
                    "and keep your signing hand visible."
                ),
            }

        model_input = resample_sequence(sequence, self.sequence_length)
        input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(self.device)
        with self._prediction_lock, torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_indices = np.argsort(probabilities)[::-1][: min(3, len(self.class_names))]
        top_predictions = [
            {
                "gloss": self.class_names[int(index)],
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ]
        top = top_predictions[0]
        return {
            "recognized": True,
            "gloss": top["gloss"],
            "confidence": top["confidence"],
            "top_predictions": top_predictions,
        }

    @staticmethod
    def extract_frame(
        frame_bgr: np.ndarray, extractor: LandmarkExtractor
    ) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = extractor.holistic.process(frame_rgb)
        return extractor.extract_keypoints(results)

    @staticmethod
    def decode_jpeg_frame(data: bytes) -> np.ndarray:
        if not data:
            raise InvalidMediaError("The image frame is empty.")
        if len(data) > MAX_JPEG_BYTES:
            raise MediaTooLargeError("Image frames must be no larger than 1 MB.")

        width, height = _jpeg_dimensions(data)
        if width * height > MAX_IMAGE_PIXELS:
            raise MediaTooLargeError("Image frames must not exceed 1920×1080 pixels.")
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise InvalidMediaError("The WebSocket message is not a valid JPEG image.")
        return frame

    def predict_video(self, content: bytes, suffix: str) -> dict[str, Any]:
        """Extract a bounded video to landmarks and predict its isolated sign."""

        if suffix.lower() not in {".mp4", ".mov", ".webm"}:
            raise InvalidMediaError("Upload an MP4, MOV, or WebM video.")
        temporary_path: str | None = None
        extractor = LandmarkExtractor()
        capture: cv2.VideoCapture | None = None
        frames: list[np.ndarray] = []
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix.lower(), prefix="sign-video-", delete=False
            ) as temporary_file:
                temporary_path = temporary_file.name
                temporary_file.write(content)

            capture = cv2.VideoCapture(temporary_path)
            if not capture.isOpened():
                raise InvalidMediaError("The uploaded video could not be opened.")
            metadata_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (
                metadata_width > 0
                and metadata_height > 0
                and metadata_width * metadata_height > MAX_IMAGE_PIXELS
            ):
                raise MediaTooLargeError(
                    "Video frames must not exceed 1920×1080 pixels."
                )

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if len(frames) >= MAX_VIDEO_FRAMES:
                    raise MediaTooLargeError(
                        f"Videos must contain no more than {MAX_VIDEO_FRAMES} frames."
                    )
                height, width = frame.shape[:2]
                if height * width > MAX_IMAGE_PIXELS:
                    raise MediaTooLargeError(
                        "Video frames must not exceed 1920×1080 pixels."
                    )
                frames.append(self.extract_frame(frame, extractor))
        finally:
            if capture is not None:
                capture.release()
            extractor.close()
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

        if len(frames) < 2:
            raise InvalidMediaError("The video does not contain enough readable frames.")
        return self.predict_landmarks(np.asarray(frames, dtype=np.float32))


def configured_checkpoints() -> dict[str, Path]:
    legacy_checkpoint = os.environ.get("SIGN_MODEL_CHECKPOINT")
    return {
        "custom": Path(
            os.environ.get(
                "SIGN_CUSTOM_MODEL_CHECKPOINT",
                legacy_checkpoint or "models/stgcn_transformer.pt",
            )
        ),
        "include50": Path(
            os.environ.get(
                "SIGN_INCLUDE50_MODEL_CHECKPOINT",
                legacy_checkpoint or "models/include50_stgcn_transformer.pt",
            )
        ),
    }


def configured_device() -> str:
    return os.environ.get("SIGN_MODEL_DEVICE", "auto")


def configured_frontend_origins() -> list[str]:
    configured = os.environ.get(
        "SIGN_FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise InvalidMediaError("The WebSocket message is not a valid JPEG image.")

    start_of_frame_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    position = 2
    while position < len(data):
        if data[position] != 0xFF:
            raise InvalidMediaError("Malformed JPEG frame header.")
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            break
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            raise InvalidMediaError("Malformed JPEG frame header.")
        if marker in start_of_frame_markers:
            if segment_length < 7:
                raise InvalidMediaError("Malformed JPEG dimensions.")
            height = int.from_bytes(data[position + 3 : position + 5], "big")
            width = int.from_bytes(data[position + 5 : position + 7], "big")
            if width <= 0 or height <= 0:
                raise InvalidMediaError("JPEG dimensions must be positive.")
            return width, height
        if marker == 0xDA:
            break
        position += segment_length
    raise InvalidMediaError("JPEG dimensions could not be read.")


def new_live_state() -> deque[np.ndarray]:
    """Create isolated rolling state for a single WebSocket connection."""

    return deque(maxlen=LIVE_WINDOW_FRAMES)
