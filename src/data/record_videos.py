"""Session-aware, non-destructive collection for isolated-sign videos."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
import time
from pathlib import Path
from typing import Any

import cv2

from src.utils.io import atomic_json_save, load_json_object


DEFAULT_CLASSES = ("hello", "help", "no", "thank_you", "yes", "no_sign")
MANIFEST_SCHEMA_VERSION = 1
MINIMUM_SAMPLES_PER_CLASS = 40


@dataclass(frozen=True)
class CaptureConditions:
    """Human-observed conditions that make collection diversity auditable."""

    signer_id: str
    session_id: str
    lighting: str
    distance: str
    clothing: str
    speed: str
    camera_angle: str
    camera_model: str /N


def next_sample_index(class_dir: Path, class_name: str) -> int:
    pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.mp4$")
    existing_indices = [
        int(match.group(1))
        for path in class_dir.glob(f"{class_name}_*.mp4")
        if (match := pattern.match(path.name))
    ]
    return max(existing_indices, default=-1) + 1


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json_object(
        path, {"schema_version": MANIFEST_SCHEMA_VERSION, "records": []}
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported collection manifest schema in {path}.")
    if not isinstance(manifest.get("records"), list):
        raise ValueError(f"Collection manifest records must be a list: {path}")
    return manifest


def _append_manifest_record(path: Path, record: dict[str, Any]) -> None:
    manifest = _load_manifest(path)
    relative_path = record["relative_video_path"]
    if any(item.get("relative_video_path") == relative_path for item in manifest["records"]):
        raise ValueError(f"Collection manifest already contains {relative_path}.")
    manifest["records"].append(record)
    atomic_json_save(manifest, path)


def record_videos(
    data_dir: str | Path,
    class_names: tuple[str, ...],
    target_samples_per_class: int | None,
    samples_this_session: int | None,
    record_seconds: float,
    camera_index: int,
    conditions: CaptureConditions,
    manifest_path: str | Path,
) -> None:
    """Record each class until it reaches the requested total number of raw clips."""

    if (target_samples_per_class is None) == (samples_this_session is None):
        raise ValueError("Set exactly one of target_samples_per_class or samples_this_session.")
    if target_samples_per_class is not None and target_samples_per_class < MINIMUM_SAMPLES_PER_CLASS:
        raise ValueError(
            f"target_samples_per_class must be at least {MINIMUM_SAMPLES_PER_CLASS} for Phase 1."
        )
    if samples_this_session is not None and samples_this_session <= 0:
        raise ValueError("samples_this_session must be greater than zero.")
    if record_seconds <= 0:
        raise ValueError("record_seconds must be greater than zero.")

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    try:
        for class_name in class_names:
            class_dir = root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            existing_count = len(list(class_dir.glob(f"{class_name}_*.mp4")))
            remaining_count = (
                max(0, target_samples_per_class - existing_count)
                if target_samples_per_class is not None
                else samples_this_session
            )
            if remaining_count == 0:
                print(f"Skipping '{class_name}': already has {existing_count} raw clips.")
                continue

            input(
                f"Press ENTER to record {remaining_count} clip(s) for '{class_name}' "
                f"under session '{conditions.session_id}'. "
            )
            start_index = next_sample_index(class_dir, class_name)

            for offset in range(remaining_count):
                sample_index = start_index + offset
                output_path = class_dir / f"{class_name}_{sample_index}.mp4"
                print(
                    f"Recording '{class_name}' {offset + 1}/{remaining_count} "
                    "in 1 second. Press q to cancel."
                )
                time.sleep(1)
                writer: cv2.VideoWriter | None = None
                frame_count = 0
                frame_size: tuple[int, int] | None = None
                started_at = time.monotonic()
                completed = False
                try:
                    try:
                        while time.monotonic() - started_at < record_seconds:
                            ok, frame = capture.read()
                            if not ok:
                                raise RuntimeError("Camera frame capture failed during recording.")
                            if writer is None:
                                height, width = frame.shape[:2]
                                frame_size = (width, height)
                                writer = cv2.VideoWriter(
                                    str(output_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    30.0,
                                    frame_size,
                                )
                                if not writer.isOpened():
                                    raise RuntimeError(f"Cannot create video file: {output_path}")
                            writer.write(frame)
                            frame_count += 1
                            cv2.putText(
                                frame,
                                f"{class_name}: {offset + 1}/{remaining_count}",
                                (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (0, 0, 255),
                                2,
                            )
                            cv2.imshow("Video Recorder", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                raise KeyboardInterrupt("Recording cancelled by user.")
                        completed = True
                    finally:
                        if writer is not None:
                            writer.release()
                except BaseException:
                    output_path.unlink(missing_ok=True)
                    raise

                if not completed:
                    output_path.unlink(missing_ok=True)
                    continue
                minimum_frames = max(10, int(record_seconds * 10))
                if frame_count < minimum_frames or not output_path.is_file() or output_path.stat().st_size == 0:
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError(f"Recording quality check failed for {output_path}.")

                _append_manifest_record(
                    manifest,
                    {
                        "relative_video_path": output_path.relative_to(root).as_posix(),
                        "class_name": class_name,
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "conditions": asdict(conditions),
                        "camera_index": camera_index,
                        "record_seconds_requested": record_seconds,
                        "frames_recorded": frame_count,
                        "frame_size": list(frame_size) if frame_size else None,
                    },
                )
                print(f"Saved {output_path} ({frame_count} frames)")
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an auditable Phase 1 isolated-sign dataset.")
    parser.add_argument("--data-dir", default="data/raw_videos")
    parser.add_argument("--manifest-path", default="data/metadata/collection_manifest.json")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    collection_size = parser.add_mutually_exclusive_group()
    collection_size.add_argument("--target-samples-per-class", type=int)
    collection_size.add_argument("--samples-this-session", type=int)
    parser.add_argument("--record-seconds", type=float, default=3.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--signer-id", required=True)
    parser.add_argument("--session-id", default=datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--lighting", required=True, help="Example: bright_indoor or low_indoor")
    parser.add_argument("--distance", required=True, help="Example: near, medium, or far")
    parser.add_argument("--clothing", required=True, help="Example: dark_long_sleeve")
    parser.add_argument("--speed", required=True, help="Example: slow, normal, or fast")
    parser.add_argument("--camera-angle", required=True, help="Example: frontal or slight_left")
    arguments = parser.parse_args()
    arguments.classes = tuple(name.strip() for name in arguments.classes.split(",") if name.strip())
    if not arguments.classes:
        parser.error("--classes must contain at least one class name.")
    if len(set(arguments.classes)) != len(arguments.classes):
        parser.error("--classes must not contain duplicate class names.")
    if arguments.target_samples_per_class is None and arguments.samples_this_session is None:
        arguments.target_samples_per_class = 50
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    capture_conditions = CaptureConditions(
        signer_id=arguments.signer_id,
        session_id=arguments.session_id,
        lighting=arguments.lighting,
        distance=arguments.distance,
        clothing=arguments.clothing,
        speed=arguments.speed,
        camera_angle=arguments.camera_angle,
    )
    record_videos(
        arguments.data_dir,
        arguments.classes,
        arguments.target_samples_per_class,
        arguments.samples_this_session,
        arguments.record_seconds,
        arguments.camera_index,
        capture_conditions,
        arguments.manifest_path,
    )
