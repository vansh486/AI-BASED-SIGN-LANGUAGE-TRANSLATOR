"""Interactive, non-destructive video collection for isolated-sign classes."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2


DEFAULT_CLASSES = ("hello", "help", "no", "thank_you", "yes", "no_sign")


def next_sample_index(class_dir: Path, class_name: str) -> int:
    pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.mp4$")
    existing_indices = [
        int(match.group(1))
        for path in class_dir.glob(f"{class_name}_*.mp4")
        if (match := pattern.match(path.name))
    ]
    return max(existing_indices, default=-1) + 1


def record_videos(
    data_dir: str | Path,
    class_names: tuple[str, ...],
    samples_per_class: int,
    record_seconds: float,
    camera_index: int,
) -> None:
    if samples_per_class <= 0 or record_seconds <= 0:
        raise ValueError("samples_per_class and record_seconds must be greater than zero.")

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    try:
        for class_name in class_names:
            class_dir = root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            start_index = next_sample_index(class_dir, class_name)
            input(f"Press ENTER to record {samples_per_class} clip(s) for '{class_name}'. ")

            for offset in range(samples_per_class):
                sample_index = start_index + offset
                print(f"Recording '{class_name}' clip {offset + 1}/{samples_per_class} in 1 second...")
                time.sleep(1)
                output_path = class_dir / f"{class_name}_{sample_index}.mp4"
                writer: cv2.VideoWriter | None = None
                started_at = time.monotonic()
                try:
                    while time.monotonic() - started_at < record_seconds:
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError("Camera frame capture failed during recording.")
                        if writer is None:
                            height, width = frame.shape[:2]
                            writer = cv2.VideoWriter(
                                str(output_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                30.0,
                                (width, height),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"Cannot create video file: {output_path}")
                        writer.write(frame)
                        cv2.putText(
                            frame,
                            f"Recording {class_name}: {offset + 1}/{samples_per_class}",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow("Video Recorder", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise KeyboardInterrupt("Recording cancelled by user.")
                finally:
                    if writer is not None:
                        writer.release()

                if not output_path.is_file() or output_path.stat().st_size == 0:
                    raise RuntimeError(f"No video frames were recorded for {output_path}.")
                print(f"Saved {output_path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record isolated-sign training videos.")
    parser.add_argument("--data-dir", default="data/raw_videos")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--record-seconds", type=float, default=3.0)
    parser.add_argument("--camera-index", type=int, default=0)
    arguments = parser.parse_args()
    arguments.classes = tuple(name.strip() for name in arguments.classes.split(",") if name.strip())
    if not arguments.classes:
        parser.error("--classes must contain at least one class name.")
    if len(set(arguments.classes)) != len(arguments.classes):
        parser.error("--classes must not contain duplicate class names.")
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    record_videos(
        arguments.data_dir,
        arguments.classes,
        arguments.samples_per_class,
        arguments.record_seconds,
        arguments.camera_index,
    )
