"""Convert raw sign videos into validated MediaPipe landmark sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.landmarks import LandmarkExtractor


def preprocess_videos(input_dir: str | Path, output_dir: str | Path, overwrite: bool = False) -> int:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Raw video directory does not exist: {input_path.resolve()}")

    processed_count = 0
    extractor = LandmarkExtractor()
    try:
        for class_dir in sorted(path for path in input_path.iterdir() if path.is_dir()):
            class_output = output_path / class_dir.name
            class_output.mkdir(parents=True, exist_ok=True)

            for video_path in sorted(class_dir.glob("*.mp4")):
                output_file = class_output / f"{video_path.stem}.npy"
                if output_file.exists() and not overwrite:
                    print(f"Skipping existing file: {output_file}")
                    continue

                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"Cannot open video: {video_path}")
                frames: list[np.ndarray] = []
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(extractor.extract_keypoints(extractor.holistic.process(rgb_frame)))
                finally:
                    capture.release()

                if not frames:
                    raise ValueError(f"No readable frames found in {video_path}")
                sequence = np.asarray(frames, dtype=np.float32)
                np.save(output_file, sequence, allow_pickle=False)
                processed_count += 1
                print(f"Processed {video_path} -> {output_file} {tuple(sequence.shape)}")
    finally:
        extractor.close()
    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from raw sign videos.")
    parser.add_argument("--input-dir", default="data/raw_videos")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    count = preprocess_videos(arguments.input_dir, arguments.output_dir, arguments.overwrite)
    print(f"Processed {count} video(s).")
