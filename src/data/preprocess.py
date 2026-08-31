"""Convert raw videos into atomic landmark tensors with provenance metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from src.data.landmarks import LandmarkExtractor
from src.utils.io import atomic_json_save, load_json_object


PROCESSED_MANIFEST_SCHEMA_VERSION = 1


def atomic_numpy_save(sequence: np.ndarray, output_path: Path) -> None:
    """Persist a tensor atomically to avoid partially-written training samples."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        suffix=".npy", prefix=f".{output_path.stem}.", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.save(temporary_path, sequence, allow_pickle=False)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _collection_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json_object(manifest_path, {"records": []})
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Collection manifest records must be a list: {manifest_path}")
    return {
        record["relative_video_path"]: record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("relative_video_path"), str)
    }


def _load_processed_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json_object(
        path, {"schema_version": PROCESSED_MANIFEST_SCHEMA_VERSION, "records": []}
    )
    if manifest.get("schema_version") != PROCESSED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported processed manifest schema in {path}.")
    if not isinstance(manifest.get("records"), list):
        raise ValueError(f"Processed manifest records must be a list: {path}")
    return manifest


def _upsert_processed_record(path: Path, record: dict[str, Any]) -> None:
    manifest = _load_processed_manifest(path)
    records = [
        item
        for item in manifest["records"]
        if item.get("relative_processed_path") != record["relative_processed_path"]
    ]
    records.append(record)
    manifest["records"] = records
    atomic_json_save(manifest, path)


def _extract_video_sequence(video_path: Path, extractor: LandmarkExtractor) -> np.ndarray:
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
    return np.asarray(frames, dtype=np.float32)


def preprocess_videos(
    input_dir: str | Path,
    output_dir: str | Path,
    collection_manifest_path: str | Path,
    processed_manifest_path: str | Path,
    overwrite: bool = False,
) -> int:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Raw video directory does not exist: {input_path.resolve()}")

    collection_records = _collection_index(Path(collection_manifest_path))
    processed_manifest = Path(processed_manifest_path)
    processed_count = 0
    extractor = LandmarkExtractor()
    try:
        for class_dir in sorted(path for path in input_path.iterdir() if path.is_dir()):
            class_output = output_path / class_dir.name
            for video_path in sorted(class_dir.glob("*.mp4")):
                output_file = class_output / f"{video_path.stem}.npy"
                if output_file.exists() and not overwrite:
                    sequence = np.load(output_file, allow_pickle=False)
                    print(f"Skipping existing file: {output_file}")
                else:
                    sequence = _extract_video_sequence(video_path, extractor)
                    atomic_numpy_save(sequence, output_file)
                    processed_count += 1
                    print(f"Processed {video_path} -> {output_file} {tuple(sequence.shape)}")

                relative_video_path = video_path.relative_to(input_path).as_posix()
                _upsert_processed_record(
                    processed_manifest,
                    {
                        "relative_processed_path": output_file.relative_to(output_path).as_posix(),
                        "relative_video_path": relative_video_path,
                        "class_name": class_dir.name,
                        "shape": list(sequence.shape),
                        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "collection": collection_records.get(relative_video_path),
                    },
                )
    finally:
        extractor.close()
    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract landmarks and preserve collection provenance.")
    parser.add_argument("--input-dir", default="data/raw_videos")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--collection-manifest", default="data/metadata/collection_manifest.json")
    parser.add_argument("--processed-manifest", default="data/metadata/processed_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    count = preprocess_videos(
        arguments.input_dir,
        arguments.output_dir,
        arguments.collection_manifest,
        arguments.processed_manifest,
        arguments.overwrite,
    )
    print(f"Processed {count} video(s).")
