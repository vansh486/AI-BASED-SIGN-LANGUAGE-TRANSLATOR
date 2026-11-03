"""Import the INCLUDE-50 video archive into this project's landmark format."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

from src.data.landmarks import FEATURE_DIM, LandmarkExtractor
from src.data.loader import validate_sequence
from src.data.preprocess import atomic_numpy_save, extract_video_sequence
from src.utils.io import atomic_json_save, load_json_object


MANIFEST_SCHEMA_VERSION = 1
VALID_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {"relative_path", "clean_label", "official_split"}


def _safe_label(value: str) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not label:
        raise ValueError(f"Dataset contains an empty or invalid label: {value!r}")
    return label


def _load_index(index_path: Path, video_root: Path) -> list[dict[str, str]]:
    if not index_path.is_file():
        raise FileNotFoundError(f"INCLUDE-50 index CSV not found: {index_path.resolve()}")
    if not video_root.is_dir():
        raise FileNotFoundError(f"INCLUDE-50 video directory not found: {video_root.resolve()}")

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                f"INCLUDE-50 CSV must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}."
            )
        rows: list[dict[str, str]] = []
        seen_outputs: set[tuple[str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            relative_path = PurePosixPath(row["relative_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe relative_path on CSV line {line_number}.")
            label = _safe_label(row["clean_label"])
            split = row["official_split"].strip().lower()
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"Invalid official_split {split!r} on CSV line {line_number}."
                )
            video_path = video_root.joinpath(*relative_path.parts).resolve()
            if not video_path.is_relative_to(video_root.resolve()):
                raise ValueError(f"Video path escapes the dataset directory on line {line_number}.")
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video listed on CSV line {line_number} is missing: {video_path.resolve()}"
                )
            output_key = (label, video_path.stem.lower())
            if output_key in seen_outputs:
                raise ValueError(
                    f"Duplicate output name for label {label!r} on CSV line {line_number}."
                )
            seen_outputs.add(output_key)
            rows.append(
                {
                    "relative_video_path": relative_path.as_posix(),
                    "class_name": label,
                    "official_split": split,
                }
            )
    if not rows:
        raise ValueError(f"INCLUDE-50 CSV contains no video records: {index_path.resolve()}")
    return rows


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json_object(
        path, {"schema_version": MANIFEST_SCHEMA_VERSION, "records": []}
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported INCLUDE-50 manifest schema: {path.resolve()}")
    if not isinstance(manifest.get("records"), list):
        raise ValueError(f"INCLUDE-50 manifest records must be a list: {path.resolve()}")
    return manifest


def import_include50(
    index_csv: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    limit: int | None = None,
    manifest_every: int = 25,
) -> int:
    """Extract 258-value MediaPipe sequences, retaining source labels and official splits."""

    input_index = Path(index_csv)
    source_root = Path(video_root).resolve()
    destination = Path(output_dir)
    manifest_file = Path(manifest_path)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero.")
    if manifest_every <= 0:
        raise ValueError("manifest_every must be greater than zero.")

    rows = _load_index(input_index, source_root)
    if limit is not None:
        rows = rows[:limit]
    manifest = _load_manifest(manifest_file)
    records: dict[str, dict[str, Any]] = {}
    for record in manifest["records"]:
        if not isinstance(record, dict):
            raise ValueError("INCLUDE-50 manifest contains a non-object record.")
        processed_path = record.get("relative_processed_path")
        if isinstance(processed_path, str):
            records[processed_path] = record

    extractor = LandmarkExtractor()
    processed_count = 0
    dirty_count = 0
    try:
        for row_number, row in enumerate(rows, start=1):
            relative_video_path = row["relative_video_path"]
            output_path = destination / row["class_name"] / (
                Path(relative_video_path).stem + ".npy"
            )
            relative_output_path = output_path.relative_to(destination).as_posix()
            existing_record = records.get(relative_output_path)

            if output_path.is_file():
                sequence = np.load(output_path, allow_pickle=False)
                validate_sequence(sequence, output_path)
                if (
                    existing_record is not None
                    and existing_record.get("relative_video_path") == relative_video_path
                    and existing_record.get("official_split") == row["official_split"]
                    and existing_record.get("class_name") == row["class_name"]
                ):
                    print(f"[{row_number}/{len(rows)}] Already imported: {relative_video_path}")
                    continue
            else:
                sequence = extract_video_sequence(
                    source_root.joinpath(*PurePosixPath(relative_video_path).parts),
                    extractor,
                )
                if sequence.shape[1] != FEATURE_DIM:
                    raise ValueError(
                        f"Unexpected feature size for {relative_video_path}: {sequence.shape}."
                    )
                atomic_numpy_save(sequence, output_path)
                processed_count += 1

            records[relative_output_path] = {
                "relative_processed_path": relative_output_path,
                "relative_video_path": relative_video_path,
                "class_name": row["class_name"],
                "shape": list(sequence.shape),
                "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_dataset": "INCLUDE-50",
                "official_split": row["official_split"],
            }
            dirty_count += 1
            if dirty_count >= manifest_every:
                atomic_json_save(
                    {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "records": list(records.values()),
                    },
                    manifest_file,
                )
                dirty_count = 0
            print(
                f"[{row_number}/{len(rows)}] Imported {relative_video_path} "
                f"-> {relative_output_path} {tuple(sequence.shape)}"
            )
    finally:
        extractor.close()
        if dirty_count or not manifest_file.exists():
            atomic_json_save(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "records": list(records.values()),
                },
                manifest_file,
            )
    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert INCLUDE-50 videos to project landmark samples."
    )
    parser.add_argument(
        "--index-csv", default="data/include50/include-50_dataset.csv"
    )
    parser.add_argument(
        "--video-root",
        default="data/include50/include-50_dataset/content/include_50_dataset",
    )
    parser.add_argument("--output-dir", default="data/include50/processed")
    parser.add_argument("--manifest", default="data/include50/processed_manifest.json")
    parser.add_argument(
        "--limit",
        type=int,
        help="Import only the first N indexed videos for a smoke test; do not train on a partial import.",
    )
    parser.add_argument("--manifest-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    count = import_include50(
        arguments.index_csv,
        arguments.video_root,
        arguments.output_dir,
        arguments.manifest,
        arguments.limit,
        arguments.manifest_every,
    )
    print(f"Imported {count} video(s) from the INCLUDE-50 dataset.")
