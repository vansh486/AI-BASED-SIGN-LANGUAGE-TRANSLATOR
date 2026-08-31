"""Dataset readiness checks for Phase 1 landmark collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.io import atomic_json_save, load_json_object


REQUIRED_CONDITION_FIELDS = (
    "signer_id",
    "session_id",
    "lighting",
    "distance",
    "clothing",
    "speed",
    "camera_angle",
)
VARIED_CONDITION_FIELDS = ("lighting", "distance", "clothing", "speed", "camera_angle")


def audit_dataset(
    processed_dir: str | Path,
    processed_manifest_path: str | Path,
    minimum_samples_per_class: int = 40,
    minimum_sessions_per_class: int = 5,
    minimum_variants_per_condition: int = 2,
) -> dict[str, Any]:
    """Report sample counts, provenance coverage, and observed collection variation."""

    root = Path(processed_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Processed data directory does not exist: {root.resolve()}")
    if minimum_samples_per_class < 1:
        raise ValueError("minimum_samples_per_class must be positive.")
    if minimum_sessions_per_class < 1 or minimum_variants_per_condition < 1:
        raise ValueError("Variation minimums must be positive.")

    manifest = load_json_object(Path(processed_manifest_path), {"records": []})
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Processed manifest records must be a list.")
    by_processed_path = {
        record.get("relative_processed_path"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("relative_processed_path"), str)
    }

    per_class: dict[str, dict[str, Any]] = {}
    missing_provenance: list[str] = []
    condition_values: dict[str, set[str]] = {field: set() for field in REQUIRED_CONDITION_FIELDS}
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(class_dir.glob("*.npy"))
        sample_paths = [path.relative_to(root).as_posix() for path in files]
        documented = 0
        class_condition_values: dict[str, set[str]] = {
            field: set() for field in REQUIRED_CONDITION_FIELDS
        }
        for sample_path in sample_paths:
            record = by_processed_path.get(sample_path)
            collection = record.get("collection") if record else None
            conditions = collection.get("conditions") if isinstance(collection, dict) else None
            if not isinstance(conditions, dict):
                missing_provenance.append(sample_path)
                continue
            documented += 1
            for field in REQUIRED_CONDITION_FIELDS:
                value = conditions.get(field)
                if isinstance(value, str) and value:
                    condition_values[field].add(value)
                    class_condition_values[field].add(value)

        session_requirement_met = (
            len(class_condition_values["session_id"]) >= minimum_sessions_per_class
        )
        condition_requirements_met = all(
            len(class_condition_values[field]) >= minimum_variants_per_condition
            for field in VARIED_CONDITION_FIELDS
        )
        per_class[class_dir.name] = {
            "samples": len(files),
            "meets_minimum": len(files) >= minimum_samples_per_class,
            "documented_samples": documented,
            "variation": {
                field: sorted(values) for field, values in class_condition_values.items()
            },
            "variation_requirements_met": session_requirement_met and condition_requirements_met,
        }

    no_sign_present = "no_sign" in per_class
    compliant = (
        bool(per_class)
        and no_sign_present
        and all(summary["meets_minimum"] for summary in per_class.values())
        and all(summary["variation_requirements_met"] for summary in per_class.values())
        and not missing_provenance
    )
    return {
        "minimum_samples_per_class": minimum_samples_per_class,
        "minimum_sessions_per_class": minimum_sessions_per_class,
        "minimum_variants_per_condition": minimum_variants_per_condition,
        "phase_1_ready": compliant,
        "no_sign_present": no_sign_present,
        "per_class": per_class,
        "missing_provenance": missing_provenance,
        "variation": {field: sorted(values) for field, values in condition_values.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase 1 processed dataset readiness.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--processed-manifest", default="data/metadata/processed_manifest.json")
    parser.add_argument("--minimum-samples-per-class", type=int, default=40)
    parser.add_argument("--minimum-sessions-per-class", type=int, default=5)
    parser.add_argument("--minimum-variants-per-condition", type=int, default=2)
    parser.add_argument("--output", default="data/metadata/dataset_audit.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when Phase 1 requirements fail")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = audit_dataset(
        arguments.processed_dir,
        arguments.processed_manifest,
        arguments.minimum_samples_per_class,
        arguments.minimum_sessions_per_class,
        arguments.minimum_variants_per_condition,
    )
    atomic_json_save(report, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if arguments.strict and not report["phase_1_ready"]:
        raise SystemExit("Phase 1 dataset audit failed.")
