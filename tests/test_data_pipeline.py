from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.loader import (
    FEATURE_DIM,
    ISLSequenceDataset,
    make_grouped_splits,
    make_stratified_splits,
    load_sample_groups,
    resample_sequence,
)


class DataPipelineTests(unittest.TestCase):
    def test_resampling_preserves_sequence_endpoints(self) -> None:
        sequence = np.zeros((4, FEATURE_DIM), dtype=np.float32)
        sequence[:, 0] = [0.0, 1.0, 2.0, 3.0]
        resampled = resample_sequence(sequence, 7)

        self.assertEqual(resampled.shape, (7, FEATURE_DIM))
        self.assertEqual(resampled[0, 0], 0.0)
        self.assertEqual(resampled[-1, 0], 3.0)

    def test_processed_dataset_and_stratified_splits_are_complete(self) -> None:
        dataset = ISLSequenceDataset(Path("data/processed"), sequence_length=90)
        train_set, validation_set, test_set = make_stratified_splits(dataset, seed=7)
        features, label = dataset[0]

        self.assertEqual(features.shape, (90, FEATURE_DIM))
        self.assertIn(label.item(), range(len(dataset.class_names)))
        self.assertEqual(len(train_set) + len(validation_set) + len(test_set), len(dataset))
        self.assertGreaterEqual(len(validation_set), len(dataset.class_names))
        self.assertGreaterEqual(len(test_set), len(dataset.class_names))

    def test_grouped_splits_keep_groups_disjoint_and_all_classes_present(self) -> None:
        dataset = ISLSequenceDataset(Path("data/processed"), sequence_length=12)
        seen_per_class: dict[str, int] = {}
        groups: list[str] = []
        for sample in dataset.samples:
            class_name = sample.class_name
            group_index = seen_per_class.get(class_name, 0)
            groups.append(f"session_{group_index % 5}")
            seen_per_class[class_name] = group_index + 1

        splits = make_grouped_splits(dataset, groups, seed=11)
        split_groups = [
            {groups[index] for index in subset.indices}
            for subset in splits
        ]
        self.assertEqual(len(split_groups[0] & split_groups[1]), 0)
        self.assertEqual(len(split_groups[0] & split_groups[2]), 0)
        self.assertEqual(len(split_groups[1] & split_groups[2]), 0)
        for subset in splits:
            self.assertEqual(
                {dataset.labels[index] for index in subset.indices},
                set(range(len(dataset.class_names))),
            )

    def test_grouped_splits_require_each_class_in_three_groups(self) -> None:
        dataset = ISLSequenceDataset(Path("data/processed"), sequence_length=12)
        groups = [f"group_{index // 15}" for index in range(len(dataset))]
        with self.assertRaisesRegex(ValueError, "Insufficient groups"):
            make_grouped_splits(dataset, groups)

    def test_sample_groups_load_from_processed_manifest(self) -> None:
        dataset = ISLSequenceDataset(Path("data/processed"), sequence_length=12)
        class_sample_numbers: dict[str, int] = {}
        records = []
        expected_groups = []
        for sample in dataset.samples:
            sample_number = class_sample_numbers.get(sample.class_name, 0)
            session_id = f"session_{sample_number % 5}"
            expected_groups.append(session_id)
            class_sample_numbers[sample.class_name] = sample_number + 1
            records.append(
                {
                    "relative_processed_path": sample.path.relative_to(
                        dataset.data_dir
                    ).as_posix(),
                    "collection": {"conditions": {"session_id": session_id}},
                }
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "processed_manifest.json"
            manifest_path.write_text(
                json.dumps({"records": records}), encoding="utf-8"
            )
            actual_groups = load_sample_groups(dataset, "session", manifest_path)

        self.assertEqual(actual_groups, expected_groups)


if __name__ == "__main__":
    unittest.main()
