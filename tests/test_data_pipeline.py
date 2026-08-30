from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.data.loader import FEATURE_DIM, ISLSequenceDataset, make_stratified_splits, resample_sequence


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


if __name__ == "__main__":
    unittest.main()
