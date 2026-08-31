from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data.augment import LandmarkAugmentationConfig, LandmarkAugmenter
from src.data.loader import FEATURE_DIM


def make_sequence(frames: int = 12) -> np.ndarray:
    sequence = np.zeros((frames, FEATURE_DIM), dtype=np.float32)
    pose = sequence[:, :132].reshape(frames, 33, 4)
    pose[..., 0] = np.linspace(0.35, 0.65, frames)[:, None]
    pose[..., 1] = 0.5
    pose[..., 2] = 0.1
    pose[..., 3] = 0.9
    pose[:, 11, :2] = (0.4, 0.4)
    pose[:, 12, :2] = (0.6, 0.4)
    right_hand = sequence[:, 195:].reshape(frames, 21, 3)
    right_hand[..., 0] = np.linspace(0.55, 0.75, frames)[:, None]
    right_hand[..., 1] = 0.45
    right_hand[..., 2] = 0.05
    return sequence


class LandmarkAugmenterTests(unittest.TestCase):
    def test_augmentation_is_seeded_and_preserves_missing_left_hand(self) -> None:
        config = LandmarkAugmentationConfig()
        source = make_sequence()
        first = LandmarkAugmenter(config, seed=17)(source)
        second = LandmarkAugmenter(config, seed=17)(source)

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, source.shape)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.array_equal(first[:, 132:195], np.zeros((12, 63), dtype=np.float32)))
        self.assertFalse(np.array_equal(first, source))

    def test_tensor_input_keeps_shape_and_dtype(self) -> None:
        source = torch.from_numpy(make_sequence())
        augmenter = LandmarkAugmenter(LandmarkAugmentationConfig(probability=1.0), seed=9)
        result = augmenter(source)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, source.dtype)


if __name__ == "__main__":
    unittest.main()
