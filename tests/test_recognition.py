from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from api.services.recognition import RecognitionEngine, configured_checkpoints
from src.data.landmarks import FEATURE_DIM


class RecognitionEngineTests(unittest.TestCase):
    def test_default_checkpoints_select_both_models(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            checkpoints = configured_checkpoints()

        self.assertEqual(checkpoints["custom"].as_posix(), "models/stgcn_transformer.pt")
        self.assertEqual(
            checkpoints["include50"].as_posix(),
            "models/include50_stgcn_transformer.pt",
        )

    def test_missing_hand_landmarks_do_not_force_a_class_prediction(self) -> None:
        engine = RecognitionEngine.__new__(RecognitionEngine)

        result = engine.predict_landmarks(np.zeros((20, FEATURE_DIM), dtype=np.float32))

        self.assertFalse(result["recognized"])
        self.assertIsNone(result["gloss"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["top_predictions"], [])
        self.assertIn("No hand landmarks", result["message"])


if __name__ == "__main__":
    unittest.main()
