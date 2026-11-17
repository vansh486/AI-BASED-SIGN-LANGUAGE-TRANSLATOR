from __future__ import annotations

import asyncio
import unittest
from threading import Lock
from unittest.mock import Mock, call, patch

import cv2
import numpy as np
import torch

from api.main import SpeechToSignRequest, speech_to_sign
from api.services.recognition import (
    RecognitionEngine,
    configured_checkpoints,
    configured_frontend_origins,
)
from api.services.nlp import gloss_to_english, text_to_glosses
from src.data.landmarks import FEATURE_DIM


class RecognitionEngineTests(unittest.TestCase):
    def test_gloss_to_english_formats_supported_sign_labels(self) -> None:
        self.assertEqual(gloss_to_english("thank_you"), "Thank you")
        self.assertEqual(gloss_to_english("good_morning"), "Good morning")
        self.assertEqual(gloss_to_english("hello"), "Hello")
        self.assertIsNone(gloss_to_english(None))

    def test_text_to_glosses_matches_phrases_and_reports_unknown_words(self) -> None:
        self.assertEqual(text_to_glosses("Hello!"), (["hello"], []))
        self.assertEqual(text_to_glosses("Thank you"), (["thank_you"], []))
        self.assertEqual(
            text_to_glosses("Hello, how are you?"),
            (["hello", "you_plural"], ["how", "are"]),
        )
        self.assertEqual(text_to_glosses(" "), ([], []))

    def test_speech_to_sign_endpoint_returns_glosses_and_motion_limitation(self) -> None:
        result = asyncio.run(speech_to_sign(SpeechToSignRequest(text="Hello")))

        self.assertEqual(result["glosses"], ["hello"])
        self.assertEqual(result["unmatched_words"], [])
        self.assertEqual(result["motions"], [])
        self.assertIn("not connected", result["message"])

    def test_default_checkpoints_select_both_models(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            checkpoints = configured_checkpoints()

        self.assertEqual(checkpoints["custom"].as_posix(), "models/stgcn_transformer.pt")
        self.assertEqual(
            checkpoints["include50"].as_posix(),
            "models/include50_stgcn_transformer.pt",
        )

    def test_default_frontend_origins_include_vite_fallback_port(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            origins = configured_frontend_origins()

        self.assertIn("http://localhost:5174", origins)
        self.assertIn("http://127.0.0.1:5174", origins)

    def test_video_prediction_can_flip_mirrored_selfie_frames(self) -> None:
        engine = RecognitionEngine.__new__(RecognitionEngine)
        engine.predict_landmarks = Mock(return_value={"gloss": "hello"})
        extractor = Mock()
        frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 4
        capture.read.side_effect = [(True, frame), (True, frame), (False, None)]
        extracted_frames: list[np.ndarray] = []

        def extract_frame(image: np.ndarray, _extractor: object) -> np.ndarray:
            extracted_frames.append(image.copy())
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        with (
            patch("api.services.recognition.cv2.VideoCapture", return_value=capture),
            patch("api.services.recognition.LandmarkExtractor", return_value=extractor),
            patch.object(RecognitionEngine, "extract_frame", side_effect=extract_frame),
        ):
            result = engine.predict_video(b"video", ".mp4", mirror_horizontal=True)

        self.assertEqual(result, {"gloss": "hello"})
        self.assertEqual(len(extracted_frames), 2)
        np.testing.assert_array_equal(extracted_frames[0], cv2.flip(frame, 1))
        np.testing.assert_array_equal(extracted_frames[1], cv2.flip(frame, 1))
        extractor.close.assert_called_once_with()

    def test_missing_hand_landmarks_do_not_force_a_class_prediction(self) -> None:
        engine = RecognitionEngine.__new__(RecognitionEngine)

        result = engine.predict_landmarks(np.zeros((20, FEATURE_DIM), dtype=np.float32))

        self.assertFalse(result["recognized"])
        self.assertIsNone(result["gloss"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["top_predictions"], [])
        self.assertIn("No hand landmarks", result["message"])

    def test_prediction_includes_readable_english_for_speech_output(self) -> None:
        engine = RecognitionEngine.__new__(RecognitionEngine)
        engine.sequence_length = 20
        engine.model = Mock(return_value=torch.tensor([[0.1, 0.9]]))
        engine.device = torch.device("cpu")
        engine.class_names = ["hello", "thank_you"]
        engine._prediction_lock = Lock()
        sequence = np.zeros((20, FEATURE_DIM), dtype=np.float32)
        sequence[:, 132] = 0.5

        result = engine.predict_landmarks(sequence)

        self.assertEqual(result["gloss"], "thank_you")
        self.assertEqual(result["english_text"], "Thank you")


if __name__ == "__main__":
    unittest.main()
