"""MediaPipe Holistic landmark extraction used by training and live inference."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


FEATURE_DIM = 258


class LandmarkExtractor:
    """Extract pose and hand landmarks in the project's fixed 258-value format."""

    def __init__(
        self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5
    ) -> None:
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_keypoints(self, results: object) -> np.ndarray:
        """Return pose (33x4), left hand (21x3), and right hand (21x3) landmarks."""

        pose = (
            np.array(
                [[point.x, point.y, point.z, point.visibility] for point in results.pose_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 4, dtype=np.float32)
        )
        left_hand = (
            np.array(
                [[point.x, point.y, point.z] for point in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3, dtype=np.float32)
        )
        right_hand = (
            np.array(
                [[point.x, point.y, point.z] for point in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3, dtype=np.float32)
        )
        keypoints = np.concatenate([pose, left_hand, right_hand])
        if keypoints.shape != (FEATURE_DIM,):
            raise RuntimeError(f"Expected {FEATURE_DIM} extracted features, got {keypoints.shape}.")
        return keypoints

    def draw_landmarks(self, image: np.ndarray, results: object) -> None:
        """Draw the available pose and hand skeletons onto an image."""

        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    def close(self) -> None:
        """Release MediaPipe resources."""

        self.holistic.close()


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    extractor = LandmarkExtractor()
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = extractor.holistic.process(rgb_frame)
            extractor.draw_landmarks(frame, results)
            cv2.imshow("Sign Language Tracker", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()
