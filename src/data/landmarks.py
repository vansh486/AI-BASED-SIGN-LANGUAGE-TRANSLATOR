import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self):
        # Initialize MediaPipe Holistic model
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, results):
        """Extracts and flattens coordinates into a single numpy array."""
        # Extract pose (33 landmarks x 4 values: x, y, z, visibility)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        
        # Extract hands (21 landmarks x 3 values: x, y, z)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, lh, rh])

    def draw_landmarks(self, image, results):
        """Draws the skeletal overlay on the image."""
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

# --- Webcam Tester ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe processing
        image, results = frame, extractor.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Draw the landmarks on the frame
        extractor.draw_landmarks(image, results)
        
        # Show the video feed
        cv2.imshow('Sign Language Tracker', image)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()