import cv2
import torch
import numpy as np
from src.models.hybrid import HybridModel
from src.data.loader import CustomISLDataset
from src.data.landmarks import LandmarkExtractor

def run_realtime_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset to retrieve class name mappings
    dataset = CustomISLDataset(data_dir="data/processed")
    class_names = dataset.class_names
    
    if len(class_names) == 0:
        print("No class names found in processed folder.")
        return
        
    # Load trained model weights
    model = HybridModel(num_classes=len(class_names))
    model.load_state_dict(torch.load("models/isl_hybrid_model.pth", map_location=device))
    model.to(device)
    model.eval()

    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)
    
    max_frames = 60
    frames_buffer = []

    print("Starting real-time sign language recognition... Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = extractor.holistic.process(image)
        keypoints = extractor.extract_keypoints(results)
        
        frames_buffer.append(keypoints)
        if len(frames_buffer) > max_frames:
            frames_buffer.pop(0)
            
        prediction_text = "Collecting frames..."
        
        if len(frames_buffer) == max_frames:
            input_tensor = torch.tensor(np.array(frames_buffer), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probabilities, dim=1)
                
                predicted_label = class_names[pred_idx.item()]
                confidence_score = conf.item()
                
                # Lowered threshold to see raw guesses
                if confidence_score > 0.15:
                    prediction_text = f"{predicted_label} ({confidence_score:.2f})"
                else:
                    prediction_text = f"Low: {predicted_label} ({confidence_score:.2f})"

        cv2.putText(frame, f"Prediction: {prediction_text}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Sign Language Real-Time Inference", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_inference()