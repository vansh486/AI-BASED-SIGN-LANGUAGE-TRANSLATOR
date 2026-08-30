import os
import cv2
import numpy as np
from landmarks import LandmarkExtractor

def preprocess_videos():
    input_dir = "data/raw_videos"
    output_dir = "data/processed"
    
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist. Record videos first.")
        return

    extractor = LandmarkExtractor()
    
    for class_name in os.listdir(input_dir):
        class_in_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_in_path):
            continue
            
        class_out_path = os.path.join(output_dir, class_name)
        os.makedirs(class_out_path, exist_ok=True)
        
        for video_file in os.listdir(class_in_path):
            if not video_file.endswith(".mp4"):
                continue
                
            video_path = os.path.join(class_in_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frames_data = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for MediaPipe processing
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = extractor.holistic.process(image)
                
                # Extract the 258 landmark values for the frame
                keypoints = extractor.extract_keypoints(results)
                frames_data.append(keypoints)
                
            cap.release()
            
            if len(frames_data) > 0:
                tensor_array = np.array(frames_data, dtype=np.float32)
                out_name = f"{os.path.splitext(video_file)[0]}.npy"
                out_file_path = os.path.join(class_out_path, out_name)
                np.save(out_file_path, tensor_array)
                print(f"Processed: {class_name}/{video_file} -> Saved shape {tensor_array.shape}")

if __name__ == "__main__":
    preprocess_videos()