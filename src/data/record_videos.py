import os
import cv2
import time

DATA_DIR = "data/raw_videos"
CLASSES = [ "help"]
SAMPLES_PER_C = 15
RECORD_SECONDS = 3

os.makedirs(DATA_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)

for class_name in CLASSES:
    class_folder = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_folder, exist_ok=True)
    
    input(f"\n[Ready] Press ENTER to record clips for sign: '{class_name}'")
    
    for i in range(SAMPLES_PER_C):
        print(f"Recording '{class_name}' - Clip {i+1}/{SAMPLES_PER_C} starting in 1s...")
        time.sleep(1.0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(class_folder, f"{class_name}_{i}.mp4")
        fps = 20.0
        # Typical webcam resolution frame dimensions
        writer = cv2.VideoWriter(out_path, fourcc, fps, (640, 480))
        
        start_time = time.time()
        while time.time() - start_time < RECORD_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            writer.write(frame)
            
            cv2.putText(frame, f"Recording {class_name}...", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Video Recorder", frame)
            cv2.waitKey(1)
            
        writer.release()
        print(f"Saved: {out_path}")

cap.release()
cv2.destroyAllWindows()
print("\nAll video clips recorded successfully!")