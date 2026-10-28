import os
import cv2
from ultralytics import YOLO

# 1. Load your trained model
MODEL_PATH = os.path.join("models", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print("[INFO] Loading trained YOLO11 model...")
model = YOLO(MODEL_PATH)

# 2. Open default webcam (0)
print("[INFO] Starting webcam stream... Press 'q' on the video window to quit.")
cap = cv2.VideoCapture(0)

# Set resolution for smooth performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Make sure your camera is connected and not used by another app.")
    exit()

# 3. Real-time video processing loop
while True:
    success, frame = cap.read()
    if not success:
        print("[WARNING] Failed to grab frame from webcam. Exiting...")
        break

    # Run inference on current frame
    results = model(frame, conf=0.40, verbose=False)

    # Render bounding boxes onto the frame
    annotated_frame = results[0].plot()

    # Display real-time output
    cv2.imshow("Industrial PPE Monitoring - Live Camera", annotated_frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Stopping video stream...")
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed successfully.")