import os
import cv2
import time
import winsound  # Built-in Windows library for audio beeps
from datetime import datetime
from ultralytics import YOLO

# 1. Paths & Directories
MODEL_PATH = os.path.join("models", "best.pt")
VIOLATIONS_DIR = "violations"
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# 2. Load Model
print("[INFO] Loading trained YOLO11 model...")
model = YOLO(MODEL_PATH)

# Target Classes:
CLASS_PERSON = 6
REQUIRED_PPE = {
    4: "Helmet",
    7: "Vest",
    2: "Glass",   # Goggles / Glasses
    3: "Glove"
}

def is_inside(ppe_box, person_box):
    """Checks if the center of a PPE item falls inside the Person bounding box."""
    px1, py1, px2, py2 = person_box
    ex1, ey1, ex2, ey2 = ppe_box
    center_x = (ex1 + ex2) / 2
    center_y = (ey1 + ey2) / 2
    return (px1 <= center_x <= px2) and (py1 <= center_y <= py2)

# 3. Alert Cooldown Management
last_alert_time = 0
ALERT_COOLDOWN = 4.0  # Seconds between alarm beeps and saved snapshots

# 4. Start Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Safety Monitoring Active.")
print("[INFO] If a violation occurs, an alarm will sound and an evidence image will be saved.")
print("[INFO] Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run detection
    results = model(frame, conf=0.35, verbose=False)
    boxes = results[0].boxes

    detected_persons = []
    detected_ppe = []

    # Separate detections
    for box in boxes:
        cls_id = int(box.cls[0].item())
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())

        if cls_id == CLASS_PERSON:
            detected_persons.append(coords)
        elif cls_id in REQUIRED_PPE:
            detected_ppe.append((cls_id, coords, conf))

    violation_in_current_frame = False

    # Check each worker
    for idx, p_box in enumerate(detected_persons):
        px1, py1, px2, py2 = map(int, p_box)
        items_worn = set()

        for cls_id, e_box, conf in detected_ppe:
            if is_inside(e_box, p_box):
                items_worn.add(cls_id)
                # Draw cyan box around PPE item
                ex1, ey1, ex2, ey2 = map(int, e_box)
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 255, 0), 1)
                cv2.putText(frame, REQUIRED_PPE[cls_id], (ex1, ey1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Check for missing items
        missing_items = [name for cid, name in REQUIRED_PPE.items() if cid not in items_worn]

        if len(missing_items) == 0:
            status_text = f"Worker #{idx+1}: SAFE"
            box_color = (0, 255, 0)
        else:
            status_text = f"Worker #{idx+1}: UNSAFE"
            box_color = (0, 0, 255)
            violation_in_current_frame = True

        # Draw Person Box and Status
        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
        cv2.putText(frame, status_text, (px1, max(20, py1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        if missing_items:
            missing_text = "MISSING: " + ", ".join(missing_items)
            cv2.putText(frame, missing_text, (px1, min(frame.shape[0] - 10, py2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Trigger Alarm and Snapshot if violation detected and cooldown elapsed
    current_time = time.time()
    if violation_in_current_frame and (current_time - last_alert_time > ALERT_COOLDOWN):
        last_alert_time = current_time
        
        # 1. Play Audio Alarm (Frequency 1000 Hz, Duration 300 ms)
        winsound.Beep(1000, 300)
        
        # 2. Save Evidence Snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(VIOLATIONS_DIR, f"violation_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[ALERT] PPE Violation Detected! Evidence saved to: {filename}")

    # Display video
    cv2.imshow("Industrial PPE Safety Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()