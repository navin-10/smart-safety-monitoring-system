import os
import re
import cv2
import time
import json
import serial
import serial.tools.list_ports
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

# Optional Windows-specific audio beep
try:
    import winsound
except ImportError:
    winsound = None

# Configure safe PyTorch globals for YOLO checkpoint serialization
torch.serialization.add_safe_globals([
    DetectionModel,
    torch.nn.Sequential,
    torch.nn.modules.container.Sequential,
    Conv
])

app = Flask(__name__)

# ----------------------------------------------------
# 1. HARDWARE & APPLICATION CONFIGURATION
# ----------------------------------------------------
MODEL_PATH = os.path.join("models", "best.pt")
VIOLATIONS_DIR = os.path.join("static", "violations")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

def detect_arduino_port(preferred="COM23"):
    """Identifies the actual COM port assigned to the connected Arduino."""
    ports = list(serial.tools.list_ports.comports())
    if preferred:
        for p in ports:
            if p.device.upper() == preferred.upper():
                return p.device
    for p in ports:
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        if "arduino" in desc or "arduino" in mfg or "ch340" in desc or "ch340" in mfg:
            return p.device
    if ports:
        return ports[0].device
    return preferred or "COM23"

# Hardware Ports & Parameters (Configurable via environment variables)
DEFAULT_PORT = detect_arduino_port("COM23")
ARDUINO_PORT = os.getenv("ARDUINO_PORT", DEFAULT_PORT)
ARDUINO_BAUD = int(os.getenv("ARDUINO_BAUD", "9600"))
GAS_THRESHOLD = int(os.getenv("GAS_THRESHOLD", "400"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "")  # Path to test video file, if provided
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))
ALERT_COOLDOWN = float(os.getenv("ALERT_COOLDOWN", "4.0"))  # Seconds between alarm and snapshot
DATA_STALE_TIMEOUT = 5.0  # Seconds without telemetry before marking link stale
SERIAL_TIMEOUT = 2.0

# ----------------------------------------------------
# 2. LOAD TRAINED YOLO11 MODEL & CLASS DEFINITIONS
# ----------------------------------------------------
print(f"[INFO] Loading trained YOLO11 model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# Verified classes from model.names:
# {0: 'Boots', 1: 'Ear-protection', 2: 'Glass', 3: 'Glove', 4: 'Helmet', 5: 'Mask', 6: 'Person', 7: 'Vest'}
CLASS_PERSON = 6
REQUIRED_PPE = {
    4: "Helmet",
    7: "Vest",
    2: "Glass",   # Goggles / Glasses
    3: "Glove"
}

print(f"[INFO] Verified Model Classes: {model.names}")
print(f"[INFO] Monitored Required PPE: {REQUIRED_PPE}")

# ----------------------------------------------------
# 3. TELEMETRY & VIOLATIONS STATE
# ----------------------------------------------------
telemetry_data = {
    "connection": "DISCONNECTED",
    "port": ARDUINO_PORT,
    "baud": ARDUINO_BAUD,
    "gas_raw": 0,
    "gas_ppm": None,
    "gas_ppm_display": "Uncalibrated",
    "gas_hazard": False,
    "gas_status": "DISCONNECTED",  # "SAFE", "HAZARD", "DISCONNECTED", "UNKNOWN"
    "fan": 0,
    "fan_status": "DISCONNECTED",  # "COMMANDED ON", "COMMANDED OFF", "UNKNOWN", "DISCONNECTED"
    "buzzer": 0,
    "led": 0,
    "safe_workers": 0,
    "unsafe_workers": 0,
    "overall_status": "NORMAL",
    "worker_details": [],
    "last_seen": 0,
    "last_telemetry_time": "Never",
    "camera_status": "INITIALIZING"
}

violations_log = []
MAX_VIOLATIONS = 50
last_alert_time = 0
last_gas_alert_time = 0

serial_lock = threading.Lock()
serial_thread_running = False
active_serial_conn = None

def is_inside(ppe_box, person_box):
    """Spatial Association: Checks if PPE center point falls inside the Worker bounding box."""
    px1, py1, px2, py2 = person_box
    ex1, ey1, ex2, ey2 = ppe_box
    center_x = (ex1 + ex2) / 2.0
    center_y = (ey1 + ey2) / 2.0
    return (px1 <= center_x <= px2) and (py1 <= center_y <= py2)

# ----------------------------------------------------
# 4. BACKGROUND ARDUINO SERIAL READER & WRITER
# ----------------------------------------------------
def send_arduino_command(cmd_dict):
    """Thread-safe helper to send JSON commands to Arduino if connected."""
    global active_serial_conn
    with serial_lock:
        if active_serial_conn is not None and active_serial_conn.is_open:
            try:
                msg = json.dumps(cmd_dict) + "\n"
                active_serial_conn.write(msg.encode("utf-8"))
                return True
            except Exception as e:
                print(f"[WARN] Failed to send command to Arduino: {e}")
                return False
    return False

def serial_reader():
    """Background thread that reads JSON lines from the Arduino, manages connection,
    handles malformed data, and updates `telemetry_data`. Continuously reconnects if dropped.
    """
    global telemetry_data, serial_thread_running, active_serial_conn, last_gas_alert_time
    
    while serial_thread_running:
        ser = None
        current_port = telemetry_data.get("port", ARDUINO_PORT)
        # Attempt to verify or resolve available port
        target_port = detect_arduino_port(current_port)
        try:
            ser = serial.Serial(target_port, ARDUINO_BAUD, timeout=SERIAL_TIMEOUT)
            now = time.time()
            with serial_lock:
                active_serial_conn = ser
                telemetry_data["port"] = target_port
                telemetry_data["connection"] = "CONNECTED"
                telemetry_data["last_seen"] = now
                telemetry_data["last_telemetry_time"] = datetime.fromtimestamp(now).strftime("%H:%M:%S")
            print(f"[INFO] Connected to Arduino on {target_port} at {ARDUINO_BAUD} baud.")
            time.sleep(2)  # Allow Arduino bootloader to settle

            while serial_thread_running and ser.is_open:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                now = time.time()
                
                # Check for staleness if no data received for a while
                if not line:
                    with serial_lock:
                        if now - telemetry_data["last_seen"] > DATA_STALE_TIMEOUT:
                            if telemetry_data["connection"] == "CONNECTED":
                                telemetry_data["connection"] = "TIMEOUT"
                                telemetry_data["gas_status"] = "UNKNOWN"
                                telemetry_data["fan_status"] = "UNKNOWN"
                    continue

                data = None
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Legacy / plain-text fallback: e.g. "Gas Value: 45" or "Gas Raw: 45"
                    match = re.search(r'Gas\s+(?:Raw|Value):\s*(\d+)', line, re.IGNORECASE)
                    if match:
                        raw_val = int(match.group(1))
                        is_haz = raw_val >= GAS_THRESHOLD
                        data = {
                            "gas_raw": raw_val,
                            "gas_hazard": is_haz,
                            "fan": 1 if is_haz else 0,
                            "buzzer": 1 if is_haz else 0,
                            "led": 1 if is_haz else 0
                        }
                    else:
                        continue

                # Parse and validate incoming fields
                raw_gas = data.get("gas_raw", telemetry_data["gas_raw"])
                # If explicit gas_hazard boolean is sent, use it; otherwise evaluate against calibrated threshold
                if "gas_hazard" in data:
                    hazard = bool(data["gas_hazard"])
                else:
                    hazard = raw_gas >= GAS_THRESHOLD

                ppm_val = data.get("gas_ppm", None)
                fan_val = data.get("fan", telemetry_data["fan"])
                buzzer_val = data.get("buzzer", telemetry_data["buzzer"])
                led_val = data.get("led", telemetry_data["led"])
                time_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")

                with serial_lock:
                    telemetry_data.update({
                        "connection": "CONNECTED",
                        "last_seen": now,
                        "last_telemetry_time": time_str,
                        "gas_raw": raw_gas,
                        "gas_ppm": ppm_val,
                        "gas_hazard": hazard,
                        "gas_status": "HAZARD" if hazard else "SAFE",
                        "fan": fan_val,
                        "fan_status": "COMMANDED ON" if fan_val else "COMMANDED OFF",
                        "buzzer": buzzer_val,
                        "led": led_val
                    })

                # Log gas hazard event if hazard condition active with deduplication cooldown
                if hazard and (now - last_gas_alert_time > ALERT_COOLDOWN):
                    last_gas_alert_time = now
                    ts_display = datetime.now().strftime("%d %b %Y, %H:%M:%S")
                    hazard_record = {
                        "time": ts_display,
                        "worker": "ENVIRONMENT",
                        "issue": f"GAS HAZARD DETECTED (Raw ADC: {raw_gas} >= {GAS_THRESHOLD})",
                        "image": ""
                    }
                    violations_log.insert(0, hazard_record)
                    if len(violations_log) > MAX_VIOLATIONS:
                        del violations_log[MAX_VIOLATIONS:]

        except serial.SerialException as e:
            with serial_lock:
                active_serial_conn = None
                telemetry_data["connection"] = "DISCONNECTED"
                telemetry_data["gas_status"] = "DISCONNECTED"
                telemetry_data["fan_status"] = "DISCONNECTED"
            time.sleep(3)
        except Exception as e:
            with serial_lock:
                active_serial_conn = None
                telemetry_data["connection"] = "DISCONNECTED"
                telemetry_data["gas_status"] = "DISCONNECTED"
                telemetry_data["fan_status"] = "DISCONNECTED"
            time.sleep(3)
        finally:
            if ser is not None and ser.is_open:
                try:
                    ser.close()
                except Exception:
                    pass
            with serial_lock:
                active_serial_conn = None

# ----------------------------------------------------
# 5. VIDEO STREAM GENERATOR & PPE DETECTION
# ----------------------------------------------------
def create_fallback_frame(message="Camera Feed Unavailable"):
    """Generates an informative placeholder frame when camera access fails."""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background styling
    cv2.rectangle(placeholder, (0, 0), (640, 480), (22, 27, 34), -1)
    cv2.rectangle(placeholder, (20, 20), (620, 460), (48, 54, 61), 2)
    # Status text
    cv2.putText(placeholder, message, (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(placeholder, f"Port / Device Index: {CAMERA_INDEX}", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(placeholder, f"Time: {timestamp_str}", (60, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 148, 158), 1)
    ret, buffer = cv2.imencode('.jpg', placeholder)
    return buffer.tobytes()

def open_camera():
    """Attempts to open camera using DirectShow on Windows, falling back to default backend."""
    source = VIDEO_SOURCE if (VIDEO_SOURCE and os.path.exists(VIDEO_SOURCE)) else CAMERA_INDEX
    # Try DirectShow first on Windows for reliability
    if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def generate_frames():
    global last_alert_time, telemetry_data
    
    cap = open_camera()

    # If camera failed to open, stream informative fallback frames at 2 Hz without burning CPU
    if not cap.isOpened():
        with serial_lock:
            telemetry_data["camera_status"] = "UNAVAILABLE"
        while True:
            frame_bytes = create_fallback_frame("Camera not available - Check connection")
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)

    with serial_lock:
        telemetry_data["camera_status"] = "ACTIVE"

    while True:
        success, frame = cap.read()
        if not success:
            # Yield placeholder and retry after short sleep
            with serial_lock:
                telemetry_data["camera_status"] = "FRAME_ERROR"
            frame_bytes = create_fallback_frame("Camera Frame Error - Retrying...")
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
            continue

        with serial_lock:
            telemetry_data["camera_status"] = "ACTIVE"

        # Run YOLO11 PPE detection
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        detected_persons = []
        detected_ppe = []

        # Separate person boxes and PPE boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())

            if cls_id == CLASS_PERSON:
                detected_persons.append(coords)
            elif cls_id in REQUIRED_PPE:
                detected_ppe.append((cls_id, coords, conf))

        safe_count = 0
        unsafe_count = 0
        worker_details = []

        # Single unified pass: evaluate each detected worker and draw annotations
        for idx, p_box in enumerate(detected_persons):
            px1, py1, px2, py2 = map(int, p_box)
            items_worn = set()

            # Identify PPE items associated with this worker
            for cls_id, e_box, conf in detected_ppe:
                if is_inside(e_box, p_box):
                    items_worn.add(cls_id)
                    ex1, ey1, ex2, ey2 = map(int, e_box)
                    # Draw cyan box around detected PPE item
                    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 255, 0), 1)
                    cv2.putText(frame, REQUIRED_PPE[cls_id], (ex1, max(12, ey1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Determine missing PPE items
            missing_items = [name for cid, name in REQUIRED_PPE.items() if cid not in items_worn]
            worn_names = [REQUIRED_PPE[cid] for cid in items_worn if cid in REQUIRED_PPE]

            if len(missing_items) == 0:
                status_text = f"Worker #{idx+1}: SAFE"
                box_color = (0, 255, 0)  # Green
                safe_count += 1
                worker_status = "SAFE"
            else:
                status_text = f"Worker #{idx+1}: UNSAFE"
                box_color = (0, 0, 255)  # Red
                unsafe_count += 1
                worker_status = "VIOLATION"

            worker_details.append({
                "id": idx + 1,
                "ppe": worn_names,
                "missing": missing_items,
                "status": worker_status
            })

            # Draw worker bounding box and status label
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, status_text, (px1, max(20, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            if missing_items:
                missing_text = "MISSING: " + ", ".join(missing_items)
                cv2.putText(frame, missing_text, (px1, min(frame.shape[0] - 10, py2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Update telemetry data for worker safety
        with serial_lock:
            telemetry_data["safe_workers"] = safe_count
            telemetry_data["unsafe_workers"] = unsafe_count
            telemetry_data["worker_details"] = worker_details

        # Trigger Audio Alarm & Snapshot on violation with cooldown deduplication
        curr_time = time.time()
        if unsafe_count > 0 and (curr_time - last_alert_time > ALERT_COOLDOWN):
            last_alert_time = curr_time
            if winsound is not None:
                try:
                    winsound.Beep(1000, 250)
                except Exception:
                    pass

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(VIOLATIONS_DIR, f"violation_{timestamp}.jpg")
            cv2.imwrite(filename, frame)

            # Record violation in log for dashboard table
            ts_display = datetime.now().strftime("%d %b %Y, %H:%M:%S")
            for w in worker_details:
                if w["status"] == "VIOLATION":
                    issue_str = "Missing " + ", ".join(w["missing"]) if w["missing"] else "PPE Violation"
                    violation_record = {
                        "time": ts_display,
                        "worker": f"Worker {w['id']}",
                        "issue": issue_str,
                        "image": f"/static/violations/violation_{timestamp}.jpg"
                    }
                    violations_log.insert(0, violation_record)

            if len(violations_log) > MAX_VIOLATIONS:
                del violations_log[MAX_VIOLATIONS:]

        # Encode frame as JPEG byte stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ----------------------------------------------------
# 6. FLASK WEB ROUTES
# ----------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def get_telemetry():
    with serial_lock:
        response = dict(telemetry_data)

    # Format human-readable gas PPM display without claiming uncalibrated accuracy
    if response["connection"] == "DISCONNECTED":
        response["gas_ppm_display"] = "Disconnected"
        response["gas_status"] = "DISCONNECTED"
        response["fan_status"] = "DISCONNECTED"
    elif response["gas_ppm"] is None:
        response["gas_ppm_display"] = f"Raw: {response['gas_raw']} (Uncalibrated)"
    else:
        response["gas_ppm_display"] = f"{response['gas_ppm']} ppm"

    # Determine overall system status accurately
    gas_hazard = response.get("gas_hazard", False) if response["connection"] == "CONNECTED" else False
    ppe_hazard = response.get("unsafe_workers", 0) > 0

    if gas_hazard and ppe_hazard:
        response["overall_status"] = "CRITICAL EMERGENCY"
    elif gas_hazard:
        response["overall_status"] = "GAS HAZARD"
    elif ppe_hazard:
        response["overall_status"] = "PPE VIOLATION"
    elif response["connection"] == "DISCONNECTED":
        response["overall_status"] = "ARDUINO OFFLINE"
    else:
        response["overall_status"] = "NORMAL"

    return jsonify(response)

@app.route('/api/violations')
def get_violations():
    return jsonify(violations_log[:20])

@app.route('/api/health')
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "classes": list(model.names.values()),
        "arduino_port": ARDUINO_PORT,
        "arduino_status": telemetry_data["connection"],
        "camera_status": telemetry_data["camera_status"]
    })

# ----------------------------------------------------
# 7. MAIN ENTRY POINT
# ----------------------------------------------------
if __name__ == '__main__':
    print("\n=======================================================")
    print(" [INFO] Industrial Safety Dashboard starting...")
    print(f" [INFO] Arduino target port: {ARDUINO_PORT} at {ARDUINO_BAUD} baud")
    print(f" [INFO] Gas hazard threshold: {GAS_THRESHOLD}")
    print(f" [INFO] Camera Index: {CAMERA_INDEX}")
    print(" [INFO] Open your web browser at: http://127.0.0.1:5000")
    print("=======================================================\n")

    # Start background serial reader daemon
    if not serial_thread_running:
        serial_thread_running = True
        serial_thread = threading.Thread(target=serial_reader, daemon=True)
        serial_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False)