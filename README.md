# AI-Powered Smart Safety Monitoring System for Industrial Workplace Safety and Hazard Detection

An integrated computer vision and IoT industrial safety monitoring platform. The system combines a trained **YOLO11** object detection model for Personal Protective Equipment (PPE) compliance detection with real-time **MQ-2 gas sensor telemetry** and an industrial web dashboard built on **Flask**.

---

## 1. System Overview & Architecture

The application continuously captures workplace video feeds, evaluates worker PPE compliance in real time, monitors environmental gas levels through microcontroller serial communication, records violation snapshots, and visualizes status on a dark-themed industrial dashboard.

```
                    +---------------------------+
                    | Live Camera (DirectShow)   |
                    +-------------+-------------+
                                  | Frame Ingestion (640x480)
                                  v
                    +---------------------------+
                    | YOLO11 Inference Engine   | <--- models/best.pt
                    | (Persons + PPE Detection) |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Spatial Association       |
                    | (is_inside Bounding Box)  |
                    +-------------+-------------+
                                  |
               +------------------+------------------+
               |                                     |
               v                                     v
  [Compliant: All PPE Worn]              [Violation: Missing PPE]
  - Tag: SAFE (Green)                    - Tag: VIOLATION (Red)
  - Increment Safe Counter               - Increment Violation Counter
                                         - Cooldown Timer (4.0s)
                                         - Save Snapshot (static/violations/)
                                         - Trigger Audio Alert (winsound)

                    +---------------------------+
                    | Arduino Hardware / MQ-2   |
                    +-------------+-------------+
                                  | Serial JSON (9600 baud)
                                  v
                    +---------------------------+
                    | Background Serial Daemon  | <--- Auto-reconnect & Stale Detection
                    +-------------+-------------+
                                  |
                                  v
+-----------------------------------------------------------------+
|                       Central Telemetry                         |
|  - Safe / Unsafe Worker Counts   - Gas Raw ADC (0-1023)         |
|  - Per-Worker PPE Detail List    - Gas Hazard Flag (>= 400 ADC) |
|  - Hardware Connection State     - Fan / Buzzer / LED States    |
+--------------------------------+--------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|               Flask Web Dashboard (Port 5000)                   |
|  - Live MJPEG Stream (/video_feed)                              |
|  - JSON Telemetry Polling (/api/telemetry)                      |
|  - Violation Log Table (/api/violations)                        |
|  - System Health & Diagnostics (/api/health)                    |
+-----------------------------------------------------------------+
```

---

## 2. Model & PPE Class Mapping

The system utilizes a custom-trained YOLO11 model checkpoint located at `models/best.pt`.

### Dataset & Training Details:
- **Base Architecture**: Ultralytics YOLO11
- **Dataset**: Industrial Workplace Safety & PPE dataset annotated across multi-hazard factory environments.
- **Model Checkpoint Size**: `~5.22 MB` (Lightweight edge-deployable PyTorch checkpoint)

### Verified Classes in Model Weights (`model.names`):
| Class ID | Class Name | Role in Safety Monitor |
|---|---|---|
| **0** | `Boots` | Detected class |
| **1** | `Ear-protection` | Detected class |
| **2** | `Glass` | **Required PPE** (Safety Goggles / Glasses) |
| **3** | `Glove` | **Required PPE** (Safety Gloves) |
| **4** | `Helmet` | **Required PPE** (Hard Hat / Helmet) |
| **5** | `Mask` | Detected class |
| **6** | `Person` | **Target Entity** (Worker Detection) |
| **7** | `Vest` | **Required PPE** (High-Visibility Safety Vest) |

### Monitored PPE Configuration:
By default, the compliance engine checks for **Helmet**, **Vest**, **Glass**, and **Glove** for every detected person. If any of these four items are not spatially associated inside the person's bounding box, the worker is flagged as `VIOLATION` with missing items explicitly listed.

---

## 3. Key Software Features

1. **DirectShow Video Capture**: Utilizes `cv2.CAP_DSHOW` on Windows platforms for reliable, unblocked 640x480 frame capture.
2. **CPU-Safe Camera Fallback**: If the camera is missing or disconnected, a lightweight placeholder frame is generated at 2 Hz (`time.sleep(0.5)`), keeping CPU utilization under 1%.
3. **Spatial Containment Algorithm**: `is_inside()` determines whether detected PPE item centers fall within individual worker bounding boxes.
4. **Snapshot Deduplication**: Violation snapshots are debounced using a 4.0-second cooldown window to prevent storage exhaustion.
5. **Persistent Serial Daemon**: Background thread handles serial connection lifecycle, automatically attempting reconnection every 3 seconds if hardware is unplugged.
6. **Truthful State Representation**: Sensor readings are never fabricated. When hardware is disconnected, gas concentration displays as `DISCONNECTED` / `N/A`, and fan status displays as `OFFLINE`.
7. **Comprehensive Unit Test Suite**: 26 standalone unit tests covering model integrity, math routines, serial parsing, gas classification, and HTTP routes without requiring hardware.

---

## 4. Hardware Architecture & Pin Configuration

The firmware runs on an **Arduino Uno** located at `arduino/smart_safety_system/smart_safety_system.ino`.

### Hardware Pin Mapping:
| Component | Arduino Pin | Mode | Description |
|---|---|---|---|
| **MQ-2 Gas Sensor** | `A0` | Analog Input | Reads gas/smoke concentration (0–1023 ADC) |
| **5V Relay Module** | `Pin 7` | Digital Output | Active-LOW control (`LOW` = Fan ON, `HIGH` = Fan OFF) |
| **Warning Buzzer** | `Pin 8` | Digital Output | Active-HIGH siren alarm |
| **Warning Red LED** | `Pin 9` | Digital Output | Active-HIGH visual hazard indicator |
| **Common Ground** | `GND` | Ground | Common reference ground for all sensors and modules |
| **VCC Power** | `5V` | Power | 5V supply from Arduino for MQ-2 and Relay VCC |

> **Note on Relay Safety**: In `setup()`, Pin 7 is pre-charged to `HIGH` before declaring it as `OUTPUT` to prevent momentary relay actuation glitches during board boot.

### Expected Serial JSON Schema:
The Arduino transmits newline-terminated JSON strings over USB Serial at **9600 baud**:

```json
{"gas_raw": 320, "gas_ppm": null, "gas_hazard": false, "fan": 0, "buzzer": 0, "led": 0}
```

### Telemetry Field Definitions:
- `gas_raw` *(integer, 0–1023)*: Raw 10-bit ADC reading from MQ-2 sensor.
- `gas_ppm` *(float or null)*: Calibrated concentration in parts per million. Sent as `null` when operating without multi-point calibration.
- `gas_hazard` *(boolean)*: Hazard flag, tripped when `gas_raw >= GAS_THRESHOLD` (default: 400).
- `fan` *(integer, 0 or 1)*: Confirmed relay activation status for the exhaust fan.
- `buzzer` *(integer, 0 or 1)*: Audible buzzer status.
- `led` *(integer, 0 or 1)*: Warning LED status.

---

## 5. Configuration & Environment Variables

Settings can be customized using environment variables without modifying source code:

| Variable | Default | Description |
|---|---|---|
| `ARDUINO_PORT` | Auto-detect (preferred: `COM23`) | Serial port for Arduino USB communication |
| `ARDUINO_BAUD` | `9600` | Baud rate for serial link |
| `GAS_THRESHOLD` | `400` | Raw ADC threshold for gas hazard trigger (0–1023) |
| `CAMERA_INDEX` | `0` | OpenCV device index for webcam |
| `VIDEO_SOURCE` | `""` | Optional path to a video file for offline demonstration |
| `CONF_THRESHOLD` | `0.35` | YOLO11 detection confidence threshold |
| `ALERT_COOLDOWN` | `4.0` | Cooldown period (seconds) between violation snapshots |

---

## 6. Installation & Setup

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.11.x
- Webcam (built-in or USB)
- Git
- *(Optional for hardware telemetry)* Arduino Uno/Nano with MQ-2 sensor, 5V relay, fan, buzzer, and LED.

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/smart-safety-monitoring-system.git
   cd smart-safety-monitoring-system
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv env
   .\env\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 7. Execution Commands

### Option A: One-Click Launcher (Windows)
Double-click `run_system.bat` or run:
```powershell
.\run_system.bat
```
This batch script automatically activates the virtual environment, detects the connected Arduino COM port, prompts for confirmation, opens the browser at `http://127.0.0.1:5000`, and starts the Flask server.

### Option B: Manual Execution
```powershell
# Run the automated test suite
python tests/test_system.py -v

# Start the Flask application
python app.py
```

### Option C: Custom Environment Parameters
```powershell
$env:ARDUINO_PORT = "COM3"
$env:GAS_THRESHOLD = "450"
$env:CAMERA_INDEX = "0"
python app.py
```

---

## 8. API Endpoints

| Endpoint | Method | Format | Description |
|---|---|---|---|
| `/` | `GET` | HTML | Main Industrial Safety Monitoring Dashboard |
| `/video_feed` | `GET` | MJPEG Stream | Live annotated video stream with bounding boxes |
| `/api/telemetry` | `GET` | JSON | Live system telemetry (workers, gas, fan, connection) |
| `/api/violations` | `GET` | JSON | List of recent PPE violation records (up to 20) |
| `/api/health` | `GET` | JSON | Application health, model readiness, and class list |

---

## 9. Academic Limitations & Hardware Testing Notice

The software components have been tested and verified through automated test suites and real-time live browser execution. However, the following physical hardware interactions remain **unverified until the physical hardware setup is connected**:

1. **Physical Serial Handshake**: While the software serial daemon, JSON parser, and reconnection logic are verified, physical data transfer across the USB COM port depends on the connected board.
2. **MQ-2 Analog Gas Calibration**: The MQ-2 sensor produces an analog voltage that varies with heater burn-in, ambient temperature, and humidity. Displayed values are raw ADC units (0–1023). True PPM concentration requires clean-air baseline calibration ($R_0$) and gas-specific curve fitting.
3. **Relay & Exhaust Fan Operation**: Automatic relay closure upon gas hazard is logically implemented in code, but physical fan actuation requires testing with the relay circuit.
4. **Physical Buzzer & LED Actuation**: Sound and light emission on the physical breadboard/shield require hardware confirmation.

---

## 10. Project Directory Structure

```
smart_safety_system/
├── .gitignore                 # Files and patterns ignored by Git
├── app.py                     # Main Flask server, YOLO11 inference, serial daemon
├── check_model.py             # Utility to inspect YOLO11 checkpoint classes
├── requirements.txt           # Pinned Python package dependencies
├── run_system.bat             # One-click Windows startup controller
├── safety_monitor.py          # Standalone OpenCV monitoring script
├── test_arduino.py            # Serial port diagnostic script
├── webcam_detect.py           # Standalone camera test script
├── arduino/
│   └── smart_safety_system/
│       └── smart_safety_system.ino # Arduino Uno firmware (MQ-2, Relay, Buzzer, LED)
├── models/
│   └── best.pt                # Trained YOLO11 model weights (~5.22 MB)
├── static/
│   └── violations/
│       └── .gitkeep           # Keeps directory tracked without committing evidence images
├── templates/
│   └── index.html             # Industrial dashboard UI (HTML/CSS/JavaScript)
└── tests/
    ├── __init__.py            # Test package marker
    └── test_system.py         # 26 automated unit tests with state isolation
```
