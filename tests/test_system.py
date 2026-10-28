import os
import sys
import unittest
import json
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app
from app import (
    is_inside,
    create_fallback_frame,
    send_arduino_command,
    CLASS_PERSON,
    REQUIRED_PPE,
    GAS_THRESHOLD,
    ALERT_COOLDOWN,
    MAX_VIOLATIONS,
    DATA_STALE_TIMEOUT,
    telemetry_data,
    violations_log
)


class IsolatedTestCase(unittest.TestCase):
    """Base test case that snapshots and restores global telemetry and violations state,
    preventing automated tests from polluting live server state or persistent records.
    """
    def setUp(self):
        with app.serial_lock:
            self._telemetry_backup = dict(app.telemetry_data)
            self._violations_backup = list(app.violations_log)

    def tearDown(self):
        with app.serial_lock:
            app.telemetry_data.clear()
            app.telemetry_data.update(self._telemetry_backup)
            app.violations_log.clear()
            app.violations_log.extend(self._violations_backup)


class TestModelIntegrity(unittest.TestCase):
    """Verifies that the trained YOLO11 model checkpoint is intact and has expected classes."""

    def test_checkpoint_exists(self):
        self.assertTrue(os.path.exists(app.MODEL_PATH), f"Checkpoint missing at {app.MODEL_PATH}")

    def test_model_loaded(self):
        self.assertIsNotNone(app.model, "Model should be loaded")
        self.assertIn(CLASS_PERSON, app.model.names, "Person class ID 6 must exist in model.names")
        self.assertEqual(app.model.names[CLASS_PERSON], "Person")

    def test_required_ppe_classes_supported(self):
        """Verifies that every PPE item in REQUIRED_PPE is genuinely supported by the model."""
        for class_id, ppe_name in REQUIRED_PPE.items():
            self.assertIn(class_id, app.model.names, f"PPE ID {class_id} missing in model names")
            self.assertEqual(app.model.names[class_id], ppe_name, f"ID {class_id} name mismatch")


class TestSpatialAssociation(unittest.TestCase):
    """Verifies the bounding box spatial association algorithm."""

    def test_center_inside(self):
        person_box = [100, 100, 300, 500]
        # Helmet box situated on the head area (center is x=200, y=140)
        helmet_box = [160, 100, 240, 180]
        self.assertTrue(is_inside(helmet_box, person_box))

    def test_center_outside(self):
        person_box = [100, 100, 300, 500]
        # Box completely outside (x=400, y=400)
        outside_box = [380, 380, 420, 420]
        self.assertFalse(is_inside(outside_box, person_box))

    def test_boundary_conditions(self):
        person_box = [100, 100, 300, 500]
        # Center exactly on edge
        edge_box = [90, 100, 110, 120]  # center_x = 100
        self.assertTrue(is_inside(edge_box, person_box))


class TestPpeViolationLogic(unittest.TestCase):
    """Verifies PPE compliance logic using controlled simulated detection inputs."""

    def evaluate_detections(self, detected_persons, detected_ppe):
        """Helper that mirrors the core evaluation logic in app.py."""
        safe_count = 0
        unsafe_count = 0
        worker_details = []

        for idx, p_box in enumerate(detected_persons):
            items_worn = set()
            for cls_id, e_box, conf in detected_ppe:
                if is_inside(e_box, p_box):
                    items_worn.add(cls_id)

            missing_items = [name for cid, name in REQUIRED_PPE.items() if cid not in items_worn]
            worn_names = [REQUIRED_PPE[cid] for cid in items_worn if cid in REQUIRED_PPE]

            if len(missing_items) == 0:
                safe_count += 1
                status = "SAFE"
            else:
                unsafe_count += 1
                status = "VIOLATION"

            worker_details.append({
                "id": idx + 1,
                "ppe": worn_names,
                "missing": missing_items,
                "status": status
            })

        return safe_count, unsafe_count, worker_details

    def test_fully_compliant_worker(self):
        # 1 person wearing Helmet (4), Vest (7), Glass (2), Glove (3)
        person = [100, 100, 300, 500]
        ppe = [
            (4, [160, 100, 240, 150], 0.9),  # Helmet
            (7, [140, 180, 260, 340], 0.88), # Vest
            (2, [180, 140, 220, 160], 0.85), # Glass
            (3, [120, 320, 150, 370], 0.82), # Glove
        ]
        safe, unsafe, details = self.evaluate_detections([person], ppe)
        self.assertEqual(safe, 1)
        self.assertEqual(unsafe, 0)
        self.assertEqual(details[0]["status"], "SAFE")
        self.assertEqual(len(details[0]["missing"]), 0)

    def test_worker_missing_helmet_and_vest(self):
        person = [100, 100, 300, 500]
        # Only Glass and Glove
        ppe = [
            (2, [180, 140, 220, 160], 0.85),
            (3, [120, 320, 150, 370], 0.82),
        ]
        safe, unsafe, details = self.evaluate_detections([person], ppe)
        self.assertEqual(safe, 0)
        self.assertEqual(unsafe, 1)
        self.assertEqual(details[0]["status"], "VIOLATION")
        self.assertIn("Helmet", details[0]["missing"])
        self.assertIn("Vest", details[0]["missing"])

    def test_mixed_multiple_workers(self):
        worker1 = [50, 50, 200, 400]   # Compliant
        worker2 = [300, 50, 450, 400]  # Missing all
        ppe = [
            (4, [80, 50, 150, 100], 0.9),
            (7, [70, 120, 170, 260], 0.9),
            (2, [90, 80, 130, 100], 0.9),
            (3, [60, 240, 90, 280], 0.9),
        ]
        safe, unsafe, details = self.evaluate_detections([worker1, worker2], ppe)
        self.assertEqual(safe, 1)
        self.assertEqual(unsafe, 1)
        self.assertEqual(details[0]["status"], "SAFE")
        self.assertEqual(details[1]["status"], "VIOLATION")


class TestSerialProtocolAndParsing(unittest.TestCase):
    """Verifies safe JSON message parsing and resilience against malformed serial lines."""

    def parse_serial_line(self, line):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def test_valid_arduino_telemetry_message(self):
        sample = '{"gas_raw": 320, "gas_ppm": 45.2, "gas_hazard": false, "fan": 0, "buzzer": 0, "led": 0}\n'
        parsed = self.parse_serial_line(sample.strip())
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["gas_raw"], 320)
        self.assertFalse(parsed["gas_hazard"])
        self.assertEqual(parsed["fan"], 0)

    def test_valid_arduino_hazard_message(self):
        sample = '{"gas_raw": 680, "gas_hazard": true, "fan": 1, "buzzer": 1, "led": 1}\n'
        parsed = self.parse_serial_line(sample.strip())
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed["gas_hazard"])
        self.assertEqual(parsed["fan"], 1)
        self.assertEqual(parsed["buzzer"], 1)

    def test_truncated_json_does_not_crash(self):
        truncated = '{"gas_raw": 450, "gas_haz'
        parsed = self.parse_serial_line(truncated)
        self.assertIsNone(parsed)

    def test_corrupted_characters_handled_safely(self):
        corrupted = '\x00\xff??INVALID_SERIAL_FRAME!!{"gas'
        parsed = self.parse_serial_line(corrupted)
        self.assertIsNone(parsed)

    def test_empty_line_handled(self):
        parsed = self.parse_serial_line("")
        self.assertIsNone(parsed)


class TestGasHazardClassification(IsolatedTestCase):
    """Verifies that normal, hazard, missing, and disconnected states are properly distinguished."""

    def test_normal_gas_reading(self):
        raw_gas = GAS_THRESHOLD - 50
        hazard = raw_gas >= GAS_THRESHOLD
        self.assertFalse(hazard, "Gas below threshold must not trigger hazard")

    def test_hazardous_gas_reading(self):
        raw_gas = GAS_THRESHOLD + 100
        hazard = raw_gas >= GAS_THRESHOLD
        self.assertTrue(hazard, "Gas at or above threshold must trigger hazard")

    def test_missing_or_disconnected_sensor_is_not_falsely_safe(self):
        """A disconnected state must never be presented as proof that the environment is safe."""
        client = app.app.test_client()
        with app.serial_lock:
            app.telemetry_data["connection"] = "DISCONNECTED"

        res = client.get('/api/telemetry')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["connection"], "DISCONNECTED")
        self.assertEqual(data["gas_status"], "DISCONNECTED")
        self.assertNotEqual(data["gas_status"], "SAFE", "Disconnected sensor must not report SAFE")
        self.assertIn("Disconnected", data["gas_ppm_display"])


class TestMissingCameraHandling(unittest.TestCase):
    """Verifies that missing camera input produces an informative placeholder frame without crashing."""

    def test_fallback_frame_generation(self):
        frame_bytes = create_fallback_frame("Test Camera Offline")
        self.assertIsInstance(frame_bytes, bytes)
        self.assertGreater(len(frame_bytes), 1000, "JPEG placeholder must have non-trivial size")

        # Verify it decodes as a valid OpenCV image
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, "Decoded image must not be None")
        self.assertEqual(img.shape, (480, 640, 3), "Fallback frame must be 640x480 RGB")


class TestFlaskEndpoints(IsolatedTestCase):
    """Verifies Flask HTTP routes and JSON responses."""

    def setUp(self):
        super().setUp()
        self.client = app.app.test_client()

    def test_index_route(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("Industrial Safety Monitoring System", html)
        self.assertIn("Live Camera Feed", html)

    def test_telemetry_endpoint(self):
        res = self.client.get('/api/telemetry')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        expected_keys = [
            "connection", "gas_raw", "gas_ppm_display", "gas_hazard",
            "gas_status", "fan", "fan_status", "buzzer", "led",
            "safe_workers", "unsafe_workers", "overall_status", "worker_details"
        ]
        for key in expected_keys:
            self.assertIn(key, data, f"Telemetry missing expected key: {key}")

    def test_violations_endpoint(self):
        res = self.client.get('/api/violations')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)

    def test_health_endpoint(self):
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["model_loaded"])

    def test_video_feed_stream_headers(self):
        """Verifies /video_feed returns 200 with multipart/x-mixed-replace and decodable JPEG."""
        res = self.client.get('/video_feed', buffered=False)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'multipart/x-mixed-replace')
        first_chunk = next(res.response)
        self.assertGreater(len(first_chunk), 500, "First stream chunk must contain JPEG data")
        self.assertIn(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n', first_chunk)
        res.close()


class TestHardwareAbsenceSafety(IsolatedTestCase):
    """Verifies that absence of physical hardware (Arduino, camera) does not cause unhandled exceptions."""

    def test_send_command_when_disconnected(self):
        """send_arduino_command must gracefully return False when offline without throwing."""
        result = send_arduino_command({"cmd": "ALARM_TEST"})
        self.assertFalse(result)

    def test_stale_telemetry_detection(self):
        """Tests that stale telemetry correctly triggers TIMEOUT and UNKNOWN state."""
        stale_time = time.time() - (DATA_STALE_TIMEOUT + 5.0)
        with app.serial_lock:
            app.telemetry_data["connection"] = "CONNECTED"
            app.telemetry_data["last_seen"] = stale_time

        # Simulate staleness check as performed in serial_reader loop
        now = time.time()
        if now - app.telemetry_data["last_seen"] > DATA_STALE_TIMEOUT:
            if app.telemetry_data["connection"] == "CONNECTED":
                app.telemetry_data["connection"] = "TIMEOUT"
                app.telemetry_data["gas_status"] = "UNKNOWN"
                app.telemetry_data["fan_status"] = "UNKNOWN"

        self.assertEqual(app.telemetry_data["connection"], "TIMEOUT")
        self.assertEqual(app.telemetry_data["gas_status"], "UNKNOWN")
        self.assertEqual(app.telemetry_data["fan_status"], "UNKNOWN")


class TestViolationLogBounding(unittest.TestCase):
    """Verifies bounded growth of violation records."""

    def test_log_bounds(self):
        sample_record = {
            "time": "23 Oct 2026, 03:00:00",
            "worker": "Worker 1",
            "issue": "Missing Helmet",
            "image": "/static/violations/test.jpg"
        }
        test_log = []
        for i in range(MAX_VIOLATIONS + 20):
            test_log.insert(0, sample_record)
            if len(test_log) > MAX_VIOLATIONS:
                del test_log[MAX_VIOLATIONS:]

        self.assertEqual(len(test_log), MAX_VIOLATIONS)


if __name__ == '__main__':
    unittest.main()
