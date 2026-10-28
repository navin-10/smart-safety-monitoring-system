import json
import sys
import time

import serial


PORT = "COM23"
BAUD_RATE = 9600
MAX_LINES = 10
READ_TIMEOUT_SECONDS = 1
READ_WINDOW_SECONDS = 10


def main():
    arduino = None
    lines_received = 0

    try:
        arduino = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=READ_TIMEOUT_SECONDS,
        )
        print(f"Connected to Arduino on {PORT} at {BAUD_RATE} baud.")
        time.sleep(2)
        print(f"Waiting up to {READ_WINDOW_SECONDS} seconds for data...")

        deadline = time.monotonic() + READ_WINDOW_SECONDS
        while lines_received < MAX_LINES and time.monotonic() < deadline:
            line = arduino.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue

            lines_received += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(line)
            else:
                print(json.dumps(data, indent=2))

        if lines_received == 0:
            print(
                "No serial data received. Check that the Arduino sketch prints "
                "messages, the baud rate is 9600, and COM23 is the correct port."
            )
            return 1
    except serial.SerialException as error:
        print(f"Could not connect to Arduino on {PORT}: {error}")
        print("Check the USB cable, COM port, and that no other app is using it.")
        return 1
    finally:
        if arduino is not None and arduino.is_open:
            arduino.close()
            print("Serial connection closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
