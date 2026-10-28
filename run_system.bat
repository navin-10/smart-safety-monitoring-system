@echo off
setlocal enabledelayedexpansion

title Smart Industrial Safety System - Controller
cd /d "%~dp0"

echo =====================================================================
echo  AI-Powered Smart Safety Monitoring System - Single Startup Process
echo =====================================================================
echo.

:: 1. Verify and activate Python virtual environment
if not exist "env\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'env' not found in current directory.
    echo Please make sure 'env' is present or create it using: python -m venv env
    echo.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment (env)...
call env\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    echo.
    pause
    exit /b 1
)

:: 2. Check required Python packages
echo [INFO] Verifying Python package dependencies...
python -c "import flask, cv2, torch, ultralytics, serial; print('[OK] Dependencies verified: Flask, OpenCV, PyTorch, YOLO, PySerial')"
if errorlevel 1 (
    echo [ERROR] Missing required Python packages in virtual environment.
    echo Run: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: 3. Detect connected COM port
for /f "delims=" %%I in ('python -c "import serial.tools.list_ports as lp; ports=list(lp.comports()); ard=[p.device for p in ports if 'arduino' in (p.description or '').lower() or 'ch340' in (p.description or '').lower() or '2341' in (p.hwid or '')]; print(ard[0] if ard else (ports[0].device if ports else 'COM23'))" 2^>nul') do set DETECTED_PORT=%%I

if "%DETECTED_PORT%"=="" set DETECTED_PORT=COM23

echo.
echo Connected Arduino Port Auto-Detected: %DETECTED_PORT%
set /p USER_INPUT_PORT="Press ENTER to use [%DETECTED_PORT%] or type another COM port (e.g. COM3): "
if "%USER_INPUT_PORT%"=="" (
    set ARDUINO_PORT=%DETECTED_PORT%
) else (
    set ARDUINO_PORT=%USER_INPUT_PORT%
)

:: 4. Baud rate & Gas threshold
if "%ARDUINO_BAUD%"=="" set ARDUINO_BAUD=9600
if "%GAS_THRESHOLD%"=="" set GAS_THRESHOLD=400

echo.
echo [CONFIG] Active Configuration:
echo   - Arduino Port:   %ARDUINO_PORT%
echo   - Baud Rate:      %ARDUINO_BAUD%
echo   - Gas Threshold:  %GAS_THRESHOLD% ADC
echo   - Dashboard URL:  http://127.0.0.1:5000
echo.

:: 5. Launch browser once in background after brief delay for server initialization
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"

:: 6. Launch main application
echo [INFO] Starting Flask web server and background telemetry daemon...
echo [INFO] Press Ctrl+C in this terminal window to stop the system.
echo.

python app.py

if errorlevel 1 (
    echo.
    echo =====================================================================
    echo [ERROR] Application terminated with error code %ERRORLEVEL%.
    echo =====================================================================
    pause
)

endlocal
