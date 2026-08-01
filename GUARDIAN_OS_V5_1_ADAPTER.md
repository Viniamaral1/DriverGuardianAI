# Guardian OS V5.1 — Reliable V3 Adapter

This update keeps the existing `realtime_driver_guardian_v3_alerts.py`
algorithms intact. It changes only the lifecycle adapter that opens/releases
the camera and publishes V3 values to Guardian OS.

## Main fixes

- Opens and validates the webcam before loading the model.
- Tries DirectShow, Media Foundation, and the default Windows backend.
- Requires an actual camera frame before reporting CONNECTED.
- Keeps Stop enabled while the service is STARTING.
- Releases `VideoCapture` immediately when Stop is pressed, unblocking Windows.
- Reports STARTING, STOPPING, camera backend, thread state, and detailed errors.
- Uses the V3 heuristic fallback when a persisted model cannot predict.
- Adds `/api/monitoring/diagnostics`.
- Includes a standalone camera/environment diagnostic using the exact Python
  executable that will run Guardian OS.

## Install

In the VS Code PowerShell terminal:

```powershell
cd C:\Users\Vinic\DriverGuardianAI-GitHub
```

Stop FastAPI with `Ctrl+C`, then back up the current adapter:

```powershell
Copy-Item app\services\live_monitoring_service.py `
  app\services\live_monitoring_service-before-v5.1.py
```

Extract this ZIP into the repository root and replace matching files.

## Required first test

Run the included diagnostic from the same VS Code terminal:

```powershell
python diagnose_guardian_camera.py --camera 0
```

The final lines will show the exact command to run Guardian OS with the same
Python executable. Use that command.

Example:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Do **not** use `--reload` during webcam testing. The reload process can create a
second application process and complicate camera ownership.

## Test Guardian OS

1. Open `http://127.0.0.1:8010`.
2. Press `Ctrl+F5`.
3. Close Jupyter/OpenCV/Teams/Zoom before pressing Start.
4. Click **Start monitoring**.
5. Confirm the event log identifies the selected camera backend.
6. Wait for calibration.
7. Click **Stop** and confirm the camera light turns off.

## Diagnostics endpoint

While FastAPI is running, open:

```text
http://127.0.0.1:8010/api/monitoring/diagnostics
```

It reports:

- `camera_status`
- `camera_backend`
- `thread_alive`
- `frame_available`
- `starting`
- `stopping`
- `error`
- `model_warning`

## scikit-learn warning

Your model was persisted with scikit-learn 1.7.1. For reproducible model
predictions, run Guardian OS in the same environment used by the working V3
notebook, or install:

```powershell
python -m pip install "scikit-learn==1.7.1"
```

The adapter still starts the camera and uses the existing V3 heuristic fallback
if model prediction is unavailable.
