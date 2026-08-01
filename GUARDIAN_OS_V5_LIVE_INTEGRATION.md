# Guardian OS V5 — Live DriverGuardian Integration

This update replaces the simulated Guardian OS telemetry with the existing
`realtime_driver_guardian_v3_alerts.py` pipeline.

## What is connected

- Real webcam capture
- MediaPipe Face Mesh
- EAR, yawn score, and head-tilt extraction
- Existing trained model with heuristic fallback
- Personal 10-second calibration
- Existing calibrated decision probability
- Existing temporal state engine
- Existing controlled alert manager
- Live JPEG camera stream inside Guardian OS
- Real WebSocket metrics for Dashboard and Commander
- Automatic V3 frame logging to `logs/v3/`
- Rolling blink-rate estimate derived from eye-closure transitions

## Important prerequisites

The local repository must still contain:

```text
realtime_driver_guardian_v3_alerts.py
models/v2/ablation/driver_guardian_core_behaviour.joblib
```

The service can run with the V3 heuristic fallback if the model cannot load,
but the repository's trained model is recommended.

## Installation

1. Stop FastAPI using `Ctrl+C`.
2. Back up the current application:

```powershell
cd C:\Users\Vinic\DriverGuardianAI-GitHub
Copy-Item app app-backup-before-v5-live -Recurse
```

3. Extract this ZIP.
4. Copy its `app` folder into the repository root and replace matching files.
5. Replace `requirements.txt`.
6. Install or confirm dependencies:

```powershell
python -m pip install -r requirements.txt
```

7. Start Guardian OS:

```powershell
python -m uvicorn app.main:app --reload --port 8010
```

8. Open `http://127.0.0.1:8010` and press `Ctrl+F5`.

## Test sequence

1. Close Jupyter, OpenCV windows, Teams, Zoom, or anything else using the webcam.
2. Click **Start monitoring**.
3. Allow approximately 10 seconds for personal calibration.
4. Confirm the real camera appears inside the dashboard.
5. Confirm EAR, yawn score, head tilt, FPS, risk, and state update.
6. Open Commander and ask: `What is my current status?`
7. Click **Stop** and confirm the camera releases.

## Troubleshooting

### Camera cannot open

Close other camera applications, verify Settings → Camera index is `0`, and
restart FastAPI.

### Model shows HEURISTIC

Confirm this file exists:

```text
models/v2/ablation/driver_guardian_core_behaviour.joblib
```

### Camera is delayed

Use a lower webcam resolution or close other CPU-heavy programs. The stream
uses JPEG quality 82 and discards old camera buffers where supported.

## Scope

This update connects the existing V3 pipeline rather than rewriting it.
Automatic HTML-report generation after Stop is intentionally left for the next
milestone; this version records the frame-level CSV needed by the existing
report generator.
