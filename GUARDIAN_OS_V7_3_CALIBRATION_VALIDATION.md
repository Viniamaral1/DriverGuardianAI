# Guardian OS V7.3 — Personal Calibration Validation

## What the audit established

The held-out test contains paired normal and drowsy sessions for the same
held-out participants and conditions. This allows a controlled replay of the
personal calibration layer without retraining the model.

## New Research Lab experiment

The Personal Calibration Validation section:

1. Loads the unchanged core-behaviour `.joblib`.
2. Loads the held-out `test.csv`.
3. Imports the same `realtime_driver_guardian_v3_alerts.PersonalCalibration`
   class used by live Monitoring.
4. For each held-out participant + condition:
   - uses the recorded normal session to establish a 10-second / 80-sample
     personal baseline;
   - excludes those calibration rows from scoring;
   - scores the remaining normal rows;
   - reuses that personal baseline for the paired drowsy session.
5. Compares:
   - generic trained-model probability;
   - personalized V3 fused probability.
6. Reports:
   - balanced accuracy;
   - accuracy;
   - precision;
   - recall;
   - F1;
   - ROC-AUC;
   - false-positive change;
   - false-negative change;
   - results by participant and condition.

## Scientific boundary

This is a controlled calibration-effect study.

It uses the real V3 PersonalCalibration fusion, but deliberately does NOT replay:

- TemporalStateEngine;
- warning/critical dwell-time rules;
- AlertManager cooldown;
- audible alerts.

Therefore the result must not be described as final live-alert accuracy.

The experiment also uses the recorded normal-session label to define a safe
baseline session. That is a research-control decision, not an automatic
real-world driver-state detector.

## WebSocket fix

`app/routers/websocket.py` now treats page navigation and refresh as normal
disconnects. It catches the race where the browser closes between the WebSocket
state check and `send_json()`, preventing the repeated:

`Unexpected ASGI message 'websocket.send', after sending 'websocket.close'`

traceback.

## Unchanged

- `live_monitoring_service.py`
- camera Start/Stop
- live model prediction
- PersonalCalibration implementation
- saved driver profiles
- quick verification
- alert beep
- Commander
- Reports
- weather/context
- Edge Memory
- trained `.joblib`

## Expected local inputs

- `models\v2\ablation\driver_guardian_core_behaviour.joblib`
- `data\splits\v2\test.csv`

Run Guardian from the same Anaconda/Python environment where Monitoring works,
because the validation imports the standalone V3 module.
