# Guardian OS V6 — Explainable and Predictive Intelligence

This update was built from the uploaded GuardianOS V5 source.

## Protected stable core

The package does not include or modify:

- `app/templates/monitoring.html`
- `app/static/js/monitoring.js`
- `app/services/live_monitoring_service.py`
- the trained model
- calibration
- camera start/stop
- alert beep logic
- automatic report generation

## Added

### Guardian Intelligence page

Open:

```text
http://127.0.0.1:8010/intelligence
```

It provides:

- explainable live risk contributions;
- live perception-quality summary;
- current calibration versus historical average;
- transparent near-term fatigue outlook;
- automatic local time-of-day and external-light context;
- personal Edge Memory patterns;
- journey-context caution;
- clear safety and method boundaries.

### Commander upgrades

Commander can answer:

- `Why did you alert me?`
- `Explain my current risk.`
- `What is the fatigue outlook?`
- `Should I stop soon?`
- `What is the signal quality?`
- `Compare my baseline with history.`
- `What time-of-day context is active?`

## Important boundary

The near-term outlook is a transparent rule-based layer. It combines the
existing live risk, decision probability, session duration, alert count and
historical average risk. It is not a newly trained medical or automotive
prediction model.

Automatic context currently covers local time period and a time-derived
external-light estimate. Weather, cabin lighting and occlusion remain the
manual context already stored by Guardian Edge. Camera-derived lighting and
occlusion detection should be a later, separately tested perception milestone.

## Installation

1. Stop Guardian OS with `Ctrl+C`.
2. Back up the current application.
3. Extract this ZIP into the repository root and replace matching files.
4. Restart:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

5. Open `http://127.0.0.1:8010`.
6. Press `Ctrl+F5`.
7. Test Monitoring first to confirm the protected core remains unchanged.
8. Open Guardian Intelligence.
