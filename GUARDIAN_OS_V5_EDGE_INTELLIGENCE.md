# Guardian OS V5 — Edge Intelligence

This update was built from the uploaded stable v4.3 source.

## Safety boundary

The following stable systems were not modified:

- `app/templates/monitoring.html`
- `app/static/js/monitoring.js`
- `app/services/live_monitoring_service.py`
- camera start and stop
- calibration
- trained model and V3 decision logic
- alert beep and controlled alerts
- existing reports and PDF generation

## New features

### Guardian Edge Memory

- Imports completed JSON session reports into a local memory index.
- Persists summaries in `guardian_data/edge_memory.json`.
- Stores no raw camera video.
- Can rebuild the index from `reports/v3`.
- Calculates aggregate patterns:
  - stored session count;
  - total and average duration;
  - average and highest risk;
  - total alerts;
  - alert-session rate;
  - average baseline EAR;
  - highest-risk time period;
  - latest-session summary.

### Offline sync queue

Every newly discovered session is placed in a local pending queue. Monitoring
does not depend on the queue. The page can export the complete local bundle for
future cloud integration.

### Journey context

The Edge page can store local context separately from the fatigue model:

- weather;
- road condition;
- external light;
- cabin light;
- glasses, sunglasses or hat;
- optional notes.

These fields enrich explanations but do not directly classify fatigue.

### Commander memory

Commander can now answer:

- “What have you learned about me?”
- “What do you remember?”
- “Tell me about my last session.”
- “Does Guardian work offline?”
- “How many records are waiting to sync?”
- “What are the current weather and lighting conditions?”

## Install

Stop Guardian OS, extract this ZIP into the repository root, and replace
matching files. Restart with:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

Open `http://127.0.0.1:8010/edge` and press `Ctrl+F5`.

## Test

1. Open Edge Intelligence.
2. Click Refresh local memory.
3. Confirm existing reports appear as sessions.
4. Save journey context.
5. Download the memory bundle.
6. Ask Commander: “What have you learned about me?”
7. Start and stop Monitoring to confirm the camera remains unchanged.
