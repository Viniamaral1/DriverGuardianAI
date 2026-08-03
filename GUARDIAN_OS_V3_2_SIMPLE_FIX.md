# Guardian OS V3.2 — Camera and Reports Simplification

This focused update fixes the two reported product issues.

## Monitoring

- Removed the acceptance modal and all remembered acceptance state.
- Added a permanent, non-blocking local-processing information card.
- `Start camera` now calls the working `/api/monitoring/start` endpoint directly.
- `Stop` continues to call `/api/monitoring/stop`.
- Backend error details are displayed when startup fails.
- Camera, calibration, alerts and Commander remain only on Monitoring.

## Reports and Metrics

- Removed the duplicate Metrics navigation item.
- Combined reports, detailed metrics and exports into one page:
  `/reports`
- `/metrics` now redirects to `/reports`, so old links still work.
- The merged page provides:
  - session list;
  - report summary cards;
  - full probability and behavioural metrics;
  - state timeline;
  - JSON download;
  - HTML report view;
  - PDF download.

## Install

1. Stop Guardian OS with `Ctrl+C`.
2. Extract this ZIP into the repository root and replace matching files.
3. Start with the working interpreter:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

4. Open `http://127.0.0.1:8010`.
5. Press `Ctrl+F5`.

## Essential test

1. Open `/monitoring`.
2. Click `Start camera` once.
3. Confirm the backend starts and the live feed appears.
4. Click `Stop`.
5. Open `Reports & Metrics`.
6. Confirm there is only one combined page and all downloads work.
