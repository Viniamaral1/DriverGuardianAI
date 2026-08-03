# Guardian OS V3.1 Dashboard and Monitoring Fix

This focused patch corrects the two reported issues.

## Fixed

- Dashboard no longer contains or starts the camera.
- Dashboard is now a welcome, system-status and navigation hub.
- Live camera, calibration and safety controls exist only on Monitoring.
- The Monitoring privacy notice no longer blocks the page on arrival.
- Pressing `Review & start camera` opens the notice.
- The camera starts only after the acknowledgement checkbox is selected.
- Added an explicit Cancel button to close the privacy notice.
- Made Monitoring controls more resilient to a missing optional element.
- Preserved the working V3 camera adapter, alert logic, reports and model.

## Install

1. Stop FastAPI with `Ctrl+C`.
2. Copy this ZIP into the repository root and replace matching files.
3. Restart with the working Anaconda interpreter:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

4. Open `http://127.0.0.1:8010`.
5. Press `Ctrl+F5`.

## Test

- Dashboard contains no camera feed.
- `Open live monitoring` opens `/monitoring`.
- Monitoring loads without a blocking modal.
- `Review & start camera` opens the local-processing notice.
- Selecting the checkbox enables `Start local camera`.
- Cancel closes the notice without starting the camera.
- Start and Stop continue to control the working V3 adapter.
