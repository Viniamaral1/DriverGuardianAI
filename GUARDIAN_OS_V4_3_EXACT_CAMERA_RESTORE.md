# Guardian OS V4.3 — Exact Camera Restore + Commander

This patch was rebuilt from the confirmed-working V3.4 camera version.

## Camera guarantee

The following functions are byte-for-byte unchanged from V3.4:

- `startCamera()`
- `stopCamera()`

The Start button now calls the V3.4 `startCamera()` function directly.

Removed completely:

- camera consent modal;
- camera-information card;
- checkboxes;
- acceptance buttons;
- camera-related local-storage checks.

## Preserved

- camera start and stop;
- live stream;
- calibration;
- fatigue alert beep;
- backend and model;
- reports and metrics.

## Added only

- Commander on/off button;
- “Voice commands on/off” confirmation;
- visible OFF/LISTENING/PROCESSING/SPEAKING state;
- “Yes?” wake-word acknowledgement;
- pause recognition while speaking;
- resume recognition afterward.

## Install

Stop Guardian OS, extract into the repository root, replace matching files,
restart with the Anaconda interpreter, then press `Ctrl+F5`.

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```
