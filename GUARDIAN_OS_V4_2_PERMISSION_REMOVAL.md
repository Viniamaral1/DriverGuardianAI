# Guardian OS V4.2 — Permission UI Removal

This small patch removes only the interactive camera-information feature that
was interfering with testing.

## Removed

- Camera information card inside the camera area
- `I understand` button
- Local-storage acknowledgement logic

## Unchanged

- Working Start Camera code
- Working Stop code
- Camera stream
- Calibration
- Fatigue alert beep
- Commander on/off
- Commander wake word and visible voice states
- Backend, model, reports and metrics

## Install

Stop Guardian OS, extract this ZIP into the repository root, replace the two
matching files, restart Guardian OS, and press `Ctrl+F5`.

Start command:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```
