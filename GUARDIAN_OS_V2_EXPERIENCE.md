# Guardian OS V2 Experience Update

This package builds on the confirmed-working Guardian OS v1 live pipeline.

## Included

### Visible personal calibration
- Full in-camera calibration overlay.
- Ten-second countdown and progress ring.
- Clear instruction when the face is not visible.
- Monitoring remains based on the existing V3 calibration logic.

### Embedded Commander
- Ask Commander from the live Dashboard.
- Typed commands, one-shot browser microphone, and quick questions.
- Spoken responses follow the existing Voice Responses setting.
- The full Commander page remains available for longer conversations.
- The short commands `stop`, `end`, and `cancel` now stop monitoring.

### Driver greeting
- First-run prompt asking how Commander should address the driver.
- Name is stored only in `config/web_settings.json`.
- Name can be edited later under Settings.

### Diagnostics interface
- New `/diagnostics` page.
- Camera/backend/thread/frame status.
- Model and calibration status.
- Voice status.
- Automatic report status.
- Python executable and dependency versions.
- Automatic refresh every five seconds.

### Automatic reports
- After Stop, Guardian OS runs the existing
  `generate_session_report_v3.py` in the background.
- Uses the exact Python interpreter running FastAPI.
- Reports continue to appear in `reports/v3`.
- Can be disabled in Settings.

### Live risk trend
- Lightweight, dependency-free Canvas chart.
- Displays the latest sixty calibrated risk samples.
- Shows warning and critical thresholds.

## Install

Use the VS Code PowerShell terminal.

1. Stop Guardian OS with `Ctrl+C`.
2. Go to the repository:

```powershell
cd C:\Users\Vinic\DriverGuardianAI-GitHub
```

3. Create a backup:

```powershell
Copy-Item app app-backup-before-v2-experience -Recurse
```

4. Extract this ZIP into the repository root.
5. Replace matching files.
6. Start Guardian OS with the known-good Anaconda interpreter:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

7. Open `http://127.0.0.1:8010`.
8. Press `Ctrl+F5`.

Do not use `--reload` during camera testing.

## Recommended test

1. Enter a preferred driver name.
2. Start monitoring.
3. Confirm the calibration overlay counts down.
4. Confirm the live risk trend begins after calibration.
5. Ask Commander from the Dashboard.
6. Open Diagnostics.
7. Stop monitoring.
8. Wait for the report badge to change:
   `QUEUED` → `GENERATING` → `COMPLETE`.
9. Open Reports and confirm the new report appears.

## Automatic-report requirements

The existing report generator uses packages such as pandas and matplotlib.
They should already be installed in the Anaconda environment used by the V3
project. If report generation reports a missing package, install that package
into the same environment rather than using `C:\Python313`.
