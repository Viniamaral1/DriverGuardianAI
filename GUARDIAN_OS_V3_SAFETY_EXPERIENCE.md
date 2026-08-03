# Guardian OS V3 — Safety Experience Update

This package contains only the agreed product changes built on top of the
working Guardian OS V2 live camera integration.

## Added

### Dedicated Live Monitoring page

- New `/monitoring` operational camera console.
- Privacy and local-processing notice before the camera starts.
- Live camera, calibration, risk, biometrics, Commander and trend graph.
- Direct links to Metrics and Diagnostics.

### Multimodal fatigue alerts

- Browser alert tones for WARNING and CRITICAL state transitions.
- Concise Commander safety message for elevated fatigue.
- Visible safety alert banner.
- Mute/unmute alert-tone control.
- Stop Commander speech control.

### Metrics and PDF exports

- New `/metrics` page for generated sessions.
- Detailed probability, behavioural and state-duration metrics.
- JSON, HTML and PDF export options.
- PDF contains metrics and summaries only; raw camera frames are excluded.

### Commander control

- Stop-speaking button on the Dashboard.
- Stop-speaking button on the full Commander page.
- Stop-speaking button inside Live Monitoring.

## Not included in this update

The following were discussed but should remain separate, testable milestones:

- explicit glasses/hat classifier;
- cabin brightness, glare and blur scoring;
- weather service;
- simulation mode;
- offline Edge Memory and cloud synchronisation.

Your existing dataset already includes occlusion conditions. The next research
step should be an occlusion performance audit before changing the fatigue model.

## Installation

From the VS Code PowerShell terminal:

```powershell
cd C:\Users\Vinic\DriverGuardianAI-GitHub
```

Stop Guardian OS using `Ctrl+C`.

Back up the current web application:

```powershell
Copy-Item app app-backup-before-v3-safety -Recurse
```

Extract this ZIP into the repository root and replace matching files.

Install the added PDF dependency using the same Anaconda Python environment
that passed the camera diagnostic:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" -m pip install "reportlab>=4.2"
```

Start Guardian OS:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

Open `http://127.0.0.1:8010` and press `Ctrl+F5`.

## Test sequence

1. Open **Monitoring**.
2. Read and accept the local-processing notice.
3. Confirm the camera starts and calibration is visible.
4. Simulate a WARNING and CRITICAL condition safely while seated.
5. Confirm the tone, visual banner and concise Commander warning.
6. Press **Stop Commander** while speech is active.
7. Test **Mute alerts** and **Unmute alerts**.
8. Stop the session and wait for the automatic report.
9. Open **Metrics**.
10. Select the latest session and download JSON, HTML and PDF.
11. Confirm the PDF opens and contains no raw camera image.

## Important safety note

Do not test fatigue behaviour while driving on a public road. Use controlled
desk-based simulation or prerecorded material.
