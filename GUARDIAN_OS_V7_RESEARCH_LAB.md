# Guardian OS V7.0 — AI Research Lab

Built from the stable V6.3.3 source and the uploaded
`driver_guardian_v2_clean.csv`.

## Evidence confirmed

The uploaded cleaned dataset contains:

- 39,426 rows
- 17 columns
- 9 participant identifiers in the cleaned file
- 45 sessions
- conditions: none, glasses, dark and hat
- state labels: normal and drowsy
- no missing cells
- no completely duplicated rows

The earlier notebook indicates that participant aliases were later normalised
when participant-aware train/calibration/test splits were created. Research Lab
therefore reports the cleaned file exactly as stored and does not silently
rewrite participant identities.

## Added

Open:

```text
http://127.0.0.1:8010/research-lab
```

The page can audit a cleaned CSV and displays:

- participant, session, state, condition and fatigue-level distributions
- missing cells and duplicate rows
- condition-specific signal summaries
- dataset feature-to-label mutual-information association
- research readiness for participant-aware evaluation, condition robustness,
  sequence research, forecasting and image-based occlusion models
- exportable JSON results
- explicit scientific boundaries

## Important distinction

Mutual information describes association inside the dataset. It is not the
trained model's feature importance and is not causal explainability.

The current cleaned CSV can support condition robustness research. A true live
glasses/hat classifier still requires raw images, video frames or image paths
paired with condition labels.

## Dataset path

The default project-relative path is:

```text
data\processed\driver_guardian_v2_clean.csv
```

You may also enter the original absolute path.

## Protected stable systems

This package does not include or modify:

- monitoring.html
- monitoring.js
- live_monitoring_service.py
- persistent_calibration.py
- driver_profile_service.py
- environmental perception
- automatic context/weather
- Commander
- reports
- alerts
- trained model files

## Install

Stop Guardian OS, extract the ZIP into the repository root, replace matching
files, restart with the working Anaconda interpreter and press Ctrl+F5.

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

## Test

1. Confirm Monitoring still starts, calibrates, alerts and stops.
2. Open AI Research Lab.
3. Enter the cleaned dataset path.
4. Run the audit.
5. Confirm 39,426 rows and 45 sessions are reported.
6. Confirm four conditions appear.
7. Export the latest JSON.
