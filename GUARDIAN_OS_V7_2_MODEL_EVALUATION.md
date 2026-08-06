# Guardian OS V7.2 — Model Evaluation Engine

Built from the uploaded V7.1 stable source.

## Purpose

V7.2 reruns the evidence for the saved core-behaviour model against the real
participant-aware split files. It does not use historical notebook numbers as
substitutes for a reproducible evaluation.

## Inputs

Expected local files:

```text
models\v2\ablation\driver_guardian_core_behaviour.joblib
data\splits\v2\calibration.csv
data\splits\v2\test.csv
```

Absolute paths from the original DriverGuardianAI project are also accepted.

## Calculated evidence

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1 score;
- ROC-AUC where both classes are present;
- confusion matrix;
- false-positive and false-negative rates;
- condition-specific metrics;
- participant-specific metrics;
- session-specific metrics;
- probability calibration bins;
- calibration-split threshold sweep;
- exportable evaluation JSON.

## Scientific boundaries

- These metrics describe the supplied saved model and held-out split.
- Condition metrics evaluate robustness; they do not recognise glasses or hats.
- The evaluation reproduces the trained model, not the live personal-calibration
  and temporal decision layer.
- Participant groups with few rows should not be over-interpreted.
- Production deployments should keep Research Lab disabled.

## Protected stable systems

This package does not include or modify:

- `app/services/live_monitoring_service.py`
- `app/templates/monitoring.html`
- `app/static/js/monitoring.js`
- `app/services/persistent_calibration.py`
- `app/services/driver_profile_service.py`
- camera Start/Stop;
- fatigue alerts and beep;
- Commander;
- reports;
- weather;
- Edge Memory.

## Installation

1. Stop Guardian OS.
2. Extract this ZIP into the repository root.
3. Replace matching files.
4. Restart with the known Anaconda interpreter.
5. Press `Ctrl+F5`.
6. Open `/research-lab`.
7. Enter the real model, calibration split and test split paths.
8. Run the held-out evaluation.
