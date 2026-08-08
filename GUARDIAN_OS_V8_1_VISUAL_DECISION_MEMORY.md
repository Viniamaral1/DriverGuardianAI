# Guardian OS V8.1 — Visual Intelligence & Decision Memory

## Added

### Live Visual Intelligence
Four live trend visualisations on the Intelligence page:
- advisory risk + decision confidence;
- current EAR + personal baseline;
- yawn + head-pose evidence;
- signal quality + image quality.

The browser keeps roughly the latest two minutes of the current visible live
stream. These graphs are explanatory only.

### Guardian Decision Memory
Decision Memory writes a research trace under:

`guardian_data/decision_memory/`

Each observed session stores:
- EAR and personal baseline;
- yawn/head tilt;
- raw model probability;
- existing V3 personalized/smoothed probabilities;
- V8 advisory risk;
- risk band;
- decision confidence;
- signal and image quality;
- environmental context;
- dominant evidence;
- recorded recommendation;
- alert count and temporal state.

### Replay
A saved Decision Memory session can be selected and replayed with a slider.
The replay chart shows advisory risk, confidence and the unmodified model
probability. The selected point shows the corresponding EAR, context, dominant
evidence and recommendation.

### Session comparison
Two Decision Memory traces can be compared using:
- average advisory risk;
- peak advisory risk;
- average confidence;
- average EAR;
- alert-count delta.

### Research export
Each Decision Memory session can be exported as:
- JSON (complete evidence ledger);
- CSV (tabular research trace).

## Important data-quality boundary

Decision Memory is intentionally attached to the read-only Intelligence polling
path instead of the stable camera thread.

Therefore:
- it records while Guardian Intelligence is open/polling;
- it is NOT guaranteed to contain every camera frame;
- it should be described as an observed advisory trace, not complete journey
  telemetry.

This design protects the stable V3 Monitoring/Alert pipeline.

## Protected / unchanged
- live_monitoring_service.py
- monitoring.js
- monitoring.html
- PersonalCalibration
- TemporalStateEngine
- AlertManager
- camera Start/Stop
- trained model
- alerts/beep
- Commander
- Reports

A later version can move sampling into an independent telemetry worker only
after this visualization/memory design is validated.
