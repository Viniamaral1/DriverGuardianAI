# Guardian OS V8.0 — Advisory AI Decision Engine

V7.3 showed that directly replacing the trained model probability with the
existing personal-calibration fusion reduced held-out recall. V8 therefore
changes the architecture instead of tuning coefficients until the result looks
better.

V8 adds a separate advisory layer to Guardian Intelligence:

- trained core-behaviour model probability: PRIMARY evidence;
- personal EAR deviation: bounded SUPPORTING evidence;
- yawn evidence: supporting evidence;
- head-pose evidence: supporting evidence;
- image/camera quality: confidence evidence;
- weather/road/light/occlusion: caution guidance only;
- short decision timeline: explainability.

The V8 score is an advisory risk index, NOT a calibrated probability.

V8 does not modify:
- live_monitoring_service.py;
- PersonalCalibration;
- the trained model;
- TemporalStateEngine;
- AlertManager;
- camera Start/Stop;
- alerts/beeps;
- profiles;
- Commander;
- Reports.

The existing V3 monitoring path remains authoritative until V8 has its own
independent validation.
