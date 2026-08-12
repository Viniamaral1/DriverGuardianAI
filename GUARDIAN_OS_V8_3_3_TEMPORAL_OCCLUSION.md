# Guardian OS V8.3.3 — Temporal Occlusion Stabilisation

V8.3.2 proved that the eye-region measurements and raw sunglasses detector work,
but individual frames oscillated between none, uncertain, eye_occlusion and
sunglasses.

V8.3.3 separates raw frame-level evidence from the stable context label.

Key behaviour:
- raw_automatic_occlusion = what the current analysed frame says;
- automatic_occlusion = temporally stabilised context used by Intelligence;
- sunglasses requires persistent direct sunglasses evidence before promotion;
- one bright/reflection frame cannot immediately clear established occlusion;
- uncertain frames do not instantly erase an already-established physical state;
- returning to none requires stronger persistence (hysteresis);
- Decision Memory stores both raw and stable labels for audit/replay.

This version does not change the sunglasses image thresholds, trained fatigue
model, PersonalCalibration, TemporalStateEngine, AlertManager, camera ownership
or alert/beep/voice behaviour.
