# Guardian OS V8.7 — Personal AI Calibration Passport

V8.7 adds a versioned, local-only Personal AI Calibration Passport on top of
Guardian's existing manual driver profiles.

The passport reuses:
- saved personal EAR / yawn / head-tilt calibration;
- calibration and verification history;
- Decision Memory session history;
- V8.6 perception confidence states and reason codes;
- observed visibility / occlusion conditions;
- existing Visual Evidence privacy setting.

The passport does NOT contain:
- face embeddings;
- biometric identity templates;
- raw video;
- cloud identifiers;
- model weights.

Features:
- stable anonymous Passport ID per local profile;
- versioned schema `guardian-calibration-passport-v1`;
- local baseline and calibration-history summary;
- perception reliability summary;
- common visibility limitations / sunglasses observations;
- privacy controls for export and perception-history inclusion;
- JSON export;
- explicit JSON import into a selected profile;
- Passport metadata reset without deleting the driver's calibration.

Import safety:
- Monitoring must be stopped before import;
- schema is validated;
- baseline EAR is range checked;
- import replaces only the selected profile's saved behavioural baseline;
- trained model weights and alert thresholds are untouched.

Cross-device/cross-vehicle validation is deliberately deferred to V8.8.
