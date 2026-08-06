# Guardian OS V7.2.1 — Evaluation JSON Fix

The held-out evaluation completed, but some subgroup ROC-AUC values were
undefined because those groups contained only one target class. Scikit-learn
returned NaN and FastAPI rejected it during strict JSON serialization.

Fixed:
- undefined and non-finite values become JSON null;
- valid metrics remain unchanged;
- saved evaluations use strict JSON;
- previously saved NaN results can load safely;
- the browser handles non-JSON error responses safely.

Camera, Monitoring, fatigue inference, calibration, profiles, alerts, Commander,
reports, weather and Edge Memory are unchanged.
