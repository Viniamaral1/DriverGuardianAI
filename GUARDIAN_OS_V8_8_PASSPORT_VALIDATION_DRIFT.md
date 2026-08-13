# Guardian OS V8.8 — Passport Validation & Drift Management

V8.8 adds a retrospective trust layer for the Personal AI Calibration Passport.

States:
- VALID
- WATCH
- DRIFT DETECTED
- RECALIBRATION RECOMMENDED

Validation signals:
- recent reliable EAR versus saved baseline;
- recent head-pose versus saved baseline;
- V8.6 perception-confidence and INSUFFICIENT rate;
- quick-verification result and historical fallback rate;
- camera-index change;
- calibration age;
- amount of recent Decision Memory evidence.

The assessment is advisory only. It does not rewrite calibration, change the
trained model, alter alert thresholds, identify the driver biometrically or
classify poor visibility as fatigue.

The Profiles page now shows the Passport state, drift score, recent evidence,
calibration age, individual validation factors and a recommended action.
Validation can be re-run on demand and is otherwise derived from existing local
history.
