# Guardian OS V8.9.2 — Calibration / Passport Trust Fix

Root cause confirmed from the supplied debug checkpoint:

- fresh full calibration was persisted correctly;
- recent reliable EAR was close to the newly saved baseline;
- Perception Confidence was generally healthy;
- Passport Validation was still applying a failed quick-verification that occurred
  before the replacement full calibration;
- cumulative historical fallback counts were also being treated as though they
  belonged to the new calibration.

This made a freshly recalibrated Passport immediately appear DRIFT DETECTED or
RECALIBRATION RECOMMENDED even though its current behavioural baseline matched
recent observations.

V8.9.2 changes the evidence lifecycle, not the drift thresholds:

1. Reset saved calibration clears the obsolete last-verification result.
2. Completing a full calibration clears any mismatch that the new baseline resolves.
3. Calibration records snapshot verification/fallback counters at creation time.
4. Passport Validation only evaluates quick-verification evidence newer than the
   current calibration.
5. Fallback rate is calculated from checks performed since the current calibration.
6. Legacy profiles without counter snapshots do not have historical fallback totals
   incorrectly attributed to a newer baseline when timestamps prove the old
   verification predates it.
7. Passport import receives the same calibration-generation semantics.

Unchanged:
- EAR/head-pose drift thresholds;
- trained fatigue model;
- V8 Decision Engine;
- Perception Confidence;
- Predictive Guardian withholding rules;
- alert thresholds/behaviour;
- camera/MediaPipe ownership;
- Decision Memory sampling;
- Visual Evidence.
