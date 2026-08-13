# Guardian OS V8.6 — Perception Confidence Layer

V8.6 introduces a separate perception-confidence layer that answers a different
question from the fatigue model: **is the current camera evidence observable
enough to interpret?**

States:
- TRUSTED — both detected cues and non-detection can be interpreted normally.
- DEGRADED — positive cues remain useful, but missing cues should not be treated
  as strong negative evidence.
- INSUFFICIENT — visual observability is too limited to make absence claims.

Inputs reuse the existing pipeline only: face availability, image quality,
exposure/glare/sharpness, eye visibility, occlusion confidence, temporal
occlusion stability, frame rate and head-pose geometry. There is no second camera
or MediaPipe instance.

The output contains a perception score, state, observation mode, evidence policy,
reason codes, affected regions, component scores and explicit booleans describing
whether positive/negative visual evidence is trustworthy.

Decision Memory now stores the V8.6 perception score/state/reason codes for new
sessions. Near-Miss Memory uses V8.6 INSUFFICIENT states when available to detect
repeated uncertainty; legacy V8.5 sessions keep the previous fallback logic.

Safety boundary: V8.6 is explanatory metadata. It does not alter the trained
fatigue model, PersonalCalibration, TemporalStateEngine, AlertManager, model
thresholds, fatigue probabilities or alerts.


Decision Memory integration:
- new V8.6 sessions use schema `8.6-decision-memory-v6`;
- perception score/state/observation mode/trust booleans/reason codes are stored;
- perception state transitions become replay events;
- legacy Decision Memory sessions remain readable;
- Near-Miss Memory prefers V8.6 INSUFFICIENT perception states for repeated-uncertainty analysis, while older sessions retain the V8.5 fallback.
