# Guardian OS V8.9.1 — Predictive Guardian Refinement

This refinement was made after validating V8.9 against real Decision Memory.

Historical timing verification:
- The displayed `0 min` was not caused by the wrong Passport profile.
- Real stored sessions commonly crossed the 65% advisory threshold within a few seconds.
- Across the inspected histories, the median first elevated-risk crossing was about 2 seconds.
- V8.9 rounded that correctly computed value to 0.0 minutes, which was misleading in the UI.
- Sub-minute historical timings now display in seconds.

Projection stability refinement:
- Guardian still measures and reports the real short-term live risk slope.
- Multi-minute scenario projection now limits the slope used for extrapolation to ±8 percentage points per minute.
- This prevents a very short spike from generating an implausible immediate 0–100% forecast.
- The UI explanation explicitly says when the projection slope was bounded.
- Direction detection still uses the measured live trajectory.

Unchanged safety path:
- trained fatigue model
- V8 Decision Engine
- PersonalCalibration
- Passport baseline
- Passport Validation logic
- Perception Confidence semantics
- alert thresholds / alert behaviour
- camera / MediaPipe pipeline
- Decision Memory sampling
- Visual Evidence
