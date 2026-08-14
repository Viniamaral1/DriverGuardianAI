# Guardian OS V8.9 — Predictive Guardian

V8.9 adds a transparent near-term advisory forecasting layer.

Inputs:
- current V8 advisory risk;
- recent live risk trajectory;
- current session duration;
- local Decision Memory escalation timing;
- same-time-of-day historical peaks;
- historical alert/recovery patterns;
- V8.6 Perception Confidence;
- V8.8 Passport Validation trust.

Outputs:
- STABLE / RISING / FALLING / UNCERTAIN direction;
- forecast confidence;
- near-term advisory projection;
- estimated elevated-risk timing only when justified;
- historical pattern summary;
- contributing factors;
- withholding reasons;
- recommended advisory action.

Trust controls:
- forecast is withheld while Monitoring is off or calibration is incomplete;
- personalised forecast is withheld when Perception is INSUFFICIENT;
- personalised forecast is withheld when Passport state is DRIFT DETECTED or
  RECALIBRATION RECOMMENDED;
- forecast is withheld when the current session is too short;
- profiles with insufficient Decision Memory history are not given a confident
  personalised forecast;
- WATCH Passport and DEGRADED perception cap prediction confidence.

The projection is an explainable advisory scenario, not a medical probability.
It never changes the trained fatigue model, calibration, V8 Decision Engine or
existing alert path.
