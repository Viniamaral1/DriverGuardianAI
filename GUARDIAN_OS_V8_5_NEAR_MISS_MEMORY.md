# Guardian OS V8.5 — Near-Miss Memory

Near-Miss Memory is a retrospective analysis layer built on recorded Decision
Memory. It does not change the trained fatigue model, camera pipeline, alert
thresholds, calibration or live Monitoring decisions.

Detected historical patterns:
- near-alert recovery;
- driver-state recovery after a high-risk period;
- escalation from lower risk into high risk;
- accumulation of multiple moderate evidence signals;
- repeated perception / decision uncertainty;
- gradual personal EAR-to-baseline drift;
- repeated episode types within the same session.

Each episode records:
- type and title;
- start, peak, end and optional recovery indices/timestamps;
- duration;
- start/peak/end advisory risk;
- analysis confidence;
- contributing Decision Memory evidence;
- outcome and explanation;
- whether event-linked Visual Evidence exists nearby.

The analysis is derived when a Decision Memory session is opened. It does not
rewrite historical samples.

The Intelligence replay now displays Near-Miss cards. Clicking a card jumps the
existing replay to the episode peak, where the graph, tooltip and Visual
Evidence system continue to work normally.
