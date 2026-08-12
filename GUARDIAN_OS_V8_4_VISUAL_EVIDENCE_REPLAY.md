# Guardian OS V8.4 — Visual Evidence Replay

V8.4 adds privacy-first, optional event-linked visual evidence to Decision Memory.

- OFF by default in Settings.
- No continuous session video.
- Reuses the already-encoded Monitoring JPEG stream.
- Keeps a bounded six-frame ~1 Hz memory buffer.
- Saves a small burst for alerts, strong yawns, EAR/personal-baseline events,
  and risk-band transitions.
- Stores evidence locally under guardian_data/visual_evidence/<session_id>/.
- Replay shows evidence beside the risk/confidence/EAR timeline.
- Evidence can be deleted independently while preserving Decision Memory metrics.
- No second camera or second MediaPipe pipeline.

Protected and unchanged:
trained fatigue model, PersonalCalibration, TemporalStateEngine, AlertManager,
camera ownership, model thresholds, and alert/beep/voice behaviour.

Deferred:
post-event video clips and advanced sunglasses/periocular recognition.
