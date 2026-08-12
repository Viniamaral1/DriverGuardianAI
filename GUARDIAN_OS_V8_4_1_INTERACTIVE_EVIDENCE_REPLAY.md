# Guardian OS V8.4.1 — Interactive Evidence Replay

V8.4.1 upgrades the validated V8.4 visual-evidence foundation without changing
the fatigue model, camera pipeline or event capture rules.

## Interactive replay

- Click anywhere on the Decision Memory replay graph to jump to the nearest
  recorded sample.
- Recorded events are drawn as markers directly on the replay graph.
- A horizontal event lane lists recorded yaw, alert, EAR and risk-transition
  events; clicking an event jumps directly to it.
- The existing replay slider remains available.

## Evidence playback

The existing pre-event/event JPEG burst is preserved, but now also has:

- Previous
- Play
- Pause
- Next
- current-frame counter
- clickable evidence thumbnails

Playback advances through the captured event frames at 800 ms per frame. It is
a replay of event-linked evidence images, not continuous driver video.

## Why this approach

It provides a video-like investigation workflow while preserving the V8.4
privacy/performance contract:

- no second camera
- no continuous recording
- no extra MediaPipe pipeline
- evidence remains local and deletable
- Decision Memory metrics remain independent of image deletion

## Protected / unchanged

- trained fatigue model
- PersonalCalibration
- TemporalStateEngine
- AlertManager
- camera ownership
- event capture triggers
- model thresholds
- alert/beep/voice behaviour
- Decision Memory numerical sampling
