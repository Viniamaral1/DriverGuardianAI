# Guardian OS V8.3 — Conservative Automatic Occlusion Perception

## What the review found

The previous system did not contain automatic sunglasses/hat recognition.
`EnvironmentalPerceptionService` only measured brightness, contrast, glare and
blur, and its own comment explicitly said the glare proxy was not sunglasses
recognition. `EdgeMemoryService` supplied occlusion from manual context, whose
default value was `none`. That is why sunglasses sessions were stored as
`occlusion: none`.

## V8.3 changes

- Adds conservative automatic eye-region occlusion analysis using the existing
  MediaPipe face landmarks and image statistics.
- Strong bilateral dark eye coverage can be labelled `sunglasses`.
- Ambiguous reduced eye visibility is labelled `eye_occlusion` rather than
  falsely claiming sunglasses.
- Face clipping near the image border can be labelled `partial_face`.
- Exposure problems return `uncertain` so glare/low light are not confused with
  physical occlusion.
- A short history stabilises labels across frames.
- Automatic occlusion is used when Manual Override is off and confidence is
  sufficient. Manual context remains authoritative when Manual Override is on.
- Decision confidence is reduced by visibility obstruction; occlusion does NOT
  increase biological fatigue probability.
- Decision Memory receives the resolved automatic/manual occlusion label.

## Manual categories retained/expanded

`none`, `glasses`, `sunglasses`, `hat`, `glasses_and_hat`, `eye_occlusion`,
`partial_face`, `hand_or_object`, `uncertain`.

## Important limitation

This is not a general trained accessory classifier. Clear glasses, hats/caps,
and arbitrary objects cannot be identified reliably from the existing fatigue
dataset because the project does not contain labelled image/video examples for
those categories. Guardian therefore uses automatic recognition only where the
current visual evidence is defensible and keeps manual annotation for the rest.

A future dedicated occlusion model should be trained only after collecting or
sourcing labelled image/video data for each desired category.

## Protected behaviour

V8.3 does not alter the trained fatigue model, PersonalCalibration,
TemporalStateEngine, AlertManager, fatigue thresholds, or beep/voice alert
logic. The only live-pipeline edit is passing the already-computed MediaPipe
face landmarks into the read-only environmental perception service.
