# Guardian OS V8.3.4 — Raw Occlusion Classifier Redesign

## Confirmed V8.3.3 failure

The temporal stabiliser was operating, but the raw classifier repeatedly
returned `uncertain` even when diagnostics showed:

- eye_visibility_score ~= 1.0
- eye_dark_ratio ~= 0.0
- eye_region_brightness_ratio ~= 0.9–1.1

A previously established `partial_face` state could therefore remain latched.

## Root causes fixed

1. Global glare / exposure status overrode strong clear-eye evidence.
2. `partial_face` used the min/max of all FaceMesh points, so one noisy
   landmark close to the image boundary could produce a false partial-face
   state.

## New raw-classifier order

1. Stable facial boundary anchors -> partial_face
2. Strong clear-eye evidence -> none
3. Strong bilateral dark-eye evidence -> sunglasses
4. Generic reduced-eye visibility -> eye_occlusion
5. Ambiguous evidence under poor exposure -> uncertain
6. Otherwise -> none

`uncertain` now means insufficient evidence; it is not treated as a physical
occlusion observation.

## Partial-face rule

Boundary detection now uses stable facial anchors (forehead, chin, left/right
cheek) instead of the extrema of all 468 FaceMesh landmarks. A robust trimmed
face box is also used for cheek reference patches.

## Session reset

Temporal occlusion state is explicitly reset at the beginning of every
Monitoring session.

## Protected / unchanged

- trained fatigue model
- PersonalCalibration
- TemporalStateEngine
- AlertManager
- camera ownership
- alert/beep/voice behaviour
- Decision Memory lifecycle
- sunglasses measurement thresholds are not broadly loosened
