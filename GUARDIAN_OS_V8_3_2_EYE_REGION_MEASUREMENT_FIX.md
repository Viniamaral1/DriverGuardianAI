# Guardian OS V8.3.2 — Eye-Region Measurement Fix

## Confirmed root cause

V8.3.1 diagnostics showed:

- automatic_occlusion = uncertain
- automatic_occlusion_confidence = 0.45
- eye_visibility_score = 0.45
- eye_region_brightness_ratio = 0.0
- eye_dark_ratio = 0.0
- eye_edge_density = 0.0

The review confirmed that MediaPipe landmarks and the camera frame were already
being passed correctly to EnvironmentalPerceptionService.

The zero values came from an exposure guard: whenever the overall frame was
classified as too_dark, too_bright or glare, the service returned `uncertain`
before extracting and measuring the eye regions.

## V8.3.2 fix

Guardian now:

1. extracts the left/right eye ROIs;
2. extracts cheek reference regions;
3. calculates eye/face brightness ratio;
4. calculates dark-pixel ratio;
5. calculates edge density;
6. stores those measurements even when global exposure is imperfect;
7. keeps the final occlusion label `uncertain` when exposure is unreliable.

This deliberately separates:

MEASUREMENT
from
CLASSIFICATION

so difficult lighting no longer destroys the diagnostic evidence.

## Important

Sunglasses thresholds have NOT been loosened.

After V8.3.2, normal / clear-glasses / sunglasses sessions should finally
produce real non-zero eye-region measurements. Those measurements will be used
to decide whether the existing thresholds need tuning.

## Protected and unchanged

- live_monitoring_service.py
- trained fatigue model
- PersonalCalibration
- TemporalStateEngine
- AlertManager
- camera ownership
- beep / voice alerts
- Decision Memory lifecycle
