# Guardian OS V8.3.1 — Occlusion Pipeline Diagnostics

The V8.3 review confirmed that live MediaPipe landmarks are passed into the
environmental perception service and the resulting automatic occlusion fields
are published into Monitoring metrics.

The missing piece was observability. Edge showed manual/default occlusion,
Intelligence showed only the final resolved label, and Decision Memory stored
only that final label.

V8.3.1 therefore exposes and records:
- automatic occlusion label and confidence
- eye visibility
- eye/face brightness ratio
- eye dark ratio
- eye edge density
- manual occlusion
- resolved occlusion
- occlusion source

Detector thresholds are intentionally unchanged until normal / clear-glasses /
sunglasses diagnostic measurements are collected.

Protected and unchanged:
- live Monitoring/camera ownership
- EnvironmentalPerceptionService thresholds
- trained fatigue model
- PersonalCalibration
- TemporalStateEngine
- AlertManager
- beep/voice alerts
