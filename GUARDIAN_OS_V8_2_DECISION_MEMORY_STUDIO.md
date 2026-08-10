# Guardian OS V8.2 — Decision Memory & Guardian Studio

This release turns Decision Memory into a first-class module while keeping the stable Monitoring path unchanged.

## Added
- Dedicated `/decision-memory` page and navigation item.
- Session library with search/sort.
- Interactive replay slider and autoplay.
- Risk/confidence/raw-model timeline.
- Automatic event markers for risk changes, alerts, strong yaw evidence, personal EAR deviations and reduced confidence.
- Point-in-time “Why?” evidence inspector.
- Session labels, condition tags and notes.
- JSON/CSV exports.
- Guardian Studio side-by-side session comparison with an overlay risk chart.
- Aggregate Decision Memory overview.

## Performance improvement
V8.1 re-fetched the full Intelligence payload on every WebSocket state update in addition to its timer. V8.2 removes that duplicate fetch path. Decision Memory also samples defensively at no more than once every two seconds and batches disk writes. This reduces browser/API/disk work without touching camera inference.

## Safety boundary
Decision Memory remains a read-only advisory research trace. It does not modify the trained model, calibration, TemporalStateEngine, AlertManager, camera or alerts.

## Protected files not modified
- app/services/live_monitoring_service.py
- app/services/persistent_calibration.py
- app/templates/monitoring.html
- app/static/js/monitoring.js
