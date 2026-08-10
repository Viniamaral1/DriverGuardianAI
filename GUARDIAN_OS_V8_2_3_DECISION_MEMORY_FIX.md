# Guardian OS V8.2.3 — Automatic Decision Memory Lifecycle Fix

## Confirmed cause
Reports/Metrics continued to record journeys, but V8.2 Decision Memory still
depended on a browser view polling Guardian Intelligence. New journeys could
therefore finish successfully without creating a new Decision Memory timestamp.

## V8.2.3 fix
Decision Memory now follows the application-level Monitoring lifecycle:

1. Monitoring starts.
2. Decision Memory immediately creates and persists a session header.
3. A lightweight daemon sampler records advisory evidence about every 2 seconds.
4. Intelligence/Decision Memory pages only read status; they do not control
   whether recording exists.
5. Monitoring stops.
6. Decision Memory finalises and saves automatically.

Guardian Intelligence no longer needs to remain open in another tab.

## Shared ownership
GuardianState owns one IntelligenceService instance. The automatic sampler,
Intelligence API and Decision Memory APIs therefore share the same active
Decision Memory session.

## Protected / unchanged
This patch does NOT modify:
- app/services/live_monitoring_service.py
- app/services/persistent_calibration.py
- app/templates/monitoring.html
- app/static/js/monitoring.js
- trained model
- PersonalCalibration
- TemporalStateEngine
- AlertManager
- camera ownership
- alerts/beeps

Decision Memory remains non-safety-critical; sample failures are caught and
cannot stop Monitoring.

## Test
Start Monitoring from the normal Monitoring page without opening Intelligence.
A new guardian_data/decision_memory/decision_*.json timestamp should appear
almost immediately. Stop Monitoring, refresh Decision Memory, and the same
session should show a final ended_at timestamp and summary.
