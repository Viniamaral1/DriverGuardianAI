# Guardian OS V6.1.2 — Profile Sync Fix

Fixes the selected profile appearing as Guest on Monitoring.

Changed only:
- app/services/app_state.py
- app/services/live_monitoring_service.py

Monitoring now reads the active profile during standby, startup and after stop.
The camera worker, camera opening, calibration logic, alerts and frontend controls
remain unchanged.
