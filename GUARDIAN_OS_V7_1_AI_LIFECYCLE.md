# Guardian OS V7.1 — AI Lifecycle Dashboard

This release upgrades Research Lab from a dataset explorer into an AI lifecycle dashboard.

It documents five separate stages: dataset evidence, model training protocol, held-out validation controls, live calibrated decision logic, and aggregate deployment evidence.

## Public deployment

Disable research tools on a public driver-facing server:

```powershell
$env:GUARDIAN_DEPLOYMENT_MODE = "production"
$env:GUARDIAN_RESEARCH_ENABLED = "false"
```

The production application should not publish raw rows, participant IDs, session IDs, local dataset paths, or downloadable audits.

## Scientific boundaries

- Mutual information is dataset association, not model feature importance.
- Local session reports are deployment observations, not held-out validation.
- The source ZIP intentionally excludes model binaries and full datasets.
- Monitoring, calibration, alerts and camera operation are unchanged.
