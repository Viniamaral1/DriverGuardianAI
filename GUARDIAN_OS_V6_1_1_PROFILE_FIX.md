# Guardian OS V6.1.1 — Driver Profile Creation Fix

This focused patch changes only driver-profile creation and selection.

## Fixed

- Successful creation now shows a green success toast.
- The newly created profile appears immediately without pressing Refresh.
- The newly created profile is selected immediately.
- Duplicate names are rejected case-insensitively.
- Extra spaces and capitalisation do not create accidental duplicates.
- Existing profiles and saved calibration are never overwritten.
- The last selected profile continues to persist in `driver_profiles.json`.
- API errors now appear as clear toast messages.

Examples treated as the same profile name:

- `Abbi`
- `abbi`
- `  ABBI  `

## Unchanged

Camera, calibration, quick verification, alerts, beep, Commander, reports,
Edge Memory and Guardian Intelligence are unchanged.

## Install

Stop Guardian OS, extract this ZIP into the repository root, replace matching
files, restart Guardian OS, and press `Ctrl+F5`.
