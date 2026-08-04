# Guardian OS V6.1 — Persistent Driver Profiles

This update was built from the uploaded GuardianOS V6 source.

## Purpose

Guardian can now reuse a known driver's saved personal baseline without blindly
trusting an old calibration.

### New profile

- Original full calibration: 10 seconds
- Baseline saved locally after successful calibration

### Known profile with saved calibration

- Quick verification: 3 seconds
- Current neutral EAR is compared with the saved baseline
- Match tolerance: 12 percent EAR difference in this research build
- Head-angle verification: within 8 degrees
- Different camera index: full calibration required
- Matching profile: monitoring continues immediately
- Mismatch: automatic fallback to the original full 10-second calibration

The 12 percent verification tolerance is a starting engineering value and
should later be validated using the participant dataset.

## Privacy design

Profiles are selected manually.

Guardian does not store:

- face embeddings;
- facial identity templates;
- raw video;
- biometric recognition data.

The profile file contains behavioural calibration values and verification
metadata only:

```text
guardian_data/driver_profiles.json
```

## New page

Open:

```text
http://127.0.0.1:8010/profiles
```

The page supports:

- creating a driver profile;
- selecting the active driver;
- Guest mode;
- viewing the saved baseline;
- resetting calibration;
- deleting a profile;
- viewing quick-verification and fallback counts.

## Monitoring changes

The proven camera Start and Stop functions are unchanged.

Monitoring now displays:

- active profile name;
- quick verification countdown;
- full recalibration message after a mismatch;
- calibration mode and status through the existing metrics stream.

## Commander questions

Try:

- `Which driver profile is active?`
- `Is quick calibration ready?`
- `What is the calibration mode?`
- `How many seconds remain in calibration?`

## Installation

1. Stop Guardian OS with `Ctrl+C`.
2. Back up the current application.
3. Extract this ZIP into the repository root and replace matching files.
4. Restart:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

5. Open `http://127.0.0.1:8010`.
6. Press `Ctrl+F5`.

## Test sequence

1. Open Driver Profiles.
2. Create and select a profile.
3. Start Monitoring.
4. Confirm the original full calibration runs.
5. Stop Monitoring.
6. Return to Driver Profiles and confirm a saved baseline appears.
7. Start Monitoring again.
8. Confirm the three-second verification appears.
9. Verify calibration completes quickly when the camera position is similar.
10. Change the camera or seating angle significantly and verify Guardian falls
    back to the full calibration.
11. Select Guest mode and verify the original full calibration always runs.

## Stable systems retained

- camera opening and release;
- MJPEG camera stream;
- model inference;
- temporal state engine;
- alert beep and cooldown;
- Commander voice controls;
- reports;
- Edge Memory;
- Guardian Intelligence.
