# Guardian OS V3.3 — Camera Consent and Reports Layout Fix

This patch addresses only the two reported problems.

## Camera flow

- Restores a one-time consent notice.
- Consent is stored locally in the browser after acceptance.
- The first flow is:
  1. Click `Start camera`.
  2. Review the notice.
  3. Select the consent checkbox.
  4. Click `Accept and start camera`.
  5. The same click immediately calls `/api/monitoring/start`.
- Later sessions start directly without showing the notice again.
- The camera start uses the same backend endpoint as the previously working
  dashboard implementation.
- Backend failures are shown directly in the camera panel and toast.

To force the consent notice to appear again, open the browser console and run:

```javascript
localStorage.removeItem("guardian-camera-consent")
```

## Reports & Metrics layout

- Reduced oversized session and metric typography.
- Prevented session names from forcing horizontal scrolling.
- Added responsive wrapping for JSON, HTML and PDF buttons.
- Improved the two-column width balance.
- Stacks the session list above the detail view on smaller screens.

## Install

1. Stop Guardian OS with `Ctrl+C`.
2. Extract this ZIP into the repository root and replace matching files.
3. Restart with:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

4. Open `http://127.0.0.1:8010`.
5. Press `Ctrl+F5`.

## Camera test

1. Open Monitoring.
2. Click `Start camera`.
3. Accept the consent notice.
4. Confirm the camera starts immediately.
5. Stop the camera.
6. Click Start again; the notice should not reappear.
