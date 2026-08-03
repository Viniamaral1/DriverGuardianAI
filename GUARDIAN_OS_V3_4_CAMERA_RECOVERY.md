# Guardian OS V3.4 — Camera Recovery

This patch replaces the broken Monitoring frontend with the exact camera
lifecycle used by the last working Dashboard implementation.

## Scope

- No backend monitoring code changed.
- No model code changed.
- No report code changed.
- No new camera abstraction was introduced.

## Camera flow

First use:

1. Click `Start camera`.
2. Read the consent notice.
3. Tick the consent checkbox.
4. Click `Accept and start camera`.
5. The page calls `/api/monitoring/start` immediately.

Later uses:

1. Click `Start camera`.
2. The camera starts immediately because consent is already stored locally.

The frontend uses:

```javascript
DG.post("/api/monitoring/start")
DG.post("/api/monitoring/stop")
```

which is the same lifecycle that previously worked on the Dashboard.

## Install

Stop Guardian OS, extract this ZIP into the repository root, replace matching
files, then restart with:

```powershell
& "C:\Users\Vinic\anaconda3\python.exe" `
  -m uvicorn app.main:app `
  --host 127.0.0.1 `
  --port 8010
```

Open `http://127.0.0.1:8010/monitoring` and press `Ctrl+F5`.

To force the consent notice to appear again:

```javascript
localStorage.removeItem("guardian-camera-consent")
```
