# DriverGuardianAI V4.1 Frontend Update

This package contains only the files changed for the V4.1 frontend polish pass.

## Fixes and improvements

- Removed bottom clipping caused by fixed viewport overflow.
- Added correct scrolling on Commander, Reports, and Settings pages.
- Restored full date and time in the header.
- Fixed dashboard Start/Stop state updates and button disabling.
- Confirmed the dashboard Commander button opens `/commander`.
- Added continuous browser wake-word mode: say “Commander” followed by a command.
- Kept the optional Python/backend wake-word controls.
- Added listening visualisation and improved Commander page sizing.
- Added a more visual Reports page with summary cards, a risk ring, KPIs, and timeline.
- Added live theme previews and reliable light/dark/accent persistence.
- Improved responsive behaviour for short laptop screens and mobile widths.

## Installation

1. Stop the FastAPI server with `Ctrl+C`.
2. Back up the existing app folder:

```powershell
Copy-Item app app-backup-v4-before-frontend-fix -Recurse
```

3. Copy the `app` folder from this update into the repository root and allow Windows to replace matching files.
4. Start the application:

```powershell
python -m uvicorn app.main:app --reload --port 8010
```

5. Open `http://127.0.0.1:8010`.
6. Press `Ctrl+F5` once to clear old CSS and JavaScript from the browser cache.

## Hands-free Commander

Open `/commander`, click **Start listening**, approve microphone access, and say:

- “Commander, start monitoring”
- “Commander, what is my current status?”
- “Commander, summarise the latest report”

This browser mode works only while the Commander page is open. Chrome or Edge is recommended.
