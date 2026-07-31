# DriverGuardianAI V4.2 Update

This package contains only the files changed for the V4.2 polish pass.

## Fixed

- Settings no longer resets the active theme or accent when the page opens.
- The Settings controls now follow the current top-bar theme and accent.
- Added a collapsible desktop sidebar with persistent state.
- Reduced the expanded sidebar width and top-header height.
- Improved dashboard vertical sizing and Recent Activity scrolling.
- Commander hands-free mode now pauses recognition before processing.
- Commander waits up to five seconds after the wake word for a command.
- Recognition remains paused while Commander processes and speaks.
- Hands-free listening resumes after the spoken response finishes.
- Improved browser speech-synthesis voice selection and playback reliability.
- The unavailable Python backend-listener button is disabled with an explanatory tooltip.
- Added clearer listening, processing, and speaking visual states.

## Install

1. Stop FastAPI using `Ctrl+C`.
2. Back up the current web app:

```powershell
cd C:\Users\Vinic\DriverGuardianAI-GitHub
Copy-Item app app-backup-before-v4.2 -Recurse
```

3. Extract this ZIP.
4. Copy its `app` folder into the repository root.
5. Allow Windows to replace matching files.
6. Restart:

```powershell
python -m uvicorn app.main:app --reload --port 8010
```

7. Open `http://127.0.0.1:8010`.
8. Press `Ctrl+F5` once.

## Commander hands-free test

Open `/commander`, click **Start listening**, and say:

- `Commander, what is my current status?`
- `Commander, start monitoring`
- `Commander, summarise the latest report`

Commander should:

1. detect the wake word;
2. stop listening when the command is complete, or after five seconds of silence;
3. process the request;
4. speak the answer;
5. resume hands-free listening.

Chrome or Edge is recommended for browser speech recognition.
