# DriverGuardianAI V4 Web Application

## Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8010
```

Open `http://127.0.0.1:8010`.

## Pages

- `/` compact live dashboard
- `/commander` typed, browser-microphone, and optional backend wake-word controls
- `/reports` session report browser
- `/settings` persistent theme, audio, camera, and sensitivity preferences
- `/about` architecture and safety information

## Wake word

Backend wake-word listening reuses `MicrophoneListener` and `remove_wake_phrase`
from `driverguardian_voice_commands_v3.py`. It requires a working microphone,
`SpeechRecognition`, PyAudio, and internet access for the current Google speech
recognition implementation. Start it from the Commander page.

## Integration note

The included live dashboard uses the existing V4 simulation state so the web
application is immediately runnable. The next hardware integration point is
`app/services/app_state.py`: replace `metrics()` with values published by the
real V3 monitoring pipeline, while leaving the API and UI unchanged.
