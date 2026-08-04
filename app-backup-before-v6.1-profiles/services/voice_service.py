from __future__ import annotations

import threading
from typing import Callable


class WakeWordService:
    """Optional backend microphone listener using the existing V3 recognition stack."""

    def __init__(self, on_command: Callable[[str], str]) -> None:
        self.on_command = on_command
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._status = "OFFLINE"
        self._detail = "Not started"
        self._available = False
        try:
            from driverguardian_voice_commands_v3 import MicrophoneListener, remove_wake_phrase
            self.listener_class = MicrophoneListener
            self.remove_wake_phrase = remove_wake_phrase
            probe = MicrophoneListener(timeout=2.0, phrase_time_limit=7.0, ambient_seconds=0.6)
            self._available = probe.available
            self._detail = "Microphone available" if probe.available else (probe.error or "Microphone unavailable")
        except Exception as exc:
            self.listener_class = None
            self.remove_wake_phrase = None
            self._detail = f"{type(exc).__name__}: {exc}"

    def start(self) -> bool:
        if not self._available or self._thread and self._thread.is_alive():
            return self._available
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="commander-wake-word", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        self._status = "OFFLINE"

    def _run(self) -> None:
        listener = self.listener_class(timeout=2.0, phrase_time_limit=8.0, ambient_seconds=0.8)
        self._status = "CALIBRATING"
        if not listener.calibrate():
            self._status = "ERROR"
            self._detail = listener.error or "Calibration failed"
            return
        self._status = "LISTENING"
        self._detail = 'Say "Commander" followed by a command'
        while not self._stop.is_set():
            text = listener.listen()
            if not text:
                continue
            command, detected = self.remove_wake_phrase(text)
            if not detected:
                continue
            self._status = "PROCESSING"
            try:
                self.on_command(command or "help")
            finally:
                self._status = "LISTENING"

    def status(self) -> dict:
        return {
            "available": self._available,
            "enabled": bool(self._thread and self._thread.is_alive() and not self._stop.is_set()),
            "status": self._status,
            "detail": self._detail,
        }
