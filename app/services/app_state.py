from __future__ import annotations

import asyncio
import json
import math
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class GuardianState:
    def __init__(self) -> None:
        self.root = Path.cwd()
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.monitoring_started_at: float | None = None
        self.monitoring = False
        self.alert_count = 0
        self.sequence = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=80)
        self.conversation: deque[dict[str, str]] = deque(maxlen=80)
        self.settings: dict[str, Any] = {
            "theme": "dark",
            "accent": "cyan",
            "voice_output": True,
            "wake_word_enabled": False,
            "camera_index": 0,
            "alert_volume": 80,
            "sensitivity": 65,
        }
        self.voice_service = None
        self._last_state = "READY"

    def initialise(self, root: Path) -> None:
        self.root = root
        self.started_at = time.monotonic()
        self._load_settings()
        self.add_event("SYSTEM", "DriverGuardianAI V4 services ready", "info")
        self.add_message(
            "assistant",
            "Commander online. Start monitoring or ask me about the latest session report.",
        )

    def shutdown(self) -> None:
        if self.voice_service is not None:
            self.voice_service.stop()

    @property
    def settings_path(self) -> Path:
        return self.root / "config" / "web_settings.json"

    def _load_settings(self) -> None:
        try:
            loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.settings.update(loaded)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    def save_settings(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(
            json.dumps(self.settings, indent=2),
            encoding="utf-8",
        )

    def add_event(self, source: str, message: str, level: str = "info") -> None:
        with self.lock:
            self.sequence += 1
            self.events.appendleft(
                {
                    "id": self.sequence,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "source": source,
                    "message": message,
                    "level": level,
                }
            )

    def add_message(self, role: str, content: str) -> None:
        with self.lock:
            self.conversation.append(
                {
                    "role": role,
                    "content": content,
                    "time": datetime.now().strftime("%H:%M"),
                }
            )

    def start_monitoring(self) -> None:
        with self.lock:
            if self.monitoring:
                return
            self.monitoring = True
            self.monitoring_started_at = time.monotonic()
            self.alert_count = 0
            self._last_state = "MONITORING"
        self.add_event("MONITOR", "Live monitoring started", "success")

    def stop_monitoring(self) -> None:
        with self.lock:
            if not self.monitoring:
                return
            self.monitoring = False
            self._last_state = "READY"
        self.add_event("MONITOR", "Live monitoring stopped", "info")

    def metrics(self) -> dict[str, Any]:
        with self.lock:
            monitoring = self.monitoring
            started = self.monitoring_started_at
            alerts = self.alert_count

        elapsed = (time.monotonic() - started) if monitoring and started else 0.0
        if monitoring:
            phase = elapsed / 5.2
            fatigue = max(0.04, min(0.97, 0.24 + 0.18 * math.sin(phase) + random.uniform(-0.025, 0.025)))
            ear = max(0.14, min(0.36, 0.305 - fatigue * 0.105 + random.uniform(-0.006, 0.006)))
            blink_rate = max(4.0, 12.0 + fatigue * 17.0 + random.uniform(-1.3, 1.3))
            yawn = max(0.0, min(1.0, fatigue * 0.62 + random.uniform(-0.07, 0.07)))
            tilt = max(0.0, fatigue * 16.0 + random.uniform(-1.4, 1.4))
            face = True
            if fatigue >= 0.78:
                state = "CRITICAL"
            elif fatigue >= 0.57:
                state = "WARNING"
            else:
                state = "MONITORING"
        else:
            fatigue, ear, blink_rate, yawn, tilt, face, state = 0.0, 0.0, 0.0, 0.0, 0.0, False, "READY"

        if state != self._last_state and monitoring:
            level = "danger" if state == "CRITICAL" else "warning" if state == "WARNING" else "success"
            self.add_event("AI", f"Driver state changed to {state}", level)
            if state == "CRITICAL":
                with self.lock:
                    self.alert_count += 1
                    alerts = self.alert_count
            self._last_state = state

        return {
            "monitoring": monitoring,
            "state": state,
            "fatigue_probability": round(fatigue, 3),
            "ear": round(ear, 3),
            "blink_rate": round(blink_rate, 1),
            "yawn_score": round(yawn, 3),
            "head_tilt": round(tilt, 1),
            "face_detected": face,
            "session_seconds": round(elapsed),
            "alert_count": alerts,
            "camera_status": "CONNECTED" if monitoring else "STANDBY",
            "model_status": "ACTIVE" if monitoring else "READY",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics(),
            "events": list(self.events)[:12],
            "conversation": list(self.conversation),
            "settings": dict(self.settings),
            "voice": self.voice_status(),
        }

    def voice_status(self) -> dict[str, Any]:
        if self.voice_service is None:
            return {
                "available": False,
                "enabled": False,
                "status": "OFFLINE",
                "detail": "Wake-word service not initialised",
            }
        return self.voice_service.status()


guardian_state = GuardianState()
