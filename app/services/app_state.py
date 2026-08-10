from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.automatic_report_service import AutomaticReportService
from app.services.live_monitoring_service import LiveMonitoringService
from app.services.edge_memory_service import EdgeMemoryService
from app.services.driver_profile_service import DriverProfileService


class GuardianState:
    def __init__(self) -> None:
        self.root = Path.cwd()
        self.lock = threading.RLock()
        self.sequence = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=100)
        self.conversation: deque[dict[str, str]] = deque(maxlen=80)
        self.settings: dict[str, Any] = {
            "theme": "dark",
            "accent": "cyan",
            "voice_output": True,
            "wake_word_enabled": False,
            "camera_index": 0,
            "alert_volume": 80,
            "sensitivity": 65,
            "driver_name": "",
            "automatic_reports": True,
            "persistent_calibration_enabled": True,
        }
        self.voice_service = None
        self.monitoring_service: LiveMonitoringService | None = None
        self.report_service: AutomaticReportService | None = None
        self.edge_memory_service: EdgeMemoryService | None = None
        self.driver_profile_service: DriverProfileService | None = None
        self.intelligence_service = None
        self._decision_memory_thread: threading.Thread | None = None
        self._decision_memory_stop = threading.Event()

    def initialise(self, root: Path) -> None:
        self.root = root
        self._load_settings()
        self.driver_profile_service = DriverProfileService(root)
        self.driver_profile_service.ensure_default_profile(
            str(self.settings.get("driver_name", "")).strip()
        )
        self.monitoring_service = LiveMonitoringService(
            root=root,
            settings_provider=lambda: dict(self.settings),
            event_callback=self.add_event,
            profile_provider=(
                lambda: self.driver_profile_service.active_profile()
                if self.driver_profile_service is not None
                else None
            ),
            profile_service=self.driver_profile_service,
        )
        self.report_service = AutomaticReportService(
            root=root,
            event_callback=self.add_event,
        )
        self.edge_memory_service = EdgeMemoryService(root)

        # Local import avoids an import cycle while allowing GuardianState to
        # own the Intelligence/Decision Memory lifecycle.
        from app.services.intelligence_service import IntelligenceService
        self.intelligence_service = IntelligenceService(self)

        self.add_event("SYSTEM", "Guardian OS V6.1 persistent profiles ready", "info")
        name = str(self.settings.get("driver_name", "")).strip()
        greeting = f"Welcome back, {name}." if name else "Commander online."
        self.add_message(
            "assistant",
            f"{greeting} Start monitoring when you are ready.",
        )

    def shutdown(self) -> None:
        self._stop_decision_memory_worker()
        if self.intelligence_service is not None:
            self.intelligence_service.decision_memory.finalise()
        if self.voice_service is not None:
            self.voice_service.stop()
        if self.monitoring_service is not None:
            self.monitoring_service.stop()

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

    def _decision_memory_loop(self) -> None:
        """Capture advisory evidence independently of any browser page."""
        while True:
            service = self.intelligence_service
            interval = (
                service.decision_memory.SAMPLE_INTERVAL_SECONDS
                if service is not None
                else 2.0
            )
            if self._decision_memory_stop.wait(interval):
                break
            if service is None:
                continue

            metrics = self.metrics()
            if not bool(metrics.get("monitoring")):
                if not (
                    metrics.get("starting")
                    or metrics.get("stopping")
                ):
                    break
                continue

            try:
                service.snapshot(record_memory=True)
            except Exception as error:
                # Decision Memory is research/explainability telemetry only.
                # It must never interrupt the safety-critical live path.
                self.add_event(
                    "MEMORY",
                    f"Decision Memory sample skipped: {type(error).__name__}",
                    "warning",
                )

    def _start_decision_memory_worker(self) -> None:
        self._stop_decision_memory_worker()
        self._decision_memory_stop.clear()
        self._decision_memory_thread = threading.Thread(
            target=self._decision_memory_loop,
            name="guardian-decision-memory",
            daemon=True,
        )
        self._decision_memory_thread.start()

    def _stop_decision_memory_worker(self) -> None:
        self._decision_memory_stop.set()
        thread = self._decision_memory_thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=3.0)
        self._decision_memory_thread = None

    def start_monitoring(self) -> tuple[bool, str]:
        if self.monitoring_service is None:
            return False, "Monitoring service has not been initialised."
        success, message = self.monitoring_service.start()
        if success:
            if self.intelligence_service is not None:
                self.intelligence_service.decision_memory.begin(self.metrics())
                self._start_decision_memory_worker()

            profile = (
                self.driver_profile_service.active_profile()
                if self.driver_profile_service is not None
                else None
            )
            profile_name = str((profile or {}).get("name") or "").strip()
            welcome = f" for {profile_name}" if profile_name else " in Guest mode"
            self.add_event("MONITOR", f"{message}{welcome}", "success")
        return success, message

    def stop_monitoring(self) -> tuple[bool, str]:
        if self.monitoring_service is None:
            return True, "Monitoring is already stopped."

        session_log = self.monitoring_service.snapshot().get("log_path")
        success, message = self.monitoring_service.stop()

        self._stop_decision_memory_worker()
        if self.intelligence_service is not None:
            memory_status = self.intelligence_service.decision_memory.finalise()
            completed_id = memory_status.get("completed_session_id")
            if completed_id:
                self.add_event(
                    "MEMORY",
                    f"Decision Memory saved: {completed_id}",
                    "success",
                )

        self.add_event("MONITOR", message, "info")

        if (
            success
            and self.report_service is not None
            and bool(self.settings.get("automatic_reports", True))
        ):
            self.report_service.generate_async(session_log)

        return success, message

    def metrics(self) -> dict[str, Any]:
        if self.monitoring_service is None:
            metrics = LiveMonitoringService.empty_metrics()
        else:
            metrics = self.monitoring_service.snapshot()

        if not (
            metrics.get("monitoring")
            or metrics.get("starting")
            or metrics.get("stopping")
        ):
            profile = (
                self.driver_profile_service.active_profile()
                if self.driver_profile_service is not None
                else None
            )
            calibration = (profile or {}).get("calibration") or {}
            metrics.update(
                {
                    "driver_profile_id": (profile or {}).get("id"),
                    "driver_profile_name": str(
                        (profile or {}).get("name") or "Guest"
                    ),
                    "calibration_mode": (
                        "quick" if profile and calibration else "full"
                    ),
                    "calibration_status": (
                        "VERIFYING SAVED PROFILE"
                        if profile and calibration
                        else "FULL CALIBRATION"
                    ),
                    "calibration_required_seconds": (
                        3.0 if profile and calibration else 10.0
                    ),
                }
            )
        return metrics

    def snapshot(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics(),
            "events": list(self.events)[:20],
            "conversation": list(self.conversation),
            "settings": dict(self.settings),
            "voice": self.voice_status(),
            "report": (
                self.report_service.snapshot()
                if self.report_service is not None
                else {"state": "IDLE"}
            ),
            "edge": (
                self.edge_memory_service.snapshot()
                if self.edge_memory_service is not None
                else {"available": False}
            ),
            "profiles": (
                self.driver_profile_service.snapshot()
                if self.driver_profile_service is not None
                else {"available": False}
            ),
        }

    def voice_status(self) -> dict[str, Any]:
        if self.voice_service is None:
            return {
                "available": False,
                "enabled": False,
                "status": "OFFLINE",
                "detail": "Use browser hands-free mode, or enable the optional Python microphone service.",
            }
        return self.voice_service.status()


guardian_state = GuardianState()
