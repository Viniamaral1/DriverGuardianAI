from __future__ import annotations

import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(
    title="DriverGuardianAI V4",
    description="Automotive-style driver monitoring dashboard with Commander.",
    version="4.1.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class CommanderRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class MonitoringState:
    """Thread-safe simulated monitoring state.

    Replace the simulation loop with the real V3 pipeline in the next stage.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._started_at: datetime | None = None
        self._metrics: dict[str, Any] = {
            "system_status": "standby",
            "monitoring": False,
            "camera_status": "offline",
            "ai_status": "standby",
            "commander_status": "ready",
            "driver_status": "unknown",
            "alert_level": "standby",
            "fatigue_probability": 0.0,
            "ear": 0.0,
            "yawn_score": 0.0,
            "blink_rate": 0,
            "head_tilt": 0.0,
            "face_confidence": 0.0,
            "session_seconds": 0,
            "last_updated": datetime.now().isoformat(),
        }
        self._logs: list[dict[str, str]] = []
        self._conversation: list[dict[str, str]] = []
        self._add_log_unlocked("info", "DriverGuardianAI dashboard ready.")
        self._add_message_unlocked(
            "assistant",
            "Commander online. Ask me to start monitoring, stop monitoring, explain the current driver condition, or summarise the latest report.",
        )

        threading.Thread(target=self._simulation_loop, daemon=True).start()

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._running:
                return {"success": True, "message": "Monitoring is already running."}

            self._running = True
            self._started_at = datetime.now()
            self._metrics.update(
                {
                    "system_status": "online",
                    "monitoring": True,
                    "camera_status": "online",
                    "ai_status": "active",
                    "commander_status": "ready",
                    "driver_status": "alert",
                    "alert_level": "normal",
                    "fatigue_probability": 0.04,
                    "ear": 0.31,
                    "yawn_score": 0.02,
                    "blink_rate": 16,
                    "head_tilt": 1.5,
                    "face_confidence": 0.97,
                    "session_seconds": 0,
                    "last_updated": datetime.now().isoformat(),
                }
            )
            self._add_log_unlocked("success", "Monitoring session started.")
            return {"success": True, "message": "Monitoring started."}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._running:
                return {"success": True, "message": "Monitoring is already stopped."}

            self._running = False
            self._started_at = None
            self._metrics.update(
                {
                    "system_status": "standby",
                    "monitoring": False,
                    "camera_status": "offline",
                    "ai_status": "standby",
                    "commander_status": "ready",
                    "driver_status": "unknown",
                    "alert_level": "standby",
                    "fatigue_probability": 0.0,
                    "ear": 0.0,
                    "yawn_score": 0.0,
                    "blink_rate": 0,
                    "head_tilt": 0.0,
                    "face_confidence": 0.0,
                    "session_seconds": 0,
                    "last_updated": datetime.now().isoformat(),
                }
            )
            self._add_log_unlocked("warning", "Monitoring session stopped.")
            return {"success": True, "message": "Monitoring stopped."}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._metrics,
                "logs": list(self._logs[-30:]),
                "conversation": list(self._conversation[-40:]),
            }

    def current_metrics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._metrics)

    def set_commander_status(self, status: str) -> None:
        with self._lock:
            self._metrics["commander_status"] = status

    def add_log(self, level: str, message: str) -> None:
        with self._lock:
            self._add_log_unlocked(level, message)

    def add_message(self, role: str, message: str) -> None:
        with self._lock:
            self._add_message_unlocked(role, message)

    def clear_conversation(self) -> None:
        with self._lock:
            self._conversation.clear()
            self._add_message_unlocked("assistant", "Conversation cleared. Commander is ready.")

    def _add_log_unlocked(self, level: str, message: str) -> None:
        self._logs.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            }
        )
        self._logs = self._logs[-100:]

    def _add_message_unlocked(self, role: str, message: str) -> None:
        self._conversation.append(
            {
                "role": role,
                "message": message,
                "time": datetime.now().strftime("%H:%M"),
            }
        )
        self._conversation = self._conversation[-100:]

    def _simulation_loop(self) -> None:
        while True:
            time.sleep(1)
            with self._lock:
                if not self._running:
                    continue

                fatigue = max(
                    0.01,
                    min(
                        0.99,
                        float(self._metrics["fatigue_probability"])
                        + random.uniform(-0.025, 0.035),
                    ),
                )
                ear = max(0.12, min(0.38, 0.33 - fatigue * 0.12 + random.uniform(-0.012, 0.012)))
                yawn_score = max(0.0, min(1.0, fatigue * 0.55 + random.uniform(-0.06, 0.06)))
                head_tilt = max(-20.0, min(20.0, random.uniform(-3.5, 3.5) + fatigue * 5))
                blink_rate = max(6, min(40, int(14 + fatigue * 14 + random.randint(-2, 2))))

                if fatigue >= 0.75:
                    alert_level = "critical"
                    driver_status = "fatigue detected"
                elif fatigue >= 0.45:
                    alert_level = "warning"
                    driver_status = "drowsiness warning"
                else:
                    alert_level = "normal"
                    driver_status = "alert"

                session_seconds = 0
                if self._started_at is not None:
                    session_seconds = int((datetime.now() - self._started_at).total_seconds())

                previous_level = self._metrics["alert_level"]
                self._metrics.update(
                    {
                        "fatigue_probability": round(fatigue, 3),
                        "ear": round(ear, 3),
                        "yawn_score": round(yawn_score, 3),
                        "blink_rate": blink_rate,
                        "head_tilt": round(head_tilt, 1),
                        "face_confidence": round(random.uniform(0.93, 0.99), 3),
                        "alert_level": alert_level,
                        "driver_status": driver_status,
                        "session_seconds": session_seconds,
                        "last_updated": datetime.now().isoformat(),
                    }
                )

                if previous_level != alert_level:
                    if alert_level == "critical":
                        self._add_log_unlocked("critical", "Critical fatigue condition detected.")
                    elif alert_level == "warning":
                        self._add_log_unlocked("warning", "Possible driver drowsiness detected.")
                    elif alert_level == "normal":
                        self._add_log_unlocked("success", "Driver condition returned to normal.")


class CommanderService:
    """Deterministic dashboard assistant.

    It handles live dashboard commands directly and delegates historical report
    questions to the existing V3 copilot when a JSON report is available.
    """

    def __init__(self, state: MonitoringState) -> None:
        self.state = state

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9%\s']", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _duration(seconds: int) -> str:
        hours, remainder = divmod(max(0, seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours} hours, {minutes} minutes, and {seconds} seconds"
        if minutes:
            return f"{minutes} minutes and {seconds} seconds"
        return f"{seconds} seconds"

    def _live_summary(self) -> str:
        m = self.state.current_metrics()
        if not m["monitoring"]:
            return "Monitoring is currently stopped. Say or type ‘start monitoring’ when you are ready."

        fatigue = round(float(m["fatigue_probability"]) * 100)
        return (
            f"The driver is currently classified as {m['driver_status']}. "
            f"Fatigue probability is {fatigue} percent, eye aspect ratio is {float(m['ear']):.3f}, "
            f"yawn score is {float(m['yawn_score']):.3f}, blink rate is {int(m['blink_rate'])} per minute, "
            f"and head tilt is {float(m['head_tilt']):.1f} degrees. "
            f"The current alert level is {m['alert_level']}."
        )

    def _latest_report_answer(self, question: str) -> str | None:
        try:
            from driverguardian_copilot_v3 import DriverGuardianCopilot

            copilot = DriverGuardianCopilot()
            return copilot.answer(question)
        except FileNotFoundError:
            return "I could not find a completed V3 JSON session report yet. Generate a report first, then ask me again."
        except Exception as exc:
            self.state.add_log("warning", f"Commander report integration unavailable: {type(exc).__name__}")
            return None

    def answer(self, original: str) -> tuple[str, str | None]:
        q = self._normalise(original)

        if any(phrase in q for phrase in ("start monitoring", "begin monitoring", "start session", "begin session")):
            result = self.state.start()
            return result["message"] + " I am now watching the live driver metrics.", "start"

        if any(phrase in q for phrase in ("stop monitoring", "end monitoring", "stop session", "end session")):
            result = self.state.stop()
            return result["message"] + " The dashboard has returned to standby.", "stop"

        if any(phrase in q for phrase in ("current condition", "driver condition", "how tired", "fatigue level", "status", "live summary")):
            return self._live_summary(), None

        if "ear" in q or "eye aspect" in q:
            m = self.state.current_metrics()
            if not m["monitoring"]:
                return "EAR is unavailable because monitoring is stopped.", None
            return f"The current eye aspect ratio is {float(m['ear']):.3f}.", None

        if "yawn" in q:
            m = self.state.current_metrics()
            if not m["monitoring"]:
                return "Yawn score is unavailable because monitoring is stopped.", None
            return f"The current yawn score is {float(m['yawn_score']):.3f}.", None

        if "blink" in q:
            m = self.state.current_metrics()
            if not m["monitoring"]:
                return "Blink rate is unavailable because monitoring is stopped.", None
            return f"The current blink rate is {int(m['blink_rate'])} blinks per minute.", None

        if "head tilt" in q or "head position" in q:
            m = self.state.current_metrics()
            if not m["monitoring"]:
                return "Head-tilt information is unavailable because monitoring is stopped.", None
            return f"The current head tilt is {float(m['head_tilt']):.1f} degrees.", None

        if "how long" in q and any(word in q for word in ("driving", "session", "monitoring")):
            m = self.state.current_metrics()
            if not m["monitoring"]:
                return "There is no active monitoring session.", None
            return f"The current monitoring session has been active for {self._duration(int(m['session_seconds']))}.", None

        if any(phrase in q for phrase in ("what can you do", "help", "commands")):
            return (
                "You can ask me to start or stop monitoring, describe the current driver condition, "
                "report fatigue probability, EAR, yawn score, blink rate, head tilt, session duration, "
                "or explain the latest completed V3 report.",
                None,
            )

        report_keywords = (
            "summarise the session",
            "summarize the session",
            "why did you alert",
            "explain calibration",
            "raw and calibrated",
            "each state",
            "what should the driver do",
            "latest report",
            "completed session",
        )
        if any(keyword in q for keyword in report_keywords):
            answer = self._latest_report_answer(original)
            if answer is not None:
                return answer, None

        return (
            "I did not understand that confidently. Try ‘start monitoring’, ‘how tired is the driver?’, "
            "‘what is the EAR?’, ‘how long has this session been running?’, or ‘summarise the latest report’.“".replace("“", ""),
            None,
        )


monitoring_state = MonitoringState()
commander = CommanderService(monitoring_state)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"page_title": "DriverGuardianAI", "version": "V4.1"},
    )


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    return monitoring_state.snapshot()


@app.post("/api/monitoring/start")
async def start_monitoring() -> dict[str, Any]:
    return monitoring_state.start()


@app.post("/api/monitoring/stop")
async def stop_monitoring() -> dict[str, Any]:
    return monitoring_state.stop()


@app.post("/api/commander/message")
async def commander_message(payload: CommanderRequest) -> dict[str, Any]:
    text = payload.message.strip()
    monitoring_state.add_message("user", text)
    monitoring_state.set_commander_status("thinking")

    response, action = commander.answer(text)

    monitoring_state.add_message("assistant", response)
    monitoring_state.add_log("info", f"Commander handled: {text[:80]}")
    monitoring_state.set_commander_status("ready")

    return {
        "success": True,
        "response": response,
        "action": action,
        "state": monitoring_state.snapshot(),
    }


@app.post("/api/commander/clear")
async def clear_commander() -> dict[str, Any]:
    monitoring_state.clear_conversation()
    return {"success": True, "message": "Commander conversation cleared."}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "DriverGuardianAI V4.1"}
