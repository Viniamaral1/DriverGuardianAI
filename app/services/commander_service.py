from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.services.app_state import GuardianState


class CommanderService:
    def __init__(self, state: GuardianState) -> None:
        self.state = state

    def answer(self, message: str) -> str:
        text = re.sub(r"\s+", " ", message.strip())
        q = text.lower()
        metrics = self.state.metrics()

        if not q:
            return "I did not receive a command."

        if any(term in q for term in ("start monitoring", "begin monitoring", "start session")):
            self.state.start_monitoring()
            return "Monitoring started. Vision, temporal logic, and controlled alerts are now active."

        if any(term in q for term in ("stop monitoring", "end monitoring", "stop session")):
            self.state.stop_monitoring()
            return "Monitoring stopped. The current session is no longer collecting live metrics."

        if "status" in q or "how am i" in q:
            return (
                f"System state is {metrics['state']}. Fatigue probability is "
                f"{metrics['fatigue_probability'] * 100:.1f} percent, EAR is {metrics['ear']:.3f}, "
                f"and {metrics['alert_count']} controlled alerts have occurred."
            )

        if "fatigue" in q or "risk" in q:
            return f"Current fatigue probability is {metrics['fatigue_probability'] * 100:.1f} percent and the driver state is {metrics['state']}."

        if "ear" in q or "eye" in q:
            return f"Current eye-aspect ratio is {metrics['ear']:.3f}. Lower sustained values can support fatigue evidence after personal calibration."

        if "blink" in q:
            return f"Current estimated blink rate is {metrics['blink_rate']:.1f} blinks per minute."

        if "yawn" in q:
            return f"Current yawn score is {metrics['yawn_score']:.3f}."

        if "tilt" in q or "head" in q:
            return f"Current head tilt is {metrics['head_tilt']:.1f} degrees."

        if "duration" in q or "how long" in q:
            seconds = int(metrics["session_seconds"])
            return f"The active monitoring session has run for {seconds // 60} minutes and {seconds % 60} seconds."

        if any(term in q for term in ("help", "what can you do", "commands")):
            return (
                "I can start or stop monitoring, report live fatigue metrics, explain the latest "
                "session report, describe calibration, and summarise alert evidence."
            )

        try:
            from driverguardian_copilot_v3 import DriverGuardianCopilot
            return DriverGuardianCopilot().answer(text)
        except Exception:
            return (
                "I could not map that request confidently. Ask for system status, fatigue risk, "
                "EAR, blink rate, yawning, head tilt, session duration, or the latest report."
            )
