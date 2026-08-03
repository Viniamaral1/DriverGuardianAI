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
        edge = (
            self.state.edge_memory_service.snapshot()
            if self.state.edge_memory_service is not None
            else {"available": False}
        )
        insights = edge.get("insights", {}) or {}
        context = edge.get("context", {}) or {}

        if not q:
            return "I did not receive a command."

        if any(term in q for term in ("start monitoring", "begin monitoring", "start session")):
            self.state.start_monitoring()
            return "Monitoring started. Vision, temporal logic, and controlled alerts are now active."

        if q in {"stop", "end", "cancel"} or any(term in q for term in ("stop monitoring", "end monitoring", "stop session")):
            self.state.stop_monitoring()
            return "Monitoring stopped. The current session is no longer collecting live metrics."

        if "status" in q or "how am i" in q:
            name = str(self.state.settings.get("driver_name", "")).strip()
            prefix = f"{name}, " if name else ""
            return (
                prefix +
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

        if any(term in q for term in (
            "what have you learned",
            "what do you remember",
            "driver pattern",
            "my patterns",
            "edge memory",
            "offline memory",
        )):
            session_count = int(insights.get("session_count", 0) or 0)
            if not session_count:
                return (
                    "Guardian Edge Memory is ready, but there are no completed "
                    "session reports to learn from yet."
                )
            return (
                f"Guardian Edge Memory contains {session_count} completed sessions. "
                f"Average recorded risk is {float(insights.get('average_risk', 0)) * 100:.1f} percent, "
                f"highest risk is {float(insights.get('highest_risk', 0)) * 100:.1f} percent, "
                f"and {int(insights.get('total_alerts', 0) or 0)} controlled alerts are stored. "
                f"The highest-risk time period in the available records is "
                f"{insights.get('highest_risk_period', 'not enough data')}."
            )

        if any(term in q for term in (
            "last session",
            "latest session",
            "previous session",
        )):
            latest = insights.get("latest_session")
            if not latest:
                return "No completed session is available in local memory yet."
            return (
                f"The latest stored session lasted "
                f"{float(latest.get('duration_seconds', 0)) / 60:.1f} minutes, "
                f"reached {float(latest.get('maximum_risk', 0)) * 100:.1f} percent maximum risk, "
                f"and recorded {int(latest.get('alert_count', 0) or 0)} alerts. "
                f"The dominant signal was {latest.get('dominant_signal', 'unknown')}."
            )

        if any(term in q for term in (
            "offline",
            "internet",
            "connection",
            "sync queue",
            "pending sync",
        )):
            return (
                "Core monitoring and Guardian Edge Memory work locally without an internet connection. "
                f"There are currently {int(edge.get('pending_sync', 0) or 0)} session records "
                "waiting in the optional sync queue."
            )

        if any(term in q for term in (
            "weather",
            "road condition",
            "journey context",
            "lighting",
            "glasses",
            "hat",
            "occlusion",
        )):
            return (
                f"Current journey context is weather {context.get('weather', 'unknown')}, "
                f"road condition {context.get('road_condition', 'unknown')}, "
                f"external light {context.get('external_light', 'unknown')}, "
                f"cabin light {context.get('cabin_light', 'unknown')}, and "
                f"occlusion {context.get('occlusion', 'none')}. "
                "These values provide context and do not directly determine fatigue."
            )

        if any(term in q for term in ("help", "what can you do", "commands")):
            return (
                "I can start or stop monitoring, report live fatigue metrics, explain the latest "
                "session report, describe calibration, summarise alert evidence, report local driving patterns, and describe journey context."
            )

        try:
            from driverguardian_copilot_v3 import DriverGuardianCopilot
            return DriverGuardianCopilot().answer(text)
        except Exception:
            return (
                "I could not map that request confidently. Ask for system status, fatigue risk, "
                "EAR, blink rate, yawning, head tilt, session duration, or the latest report."
            )
