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
        try:
            from app.services.intelligence_service import IntelligenceService
            intelligence = IntelligenceService(self.state).snapshot()
        except Exception:
            intelligence = {}

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
            "why did you alert",
            "why am i",
            "explain the decision",
            "explain my risk",
            "what caused",
            "why warning",
            "why critical",
        )):
            explanation = intelligence.get("explanation", {}) or {}
            contributions = explanation.get("contributions", []) or []
            active = [
                item for item in contributions
                if float(item.get("share", 0) or 0) > 0
            ]
            if not active:
                return (
                    "There is no calibrated risk contribution to explain yet. "
                    "Start Monitoring and complete calibration first."
                )
            details = ", ".join(
                f"{item.get('label', 'signal')} "
                f"{float(item.get('share', 0)) * 100:.0f} percent"
                for item in active[:3]
            )
            return (
                f"{explanation.get('summary', 'Guardian analysed the available signals')} "
                f"Relative available evidence is {details}. These percentages explain "
                "the current decision layer; they are not medical causal claims."
            )

        if any(term in q for term in (
            "forecast",
            "what happens next",
            "should i stop soon",
            "break soon",
            "predict fatigue",
            "fatigue outlook",
        )):
            outlook = intelligence.get("outlook", {}) or {}
            return (
                f"Near-term outlook is {outlook.get('level', 'standby')}. "
                f"{outlook.get('horizon', 'No live forecast')}. "
                f"{outlook.get('summary', 'Start Monitoring to activate the outlook')}"
            )

        if any(term in q for term in (
            "signal quality",
            "can you see me",
            "camera quality",
            "confidence",
        )):
            quality = intelligence.get("signal_quality", {}) or {}
            return (
                f"Current signal quality is {quality.get('level', 'standby')} "
                f"at {float(quality.get('score', 0)) * 100:.0f} percent. "
                f"{quality.get('summary', 'No live signal is available.')}"
            )

        if any(term in q for term in (
            "compare baseline",
            "personal baseline",
            "compare me to history",
            "current baseline",
        )):
            baseline = intelligence.get("baseline_comparison", {}) or {}
            return str(
                baseline.get(
                    "summary",
                    "Complete calibration to compare the current baseline.",
                )
            )

        if any(term in q for term in (
            "automatic context",
            "time of day",
            "is it night",
            "automatic lighting",
        )):
            auto_context = intelligence.get("context", {}) or {}
            return (
                f"Guardian automatically classifies the current local period as "
                f"{auto_context.get('automatic_time_period', 'unknown')} and "
                f"external light context as "
                f"{auto_context.get('automatic_external_light', 'unknown')}. "
                "Weather, cabin lighting and occlusion remain manual context in this version."
            )

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
                "I can start or stop monitoring, report live fatigue metrics, explain the current decision, provide a near-term outlook, compare the current baseline with local history, summarise session reports, report local driving patterns, and describe journey context."
            )

        try:
            from driverguardian_copilot_v3 import DriverGuardianCopilot
            return DriverGuardianCopilot().answer(text)
        except Exception:
            return (
                "I could not map that request confidently. Ask for system status, fatigue risk, "
                "EAR, blink rate, yawning, head tilt, session duration, or the latest report."
            )
