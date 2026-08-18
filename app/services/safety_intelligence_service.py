from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import math


class SafetyIntelligenceService:
    """Unifies Guardian's validated intelligence layers into one trust chain.

    V9.0 does not recalculate fatigue or alter any live safety decision. It
    normalises existing outputs so UI, future APIs and a later CAG/RAG copilot
    can consume one stable, explainable state object.
    """

    VERSION = "9.0-safety-intelligence-v1"

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value if value is not None else default)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _reason(
        code: str,
        layer: str,
        severity: str,
        message: str,
        *,
        blocking: bool = False,
        value: Any = None,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "layer": layer,
            "severity": severity,
            "message": message,
            "blocking": bool(blocking),
            "value": value,
        }

    @classmethod
    def snapshot(
        cls,
        *,
        live: dict[str, Any],
        decision_engine: dict[str, Any],
        perception: dict[str, Any],
        passport_validation: dict[str, Any] | None,
        predictive: dict[str, Any],
        decision_memory: dict[str, Any] | None,
        journey_caution: dict[str, Any],
        signal_quality: dict[str, Any],
    ) -> dict[str, Any]:
        monitoring = bool(live.get("monitoring"))
        calibrated = bool(live.get("calibrated"))
        perception_state = str(perception.get("state") or "standby").lower()
        passport_state = str(
            (passport_validation or {}).get("state") or "unavailable"
        ).lower()
        predictive_status = str(predictive.get("status") or "standby").lower()
        risk = cls._number(decision_engine.get("risk_score"))
        decision_confidence = cls._number(decision_engine.get("confidence"))
        perception_score = cls._number(perception.get("score"))
        passport_drift = cls._number(
            (passport_validation or {}).get("drift_score")
        )

        reasons: list[dict[str, Any]] = []
        if not monitoring:
            reasons.append(cls._reason(
                "MONITORING_INACTIVE", "monitoring", "info",
                "Live Monitoring is not active.", blocking=True,
            ))
        if monitoring and not calibrated:
            reasons.append(cls._reason(
                "CALIBRATION_INCOMPLETE", "personalisation", "warning",
                "Personal calibration is not complete.", blocking=True,
            ))

        for item in perception.get("reason_codes", []) or []:
            if isinstance(item, dict):
                severity = str(item.get("severity") or "info")
                reasons.append(cls._reason(
                    str(item.get("code") or "PERCEPTION_REASON"),
                    "perception",
                    severity,
                    str(item.get("message") or "Perception limitation recorded."),
                    blocking=severity == "critical" or perception_state == "insufficient",
                    value=item.get("value"),
                ))

        if passport_state == "watch":
            reasons.append(cls._reason(
                "PASSPORT_WATCH", "personalisation", "warning",
                "The Personal AI Calibration Passport requires additional observation.",
                value=passport_drift,
            ))
        elif passport_state == "drift_detected":
            reasons.append(cls._reason(
                "PASSPORT_DRIFT_DETECTED", "personalisation", "warning",
                "Passport drift has been detected; personalised forecasting is restricted.",
                blocking=True, value=passport_drift,
            ))
        elif passport_state == "recalibration_recommended":
            reasons.append(cls._reason(
                "PASSPORT_RECALIBRATION_RECOMMENDED", "personalisation", "critical",
                "A fresh calibration is recommended before personalised forecasting.",
                blocking=True, value=passport_drift,
            ))

        for text in predictive.get("withheld_reasons", []) or []:
            reasons.append(cls._reason(
                "PREDICTION_WITHHELD", "predictive", "warning",
                str(text), blocking=True,
            ))

        if bool(journey_caution.get("elevated")):
            for text in journey_caution.get("reasons", []) or []:
                reasons.append(cls._reason(
                    "JOURNEY_CONTEXT_CAUTION", "context", "info",
                    f"Journey context caution: {text}.",
                ))

        # Dedupe by code+message while preserving first occurrence.
        seen: set[tuple[str, str]] = set()
        normalised: list[dict[str, Any]] = []
        for item in reasons:
            key=(str(item.get("code")),str(item.get("message")))
            if key in seen:
                continue
            seen.add(key); normalised.append(item)
        reasons=normalised

        if not monitoring:
            trust_state = "standby"
        elif not calibrated:
            trust_state = "limited"
        elif perception_state == "insufficient":
            trust_state = "limited"
        elif passport_state in {"drift_detected", "recalibration_recommended"}:
            trust_state = "limited"
        elif perception_state == "degraded" or passport_state == "watch":
            trust_state = "guarded"
        else:
            trust_state = "trusted"

        trust_score = 1.0
        if monitoring:
            trust_score *= max(0.15, perception_score)
            if passport_state == "valid":
                trust_score *= 1.0
            elif passport_state == "watch":
                trust_score *= max(0.45, 1.0-passport_drift)
            elif passport_state in {"drift_detected", "recalibration_recommended"}:
                trust_score *= max(0.10, 1.0-passport_drift)
            elif passport_state == "unavailable":
                trust_score *= 0.65
            if not calibrated:
                trust_score *= 0.35
        else:
            trust_score = 0.0

        predictive_available = predictive_status == "available"
        memory_active = bool((decision_memory or {}).get("active"))
        memory_available = bool(decision_memory)

        layers = [
            {
                "id": "perception",
                "label": "Perception",
                "state": perception_state,
                "score": round(perception_score, 4),
                "summary": perception.get("summary") or "Camera observability state.",
                "authoritative": False,
            },
            {
                "id": "behaviour",
                "label": "Behaviour",
                "state": str(live.get("state") or "ready").lower(),
                "score": round(cls._number(signal_quality.get("score")), 4),
                "summary": signal_quality.get("summary") or "Live behavioural signal quality.",
                "authoritative": False,
            },
            {
                "id": "learned_model",
                "label": "Learned fatigue model",
                "state": str(decision_engine.get("level") or "standby").lower(),
                "score": round(risk, 4),
                "summary": "Trained model remains the primary learned fatigue evidence.",
                "authoritative": True,
            },
            {
                "id": "personalisation",
                "label": "Personalisation",
                "state": passport_state,
                "score": round(max(0.0, 1.0-passport_drift), 4) if passport_validation else 0.0,
                "summary": (passport_validation or {}).get("summary") or "Passport validation unavailable.",
                "authoritative": False,
            },
            {
                "id": "decision",
                "label": "Decision",
                "state": str(decision_engine.get("level") or "standby").lower(),
                "score": round(decision_confidence, 4),
                "summary": decision_engine.get("summary") or decision_engine.get("reasoning") or "Explainable advisory decision state.",
                "authoritative": True,
            },
            {
                "id": "memory",
                "label": "Memory",
                "state": "recording" if memory_active else "available" if memory_available else "standby",
                "score": 1.0 if memory_available else 0.0,
                "summary": "Decision Memory, Visual Evidence and Near-Miss Memory preserve explainable history.",
                "authoritative": False,
            },
            {
                "id": "predictive",
                "label": "Predictive",
                "state": str(predictive.get("forecast_state") or predictive.get("direction") or predictive_status),
                "score": round(cls._number(predictive.get("confidence")), 4),
                "summary": predictive.get("summary") or "Near-term advisory forecast state.",
                "authoritative": False,
            },
        ]

        blocked = [item for item in reasons if item.get("blocking")]
        if trust_state == "trusted":
            headline = "Guardian's intelligence chain is fully available."
        elif trust_state == "guarded":
            headline = "Guardian can reason with caution; one or more trust layers are degraded."
        elif trust_state == "limited":
            headline = "Some personalised intelligence is restricted by the current trust chain."
        else:
            headline = "Guardian Intelligence is waiting for live monitoring."

        current_action = predictive.get("recommended_action") or decision_engine.get("recommended_action")
        if not current_action:
            current_action = "Use Guardian's existing live monitoring and alerts as the authoritative safety path."

        return {
            "version": cls.VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trust_chain": {
                "state": trust_state,
                "score": round(max(0.0, min(1.0, trust_score)), 4),
                "headline": headline,
                "blocking_reason_count": len(blocked),
                "reason_count": len(reasons),
            },
            "current_assessment": {
                "advisory_risk": round(risk, 4),
                "decision_confidence": round(decision_confidence, 4),
                "decision_level": str(decision_engine.get("level") or "standby"),
                "predictive_available": predictive_available,
                "forecast_direction": str(predictive.get("direction") or "uncertain"),
                "forecast_confidence": round(cls._number(predictive.get("confidence")), 4),
                "recommended_action": current_action,
            },
            "layers": layers,
            "reason_codes": reasons,
            "dependencies": {
                "decision": ["perception", "behaviour", "learned_model", "personalisation", "context"],
                "memory": ["decision", "perception", "personalisation"],
                "predictive": ["decision", "memory", "perception", "personalisation"],
            },
            "cag_context": {
                "schema": "guardian-cag-context-v1",
                "safe_for_explanation": True,
                "monitoring": monitoring,
                "calibrated": calibrated,
                "trust_state": trust_state,
                "advisory_risk": round(risk, 4),
                "decision_level": str(decision_engine.get("level") or "standby"),
                "perception_state": perception_state,
                "passport_state": passport_state,
                "prediction_status": predictive_status,
                "forecast_direction": str(predictive.get("direction") or "uncertain"),
                "reason_codes": [item["code"] for item in reasons],
                "safety_contract": "Explain existing deterministic Guardian outputs; do not invent safety calculations or override alerts.",
            },
            "safety_contract": {
                "trained_model_authoritative": True,
                "existing_alert_path_authoritative": True,
                "changes_model_weights": False,
                "changes_calibration": False,
                "changes_alert_thresholds": False,
                "changes_passport_validation": False,
                "changes_perception_semantics": False,
                "llm_dependency": False,
                "statement": "V9 Safety Intelligence unifies and explains existing Guardian layers; it does not control the live safety path.",
            },
        }
