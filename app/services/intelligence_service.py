from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any, TYPE_CHECKING

from app.services.decision_engine_service import DecisionEngineService
from app.services.decision_memory_service import DecisionMemoryService

if TYPE_CHECKING:
    from app.services.app_state import GuardianState


class IntelligenceService:
    """Read-only intelligence layer around live metrics and Edge Memory.

    This service does not alter monitoring, model predictions, thresholds,
    alerts, camera ownership, or calibration. It explains the existing outputs
    and produces a transparent near-term outlook.
    """

    def __init__(
        self,
        state: "GuardianState",
        decision_memory: DecisionMemoryService | None = None,
    ) -> None:
        self.state = state
        self.decision_engine = DecisionEngineService()
        self.decision_memory = decision_memory or DecisionMemoryService(state.root)

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            return float(value or default)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _period(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"

    @staticmethod
    def _automatic_light(hour: int) -> str:
        if 7 <= hour < 18:
            return "daylight"
        if 5 <= hour < 7 or 18 <= hour < 21:
            return "dusk"
        return "night"

    def _edge_snapshot(self, *, refresh_reports: bool = True) -> dict[str, Any]:
        service = self.state.edge_memory_service
        if service is None:
            return {
                "available": False,
                "insights": {},
                "context": {},
                "recent_sessions": [],
            }
        if refresh_reports:
            service.refresh_from_reports()
        return service.snapshot()

    def _contributions(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        raw = {
            "Model evidence": self._number(metrics.get("model_risk")),
            "Eye evidence": self._number(metrics.get("eye_risk")),
            "Yawn evidence": self._number(metrics.get("yawn_risk")),
            "Head-pose evidence": self._number(metrics.get("tilt_risk")),
        }
        total = sum(max(0.0, value) for value in raw.values())
        if total <= 0:
            return [
                {
                    "label": label,
                    "value": round(value, 4),
                    "share": 0.0,
                }
                for label, value in raw.items()
            ]

        return sorted(
            [
                {
                    "label": label,
                    "value": round(value, 4),
                    "share": round(max(0.0, value) / total, 4),
                }
                for label, value in raw.items()
            ],
            key=lambda item: item["share"],
            reverse=True,
        )

    def _signal_quality(self, metrics: dict[str, Any]) -> dict[str, Any]:
        monitoring = bool(metrics.get("monitoring"))
        face = bool(metrics.get("face_detected"))
        fps = self._number(metrics.get("fps"))
        calibrated = bool(metrics.get("calibration_complete"))
        image_available = bool(metrics.get("environment_available"))
        image_score = self._number(metrics.get("automatic_perception_score"))
        image_quality = str(
            metrics.get("automatic_perception_quality", "standby")
        )
        image_summary = str(
            metrics.get(
                "automatic_perception_summary",
                "No camera image-quality assessment is available.",
            )
        )

        if not monitoring:
            return {
                "level": "standby",
                "score": 0.0,
                "summary": "Start Monitoring to calculate live perception quality.",
                "image_quality": "standby",
                "image_score": 0.0,
            }

        score = 0.0
        score += 0.30 if face else 0.0
        score += 0.22 if calibrated else 0.0
        score += min(0.18, max(0.0, fps) / 110.0)
        score += min(0.30, image_score * 0.30) if image_available else 0.0

        if not face:
            level = "limited"
            summary = "No stable face signal is available."
        elif not calibrated:
            level = "calibrating"
            summary = "Face detected; personal calibration is still in progress."
        elif image_available and image_quality == "limited":
            level = "limited"
            summary = image_summary
        elif fps < 8 or (image_available and image_quality == "moderate"):
            level = "moderate"
            summary = image_summary
        else:
            level = "good"
            summary = (
                "Face, calibration, frame-rate and image-quality signals "
                "are available."
            )

        return {
            "level": level,
            "score": round(min(1.0, score), 3),
            "summary": summary,
            "image_quality": image_quality,
            "image_score": round(image_score, 3),
        }

    def _baseline_comparison(
        self,
        metrics: dict[str, Any],
        insights: dict[str, Any],
    ) -> dict[str, Any]:
        current = self._number(metrics.get("baseline_ear"))
        historical = self._number(insights.get("average_baseline_ear"))

        if current <= 0:
            return {
                "available": False,
                "current": current,
                "historical": historical,
                "difference_percent": 0.0,
                "summary": "Complete calibration to compare the current baseline.",
            }

        if historical <= 0:
            return {
                "available": False,
                "current": current,
                "historical": historical,
                "difference_percent": 0.0,
                "summary": "No historical baseline is available yet.",
            }

        difference = (current - historical) / historical
        if abs(difference) < 0.08:
            summary = "The current eye baseline is close to the stored personal average."
        elif difference < 0:
            summary = "The current eye baseline is lower than the stored personal average."
        else:
            summary = "The current eye baseline is higher than the stored personal average."

        return {
            "available": True,
            "current": round(current, 4),
            "historical": round(historical, 4),
            "difference_percent": round(difference * 100.0, 1),
            "summary": summary,
        }

    def _outlook(
        self,
        metrics: dict[str, Any],
        insights: dict[str, Any],
    ) -> dict[str, Any]:
        current = self._number(metrics.get("fatigue_probability"))
        decision = self._number(metrics.get("decision_probability"))
        historical = self._number(insights.get("average_risk"))
        duration_minutes = self._number(metrics.get("session_seconds")) / 60.0
        alerts = int(self._number(metrics.get("alert_count")))
        state = str(metrics.get("state", "READY")).upper()

        score = (
            current * 0.52
            + decision * 0.23
            + historical * 0.15
            + min(1.0, duration_minutes / 120.0) * 0.10
        )
        if alerts:
            score = min(1.0, score + min(0.18, alerts * 0.05))
        if state == "CRITICAL":
            score = max(score, 0.90)
        elif state == "WARNING":
            score = max(score, 0.68)

        if not metrics.get("monitoring"):
            level = "standby"
            horizon = "No live forecast"
            summary = (
                "Start Monitoring to produce a near-term outlook. Historical "
                "patterns remain available in Edge Intelligence."
            )
        elif not metrics.get("calibration_complete"):
            level = "calibrating"
            horizon = "Waiting for calibration"
            summary = "The outlook will activate after the personal baseline is ready."
        elif score >= 0.82:
            level = "critical"
            horizon = "Immediate action"
            summary = "Current evidence supports an immediate safe break recommendation."
        elif score >= 0.65:
            level = "warning"
            horizon = "Break recommended soon"
            summary = "Risk is elevated. Plan a safe break rather than waiting for escalation."
        elif score >= 0.45:
            level = "watch"
            horizon = "Watch next 15–30 minutes"
            summary = "Risk is moderate or increasing; continue monitoring closely."
        else:
            level = "stable"
            horizon = "No near-term escalation indicated"
            summary = "Current and historical evidence do not indicate near-term escalation."

        return {
            "level": level,
            "score": round(score, 4),
            "horizon": horizon,
            "summary": summary,
            "method": (
                "Transparent rule-based outlook using live fatigue probability, "
                "decision probability, session duration, alerts and historical "
                "average risk. It is not a separate trained medical prediction."
            ),
        }

    def snapshot(self, *, record_memory: bool = False) -> dict[str, Any]:
        metrics = self.state.metrics()
        # Browser views can refresh report-derived history. The automatic
        # Decision Memory sampler reuses the current Edge snapshot to avoid a
        # filesystem/report scan every sampling interval.
        edge = self._edge_snapshot(refresh_reports=not record_memory)
        insights = edge.get("insights", {}) or {}
        resolved_context = edge.get("context", {}) or {}
        manual_context = edge.get("manual_context", {}) or {}

        now = datetime.now()
        period = self._period(now.hour)
        automatic_light = self._automatic_light(now.hour)

        contributions = self._contributions(metrics)
        dominant = contributions[0] if contributions else None
        dominant_text = (
            f"{dominant['label']} is currently the strongest available evidence."
            if dominant and dominant["share"] > 0
            else "No calibrated risk contribution is currently active."
        )

        def resolved_value(name: str, fallback: str = "unknown") -> str:
            item = resolved_context.get(name, {}) or {}
            return str(item.get("value", fallback) or fallback)

        automatic_occlusion = str(metrics.get("automatic_occlusion", "unknown") or "unknown")
        automatic_occlusion_confidence = self._number(
            metrics.get("automatic_occlusion_confidence")
        )
        manual_occlusion = resolved_value("occlusion", "none")
        use_manual_occlusion = bool(resolved_context.get("manual_override"))

        if use_manual_occlusion:
            effective_occlusion = manual_occlusion
            occlusion_source = resolved_context.get("occlusion", {}) or {}
        elif automatic_occlusion not in {"", "unknown", "uncertain"} and automatic_occlusion_confidence >= 0.58:
            effective_occlusion = automatic_occlusion
            occlusion_source = {
                "value": automatic_occlusion,
                "source": "Automatic eye-region analysis",
                "confidence": round(automatic_occlusion_confidence, 4),
                "updated_at": now.isoformat(timespec="seconds"),
                "fresh": True,
                "summary": metrics.get("automatic_occlusion_summary", ""),
            }
        else:
            effective_occlusion = "none" if automatic_occlusion == "none" else manual_occlusion
            occlusion_source = {
                "value": effective_occlusion,
                "source": (
                    "Automatic eye-region analysis"
                    if automatic_occlusion == "none" and automatic_occlusion_confidence >= 0.58
                    else "Manual/default context"
                ),
                "confidence": (
                    round(automatic_occlusion_confidence, 4)
                    if automatic_occlusion == "none"
                    else 0.0
                ),
                "updated_at": now.isoformat(timespec="seconds"),
                "fresh": bool(metrics.get("monitoring")),
                "summary": metrics.get("automatic_occlusion_summary", ""),
            }

        effective_context = {
            "weather": resolved_value("weather"),
            "road_condition": resolved_value("road_condition"),
            "external_light": resolved_value("external_light", automatic_light),
            "cabin_light": resolved_value("cabin_light"),
            "occlusion": effective_occlusion,
            "manual_occlusion": manual_occlusion,
            "resolved_occlusion": effective_occlusion,
            "occlusion_source": str(occlusion_source.get("source") or "unknown"),
            "automatic_time_period": period,
            "automatic_external_light": resolved_value("external_light", automatic_light),
            "local_time": now.isoformat(timespec="seconds"),
            "sources": {
                **{
                    key: resolved_context.get(key, {})
                    for key in ("weather", "road_condition", "external_light", "cabin_light", "local_period")
                },
                "occlusion": occlusion_source,
            },
            "automatic_occlusion": {
                "value": automatic_occlusion,
                "confidence": round(automatic_occlusion_confidence, 4),
                "summary": metrics.get("automatic_occlusion_summary", ""),
                "eye_visibility_score": self._number(metrics.get("eye_visibility_score")),
                "eye_region_brightness_ratio": self._number(
                    metrics.get("eye_region_brightness_ratio")
                ),
                "eye_dark_ratio": self._number(metrics.get("eye_dark_ratio")),
                "eye_edge_density": self._number(metrics.get("eye_edge_density")),
            },
            "automatic_weather": resolved_context.get("automatic_weather", {}),
            "location": resolved_context.get("location", ""),
            "manual_override": bool(resolved_context.get("manual_override")),
        }

        caution_reasons: list[str] = []
        if automatic_light in {"night", "dusk"}:
            caution_reasons.append(f"automatic {automatic_light} time context")
        if effective_context.get("weather") in {"rain", "snow", "fog"}:
            caution_reasons.append(str(effective_context.get("weather")))
        if effective_context.get("road_condition") in {"wet", "snow_or_ice"}:
            caution_reasons.append(
                str(effective_context.get("road_condition")).replace("_", " ")
            )
        if effective_context.get("occlusion") in {"sunglasses", "glasses_and_hat"}:
            caution_reasons.append(
                str(effective_context.get("occlusion")).replace("_", " ")
            )

        decision_context = dict(resolved_context)
        decision_context["occlusion"] = occlusion_source
        decision_engine = self.decision_engine.snapshot(
            metrics,
            decision_context,
        )

        payload = {
            "available": True,
            "decision_engine": decision_engine,
            "generated_at": now.isoformat(timespec="seconds"),
            "live": {
                "monitoring": bool(metrics.get("monitoring")),
                "state": metrics.get("state", "READY"),
                "risk": self._number(metrics.get("fatigue_probability")),
                "decision_probability": self._number(
                    metrics.get("decision_probability")
                ),
                "calibrated": bool(metrics.get("calibration_complete")),
                "session_seconds": int(self._number(metrics.get("session_seconds"))),
                "alert_count": int(self._number(metrics.get("alert_count"))),
            },
            "explanation": {
                "dominant": dominant,
                "summary": dominant_text,
                "contributions": contributions,
            },
            "signal_quality": self._signal_quality(metrics),
            "baseline_comparison": self._baseline_comparison(metrics, insights),
            "outlook": self._outlook(metrics, insights),
            "context": effective_context,
            "environment": {
                "available": bool(metrics.get("environment_available")),
                "brightness": self._number(metrics.get("frame_brightness")),
                "contrast": self._number(metrics.get("frame_contrast")),
                "blur_variance": self._number(
                    metrics.get("frame_blur_variance")
                ),
                "underexposed_ratio": self._number(
                    metrics.get("underexposed_ratio")
                ),
                "overexposed_ratio": self._number(
                    metrics.get("overexposed_ratio")
                ),
                "glare_ratio": self._number(metrics.get("glare_ratio")),
                "cabin_light": metrics.get(
                    "automatic_cabin_light",
                    "unknown",
                ),
                "sharpness": metrics.get(
                    "automatic_sharpness",
                    "unknown",
                ),
                "quality": metrics.get(
                    "automatic_perception_quality",
                    "standby",
                ),
                "quality_score": self._number(
                    metrics.get("automatic_perception_score")
                ),
                "summary": metrics.get(
                    "automatic_perception_summary",
                    "Start Monitoring to analyse camera image quality.",
                ),
                "scope_note": (
                    "This layer measures image conditions only. It does not "
                    "identify driver identity. Automatic eye-region occlusion is conservative; clear glasses and hats may still require manual context."
                ),
            },
            "journey_caution": {
                "elevated": bool(caution_reasons),
                "reasons": caution_reasons,
                "summary": (
                    "Journey context increases caution: "
                    + ", ".join(caution_reasons)
                    if caution_reasons
                    else "No additional journey-context caution is currently recorded."
                ),
            },
            "history": {
                "session_count": int(insights.get("session_count", 0) or 0),
                "average_risk": self._number(insights.get("average_risk")),
                "highest_risk": self._number(insights.get("highest_risk")),
                "highest_risk_period": insights.get(
                    "highest_risk_period",
                    "not enough data",
                ),
                "summary": insights.get(
                    "summary",
                    "No historical insight is available.",
                ),
            },
            "safety_note": (
                "Guardian Intelligence explains and contextualises the existing "
                "monitoring output. Current live signals remain the primary "
                "safety input."
            ),
        }

        if record_memory:
            payload["decision_memory"] = self.decision_memory.record(
                metrics,
                payload,
            )
        else:
            payload["decision_memory"] = self.decision_memory.status()
        return payload
