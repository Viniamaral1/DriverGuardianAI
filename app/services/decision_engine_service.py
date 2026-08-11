from __future__ import annotations

import math
import threading
from collections import deque
from datetime import datetime
from typing import Any


class DecisionEngineService:
    """Guardian V8 advisory evidence-fusion layer.

    This service is read-only with respect to Monitoring. It does not modify the
    model, calibration, temporal engine, camera, thresholds, alerts or beeps.
    The trained model remains the primary evidence source.
    """

    MODEL_WEIGHT = 0.82
    PERSONAL_WEIGHT = 0.10
    YAWN_WEIGHT = 0.05
    TILT_WEIGHT = 0.03

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._timeline: deque[dict[str, Any]] = deque(maxlen=24)
        self._last_band: str | None = None
        self._last_confidence: str | None = None

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _personal_deviation(cls, metrics: dict[str, Any]) -> dict[str, Any]:
        baseline = cls._number(metrics.get("baseline_ear"))
        current = cls._number(metrics.get("ear"))
        calibrated = bool(metrics.get("calibration_complete"))

        if not calibrated or baseline <= 0 or current <= 0:
            return {
                "available": False,
                "baseline_ear": baseline,
                "current_ear": current,
                "ratio": None,
                "risk": 0.0,
                "summary": "Waiting for a valid personal EAR baseline.",
            }

        ratio = current / baseline
        risk = cls._clamp((0.90 - ratio) / 0.28)
        if ratio >= 0.90:
            summary = "Current EAR remains close to the personal baseline."
        elif ratio >= 0.78:
            summary = "Current EAR is moderately below the personal baseline."
        else:
            summary = "Current EAR is substantially below the personal baseline."

        return {
            "available": True,
            "baseline_ear": round(baseline, 4),
            "current_ear": round(current, 4),
            "ratio": round(ratio, 4),
            "risk": round(risk, 4),
            "summary": summary,
        }

    @classmethod
    def _model_evidence(cls, metrics: dict[str, Any]) -> float:
        # raw_probability is the unmodified learned-model output.
        return cls._clamp(
            cls._number(
                metrics.get("raw_probability"),
                cls._number(metrics.get("model_risk")),
            )
        )

    @classmethod
    def _yawn_evidence(cls, metrics: dict[str, Any]) -> float:
        return cls._clamp(cls._number(metrics.get("yawn_risk")))

    @classmethod
    def _tilt_evidence(cls, metrics: dict[str, Any]) -> float:
        return cls._clamp(cls._number(metrics.get("tilt_risk")))

    @classmethod
    def _confidence(
        cls,
        metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if not metrics.get("monitoring"):
            return {
                "score": 0.0,
                "level": "standby",
                "summary": "Start Monitoring to calculate decision confidence.",
                "factors": [],
            }

        factors: list[dict[str, Any]] = []
        score = 0.0

        face = bool(metrics.get("face_detected"))
        face_value = 0.30 if face else 0.0
        score += face_value
        factors.append({"label": "Face signal", "status": "available" if face else "missing", "effect": face_value})

        active_model = str(metrics.get("model_status", "")).upper() == "ACTIVE"
        model_value = 0.25 if active_model else 0.10
        score += model_value
        factors.append({"label": "Trained model", "status": "active" if active_model else "fallback/unknown", "effect": model_value})

        calibrated = bool(metrics.get("calibration_complete"))
        calibration_value = 0.15 if calibrated else 0.05
        score += calibration_value
        factors.append({"label": "Personal baseline", "status": "ready" if calibrated else "calibrating", "effect": calibration_value})

        fps = cls._number(metrics.get("fps"))
        fps_value = 0.10 * cls._clamp(fps / 12.0)
        score += fps_value
        factors.append({"label": "Frame rate", "status": f"{fps:.1f} FPS", "effect": round(fps_value, 3)})

        image_available = bool(metrics.get("environment_available"))
        image_score = cls._clamp(cls._number(metrics.get("automatic_perception_score")))
        image_value = 0.20 * image_score if image_available else 0.08
        score += image_value
        factors.append({
            "label": "Image quality",
            "status": str(metrics.get("automatic_perception_quality", "unknown")),
            "effect": round(image_value, 3),
        })

        occlusion = str(((context.get("occlusion") or {}).get("value", "none"))).lower()
        occlusion_penalties = {
            "glasses": 0.03,
            "sunglasses": 0.12,
            "glasses_and_hat": 0.10,
            "eye_occlusion": 0.12,
            "partial_face": 0.16,
            "hand_or_object": 0.18,
            "uncertain": 0.06,
        }
        penalty = occlusion_penalties.get(occlusion, 0.0)
        if penalty > 0:
            score -= penalty
            factors.append({
                "label": "Occlusion context",
                "status": occlusion.replace("_", " "),
                "effect": -round(penalty, 3),
            })

        score = cls._clamp(score)
        if score >= 0.80:
            level = "high"
            summary = "Live perception evidence supports a high-confidence advisory assessment."
        elif score >= 0.58:
            level = "moderate"
            summary = "The advisory assessment is usable, but some perception evidence is limited."
        else:
            level = "limited"
            summary = "Treat the advisory assessment cautiously because perception evidence is limited."

        return {
            "score": round(score, 4),
            "level": level,
            "summary": summary,
            "factors": factors,
        }

    @staticmethod
    def _context_caution(context: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []

        def value(name: str, default: str = "unknown") -> str:
            item = context.get(name, {}) or {}
            return str(item.get("value", default) or default).lower()

        weather = value("weather")
        road = value("road_condition")
        light = value("external_light")
        occlusion = value("occlusion", "none")

        if weather in {"rain", "snow", "fog", "storm"}:
            reasons.append(weather)
        if road in {"wet", "snow_or_ice", "icy", "poor"}:
            reasons.append(road.replace("_", " "))
        if light in {"night", "dark", "dusk"}:
            reasons.append(light)
        if occlusion in {
            "sunglasses", "glasses_and_hat", "eye_occlusion",
            "partial_face", "hand_or_object",
        }:
            reasons.append(occlusion.replace("_", " "))

        level = "high" if len(reasons) >= 3 else "elevated" if reasons else "normal"
        return {
            "level": level,
            "reasons": reasons,
            "summary": (
                "Extra driving caution is warranted because of " + ", ".join(reasons) + "."
                if reasons
                else "No additional context caution is currently identified."
            ),
        }

    @staticmethod
    def _band(score: float) -> str:
        if score >= 0.82:
            return "high"
        if score >= 0.62:
            return "elevated"
        if score >= 0.42:
            return "watch"
        return "low"

    @staticmethod
    def _action(band: str, confidence: str, caution: str) -> str:
        if confidence == "limited":
            return "Perception confidence is limited. Keep Guardian's existing safety alerts primary and improve camera visibility if safe."
        if band == "high":
            return "High advisory risk. Follow Guardian's existing alert and plan a safe break."
        if band == "elevated":
            return "Risk evidence is elevated. Continue close monitoring and prepare for a safe break if it persists."
        if band == "watch":
            return "Some fatigue evidence is present. Continue monitoring for sustained change."
        if caution in {"elevated", "high"}:
            return "Fatigue evidence is currently low, but driving context warrants additional caution."
        return "Current evidence is stable. Continue normal monitoring."

    def _record(
        self,
        monitoring: bool,
        band: str,
        confidence: str,
        score: float,
        reason: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if not monitoring:
                self._last_band = None
                self._last_confidence = None
                return list(self._timeline)

            if band != self._last_band or confidence != self._last_confidence:
                self._timeline.appendleft({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "risk_band": band,
                    "confidence": confidence,
                    "score": round(score, 4),
                    "reason": reason,
                })
                self._last_band = band
                self._last_confidence = confidence

            return list(self._timeline)

    @staticmethod
    def safety_contract() -> dict[str, Any]:
        return {
            "controls_alerts": False,
            "controls_camera": False,
            "changes_model_probability": False,
            "changes_calibration": False,
            "validated_for_alert_control": False,
            "description": (
                "V8 is an explainable advisory layer. The existing V3 monitoring "
                "and alert path remains authoritative."
            ),
        }

    def snapshot(
        self,
        metrics: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        monitoring = bool(metrics.get("monitoring"))
        caution = self._context_caution(context)

        if not monitoring:
            return {
                "available": True,
                "active": False,
                "mode": "advisory",
                "risk_score": 0.0,
                "risk_band": "standby",
                "confidence": self._confidence(metrics, context),
                "evidence": [],
                "context_caution": caution,
                "explanation": "Start Monitoring to activate the V8 advisory engine.",
                "action": "Start Monitoring when ready.",
                "timeline": list(self._timeline),
                "safety_contract": self.safety_contract(),
            }

        model = self._model_evidence(metrics)
        personal = self._personal_deviation(metrics)
        yawn = self._yawn_evidence(metrics)
        tilt = self._tilt_evidence(metrics)

        personal_weight = self.PERSONAL_WEIGHT if personal["available"] else 0.0
        model_weight = self.MODEL_WEIGHT + (self.PERSONAL_WEIGHT - personal_weight)

        score = self._clamp(
            model * model_weight
            + personal["risk"] * personal_weight
            + yawn * self.YAWN_WEIGHT
            + tilt * self.TILT_WEIGHT
        )

        evidence = [
            {
                "key": "trained_model",
                "label": "Trained model",
                "value": round(model, 4),
                "weight": round(model_weight, 4),
                "contribution": round(model * model_weight, 4),
                "role": "primary",
                "explanation": "Unmodified core-behaviour model probability.",
            },
            {
                "key": "personal_baseline",
                "label": "Personal EAR deviation",
                "value": round(personal["risk"], 4),
                "weight": round(personal_weight, 4),
                "contribution": round(personal["risk"] * personal_weight, 4),
                "role": "supporting",
                "explanation": personal["summary"],
            },
            {
                "key": "yawn",
                "label": "Yawn evidence",
                "value": round(yawn, 4),
                "weight": self.YAWN_WEIGHT,
                "contribution": round(yawn * self.YAWN_WEIGHT, 4),
                "role": "supporting",
                "explanation": "Observed yawn evidence.",
            },
            {
                "key": "head_pose",
                "label": "Head-pose evidence",
                "value": round(tilt, 4),
                "weight": self.TILT_WEIGHT,
                "contribution": round(tilt * self.TILT_WEIGHT, 4),
                "role": "supporting",
                "explanation": "Observed head-tilt evidence.",
            },
        ]

        strongest = max(evidence, key=lambda item: item["contribution"])
        confidence = self._confidence(metrics, context)
        band = self._band(score)
        timeline = self._record(
            monitoring,
            band,
            confidence["level"],
            score,
            strongest["label"],
        )

        return {
            "available": True,
            "active": True,
            "mode": "advisory",
            "risk_score": round(score, 4),
            "risk_band": band,
            "confidence": confidence,
            "evidence": evidence,
            "personal_baseline": personal,
            "context_caution": caution,
            "explanation": (
                f"{strongest['label']} is currently the strongest advisory evidence. "
                "Personal calibration supports the learned model instead of replacing it."
            ),
            "action": self._action(band, confidence["level"], caution["level"]),
            "legacy_reference": {
                "raw_model_probability": round(model, 4),
                "existing_personalized_probability": round(
                    self._number(metrics.get("decision_probability")), 4
                ),
                "existing_smoothed_probability": round(
                    self._number(metrics.get("fatigue_probability")), 4
                ),
                "note": "V8 observes these values but does not feed anything back into Monitoring.",
            },
            "timeline": timeline,
            "safety_contract": self.safety_contract(),
        }
