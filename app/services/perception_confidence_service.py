from __future__ import annotations

from typing import Any


class PerceptionConfidenceService:
    """Explain whether current computer-vision evidence is observable enough.

    This service is metadata only. It does not modify fatigue probabilities,
    model weights, calibration, TemporalStateEngine or AlertManager.
    """

    VERSION = "8.6-perception-confidence-v1"

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            return float(value if value is not None else default)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def snapshot(cls, metrics: dict[str, Any]) -> dict[str, Any]:
        monitoring = bool(metrics.get("monitoring"))
        if not monitoring:
            return {
                "version": cls.VERSION,
                "state": "standby",
                "score": 0.0,
                "summary": "Start Monitoring to calculate perception confidence.",
                "observation_mode": "standby",
                "evidence_policy": "standby",
                "absence_interpretation": "No live visual observation is active.",
                "can_trust_presence": False,
                "can_trust_absence": False,
                "components": {},
                "reason_codes": [],
                "affected_regions": [],
                "safety_boundary": "Perception Confidence is advisory metadata and never changes fatigue or alert decisions.",
            }

        face = bool(metrics.get("face_detected"))
        environment_available = bool(metrics.get("environment_available"))
        image_score = cls._clip(cls._number(metrics.get("automatic_perception_score")))
        image_quality = str(metrics.get("automatic_perception_quality") or "standby").lower()
        eye_visibility = cls._clip(cls._number(metrics.get("eye_visibility_score")))
        occlusion = str(metrics.get("automatic_occlusion") or "unknown").lower()
        raw_occlusion = str(metrics.get("raw_automatic_occlusion") or "unknown").lower()
        occlusion_confidence = cls._clip(cls._number(metrics.get("automatic_occlusion_confidence")))
        raw_occlusion_confidence = cls._clip(cls._number(metrics.get("raw_automatic_occlusion_confidence")))
        temporal_window = max(0, int(cls._number(metrics.get("occlusion_temporal_window"))))
        fps = max(0.0, cls._number(metrics.get("fps")))
        head_tilt = abs(cls._number(metrics.get("head_tilt")))
        underexposed = cls._clip(cls._number(metrics.get("underexposed_ratio")))
        overexposed = cls._clip(cls._number(metrics.get("overexposed_ratio")))
        glare = cls._clip(cls._number(metrics.get("glare_ratio")))
        sharpness = str(metrics.get("automatic_sharpness") or "unknown").lower()

        reasons: list[dict[str, Any]] = []
        affected: set[str] = set()

        def reason(code: str, severity: str, region: str, message: str, value: float | str | None = None) -> None:
            reasons.append({
                "code": code,
                "severity": severity,
                "region": region,
                "message": message,
                "value": value,
            })
            affected.add(region)

        face_component = 1.0 if face else 0.0
        image_component = image_score if environment_available else 0.35
        eye_component = eye_visibility if face else 0.0
        occlusion_component = occlusion_confidence if face else 0.0
        temporal_component = min(1.0, temporal_window / 6.0) if face else 0.0
        fps_component = cls._clip(fps / 18.0)
        geometry_component = 1.0 if head_tilt <= 16 else cls._clip(1.0 - ((head_tilt - 16) / 28.0))

        if not face:
            reason("FACE_NOT_DETECTED", "critical", "face", "No stable face landmarks are available.", 0.0)
        if not environment_available:
            reason("IMAGE_ANALYSIS_UNAVAILABLE", "warning", "frame", "Image-quality analysis is not currently available.", None)
        elif image_quality == "limited" or image_score < 0.50:
            reason("IMAGE_QUALITY_LIMITED", "critical", "frame", "Frame quality is too limited for strong visual confidence.", round(image_score, 4))
        elif image_quality == "moderate" or image_score < 0.70:
            reason("IMAGE_QUALITY_MODERATE", "warning", "frame", "Frame quality is usable but degraded.", round(image_score, 4))

        if underexposed > 0.45:
            reason("UNDEREXPOSED", "critical" if underexposed > 0.60 else "warning", "frame", "A large part of the frame is underexposed.", round(underexposed, 4))
        if overexposed > 0.34:
            reason("OVEREXPOSED", "critical" if overexposed > 0.50 else "warning", "frame", "A large part of the frame is overexposed.", round(overexposed, 4))
        if glare > 0.14:
            reason("GLARE", "warning", "frame", "Strong highlights may obscure facial detail.", round(glare, 4))
        if sharpness == "blurred":
            reason("BLUR", "critical", "frame", "Image sharpness is too low for reliable fine facial detail.", sharpness)
        elif sharpness == "moderate":
            reason("SHARPNESS_MODERATE", "warning", "frame", "Image sharpness is moderate.", sharpness)

        if face and eye_visibility < 0.28:
            reason("EYE_VISIBILITY_INSUFFICIENT", "critical", "eyes", "The eye region is not visible enough to interpret missing eye cues as negative evidence.", round(eye_visibility, 4))
        elif face and eye_visibility < 0.58:
            reason("EYE_VISIBILITY_REDUCED", "warning", "eyes", "Eye visibility is reduced.", round(eye_visibility, 4))

        if raw_occlusion in {"unknown", "uncertain"} and raw_occlusion_confidence < 0.55:
            reason("OCCLUSION_UNCERTAIN", "warning", "eyes", "Current eye-region occlusion evidence is uncertain.", round(raw_occlusion_confidence, 4))
        if occlusion in {"sunglasses", "eye_occlusion"}:
            reason("EYE_REGION_OCCLUDED", "critical" if eye_visibility < 0.40 else "warning", "eyes", f"Stable eye-region analysis indicates {occlusion.replace('_', ' ')}.", round(occlusion_confidence, 4))
        elif occlusion == "partial_face":
            reason("PARTIAL_FACE", "critical", "face", "Part of the face is close to or outside the camera boundary.", round(occlusion_confidence, 4))

        if fps < 7:
            reason("FRAME_RATE_LOW", "critical", "temporal", "Frame rate is too low for strong temporal visual confidence.", round(fps, 2))
        elif fps < 11:
            reason("FRAME_RATE_REDUCED", "warning", "temporal", "Frame rate is reduced.", round(fps, 2))

        if head_tilt > 28:
            reason("HEAD_POSE_EXTREME", "critical", "face", "Head pose is outside the preferred observation geometry.", round(head_tilt, 2))
        elif head_tilt > 18:
            reason("HEAD_POSE_OFF_AXIS", "warning", "face", "Head pose is moderately off-axis.", round(head_tilt, 2))

        score = (
            face_component * 0.20
            + image_component * 0.24
            + eye_component * 0.24
            + occlusion_component * 0.12
            + temporal_component * 0.08
            + fps_component * 0.08
            + geometry_component * 0.04
        )
        score = cls._clip(score)

        critical_codes = {item["code"] for item in reasons if item["severity"] == "critical"}
        hard_insufficient = (
            not face
            or "EYE_VISIBILITY_INSUFFICIENT" in critical_codes
            or "PARTIAL_FACE" in critical_codes
            or ("IMAGE_QUALITY_LIMITED" in critical_codes and score < 0.52)
            or ("FRAME_RATE_LOW" in critical_codes and score < 0.50)
        )

        if hard_insufficient or score < 0.43:
            state = "insufficient"
            observation_mode = "insufficient"
            evidence_policy = "do_not_treat_absence_as_negative"
            can_trust_presence = False
            can_trust_absence = False
            summary = "Visual perception is insufficient; absence of a detected fatigue cue should not be interpreted as evidence that the cue is absent."
        elif score < 0.72 or any(item["severity"] == "warning" for item in reasons):
            state = "degraded"
            observation_mode = "observable_with_caution"
            evidence_policy = "presence_more_reliable_than_absence"
            can_trust_presence = True
            can_trust_absence = False
            summary = "Visual evidence is available but degraded; positive cues can be reported with caution, while missing cues should not be treated as strong negative evidence."
        else:
            state = "trusted"
            observation_mode = "observable"
            evidence_policy = "positive_and_negative_evidence_observable"
            can_trust_presence = True
            can_trust_absence = True
            summary = "Visual conditions support interpreting both detected cues and the absence of observed cues with normal Guardian confidence."

        absence_interpretation = (
            "A missing visual cue can be treated as meaningful negative evidence at the current observation quality."
            if can_trust_absence
            else "A missing visual cue may reflect limited visibility; do not treat non-detection as strong negative evidence."
        )

        return {
            "version": cls.VERSION,
            "state": state,
            "score": round(score, 4),
            "summary": summary,
            "observation_mode": observation_mode,
            "evidence_policy": evidence_policy,
            "absence_interpretation": absence_interpretation,
            "can_trust_presence": can_trust_presence,
            "can_trust_absence": can_trust_absence,
            "components": {
                "face_presence": round(face_component, 4),
                "image_quality": round(image_component, 4),
                "eye_visibility": round(eye_component, 4),
                "occlusion_reliability": round(occlusion_component, 4),
                "temporal_stability": round(temporal_component, 4),
                "frame_rate": round(fps_component, 4),
                "observation_geometry": round(geometry_component, 4),
            },
            "reason_codes": reasons,
            "affected_regions": sorted(affected),
            "safety_boundary": "Perception Confidence describes visual observability only. It does not change fatigue probabilities, model output or Guardian alerts.",
        }
