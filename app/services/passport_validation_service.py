from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any
import json
import math


class PassportValidationService:
    """Retrospective trust assessment for a Personal AI Calibration Passport.

    The service never mutates a baseline, model, camera configuration or alert
    threshold. It only evaluates whether the existing Passport still looks
    representative enough to reuse.
    """

    VERSION = "8.8-passport-validation-v1"

    def __init__(
        self,
        root: Path,
        profile_service,
        passport_service,
        settings_provider,
    ) -> None:
        self.root = root
        self.profile_service = profile_service
        self.passport_service = passport_service
        self.settings_provider = settings_provider

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value if value is not None else default)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    def _sessions(self, profile_name: str) -> list[dict[str, Any]]:
        directory = self.root / "guardian_data" / "decision_memory"
        if not directory.exists():
            return []
        wanted = str(profile_name or "").strip().casefold()
        result: list[dict[str, Any]] = []
        for path in directory.glob("decision_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            current = str(payload.get("driver_profile") or "").strip().casefold()
            if wanted and current != wanted:
                continue
            result.append(payload)
        result.sort(key=lambda item: str(item.get("started_at") or ""))
        return result

    @staticmethod
    def _factor(
        code: str,
        label: str,
        severity: str,
        score: float,
        detail: str,
        observed: Any = None,
        reference: Any = None,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "label": label,
            "severity": severity,
            "score": round(max(0.0, min(1.0, score)), 4),
            "detail": detail,
            "observed": observed,
            "reference": reference,
        }

    def evaluate(self, profile_id: str) -> dict[str, Any]:
        profile = self.profile_service.get(profile_id)
        if not profile:
            raise KeyError("Driver profile was not found.")

        passport = self.passport_service.build(profile_id)
        calibration = profile.get("calibration") or {}
        settings = self.settings_provider() or {}
        sessions = self._sessions(str(profile.get("name") or ""))
        recent_sessions = sessions[-8:]

        samples: list[dict[str, Any]] = []
        for session in recent_sessions:
            samples.extend(session.get("samples", []) or [])
        samples = samples[-250:]

        factors: list[dict[str, Any]] = []
        baseline_ear = self._number(calibration.get("baseline_ear"))
        baseline_tilt = self._number(calibration.get("baseline_tilt"))
        baseline_camera = int(calibration.get("camera_index", 0) or 0)
        current_camera = int(settings.get("camera_index", 0) or 0)

        if baseline_ear <= 0:
            factors.append(
                self._factor(
                    "NO_SAVED_BASELINE",
                    "No saved calibration baseline",
                    "critical",
                    1.0,
                    "The selected profile has no reusable calibration baseline.",
                )
            )

        ear_values = [
            self._number(sample.get("ear"))
            for sample in samples
            if self._number(sample.get("ear")) > 0
            and str(sample.get("perception_state") or "").lower() != "insufficient"
        ]
        if baseline_ear > 0 and len(ear_values) >= 12:
            recent_ear = median(ear_values[-80:])
            difference = abs(recent_ear - baseline_ear) / max(0.0001, baseline_ear)
            if difference >= 0.18:
                severity, score = "critical", min(1.0, 0.70 + difference)
            elif difference >= 0.12:
                severity, score = "high", min(0.85, 0.50 + difference)
            elif difference >= 0.07:
                severity, score = "watch", min(0.60, 0.28 + difference)
            else:
                severity, score = "ok", difference
            factors.append(
                self._factor(
                    "EAR_BASELINE_DRIFT",
                    "EAR baseline drift",
                    severity,
                    score,
                    (
                        f"Recent reliable EAR is {recent_ear:.3f}, "
                        f"{difference:.1%} from the saved baseline."
                    ),
                    observed=round(recent_ear, 6),
                    reference=round(baseline_ear, 6),
                )
            )

        tilt_values = [
            abs(self._number(sample.get("head_tilt")))
            for sample in samples
            if self._number(sample.get("head_tilt")) != 0
            and str(sample.get("perception_state") or "").lower() != "insufficient"
        ]
        if tilt_values and baseline_tilt != 0:
            recent_tilt = median(tilt_values[-80:])
            difference_degrees = abs(recent_tilt - abs(baseline_tilt))
            if difference_degrees >= 12:
                severity, score = "high", 0.68
            elif difference_degrees >= 7:
                severity, score = "watch", 0.42
            else:
                severity, score = "ok", min(0.20, difference_degrees / 35)
            factors.append(
                self._factor(
                    "HEAD_POSE_DRIFT",
                    "Head-pose drift",
                    severity,
                    score,
                    (
                        f"Recent absolute head tilt differs from the saved "
                        f"baseline by {difference_degrees:.1f}°."
                    ),
                    observed=round(recent_tilt, 3),
                    reference=round(abs(baseline_tilt), 3),
                )
            )

        perception_states = [
            str(sample.get("perception_state") or "").lower()
            for sample in samples
            if str(sample.get("perception_state") or "").lower()
            in {"trusted", "degraded", "insufficient"}
        ]
        perception_scores = [
            self._number(sample.get("perception_confidence"))
            for sample in samples
            if self._number(sample.get("perception_confidence")) > 0
        ]
        insufficient_rate = (
            perception_states.count("insufficient") / len(perception_states)
            if perception_states else 0.0
        )
        average_perception = fmean(perception_scores) if perception_scores else 0.0

        if perception_states:
            if insufficient_rate >= 0.45:
                severity, score = "high", 0.72
            elif insufficient_rate >= 0.28:
                severity, score = "watch", 0.48
            else:
                severity, score = "ok", min(0.24, insufficient_rate)
            factors.append(
                self._factor(
                    "PERCEPTION_RELIABILITY",
                    "Recent perception reliability",
                    severity,
                    score,
                    (
                        f"{insufficient_rate:.1%} of recent perception samples "
                        "were INSUFFICIENT."
                    ),
                    observed=round(insufficient_rate, 4),
                    reference="below 28% watch threshold",
                )
            )

        if perception_scores:
            if average_perception < 0.58:
                severity, score = "high", 0.70
            elif average_perception < 0.72:
                severity, score = "watch", 0.43
            else:
                severity, score = "ok", max(0.0, (0.85 - average_perception) * 0.25)
            factors.append(
                self._factor(
                    "PERCEPTION_CONFIDENCE_TREND",
                    "Perception-confidence trend",
                    severity,
                    score,
                    f"Recent average perception confidence is {average_perception:.1%}.",
                    observed=round(average_perception, 4),
                    reference="72% minimum preferred recent mean",
                )
            )

        last_verification = profile.get("last_verification") or {}
        calibrated_at_for_checks = self._parse_time(calibration.get("updated_at"))
        verification_at = self._parse_time(last_verification.get("at"))

        # Verification is calibration-version-specific. A mismatch that occurred
        # before a later full calibration/import is resolved evidence and must
        # not invalidate the replacement baseline.
        verification_is_current = bool(last_verification)
        if calibrated_at_for_checks is not None and verification_at is not None:
            verification_is_current = verification_at >= calibrated_at_for_checks
        elif calibrated_at_for_checks is not None and last_verification:
            # If an old record has no parseable timestamp, do not use it against
            # a known newer calibration.
            verification_is_current = False

        cumulative_fallbacks = int(profile.get("fallback_count", 0) or 0)
        cumulative_verifications = int(profile.get("verification_count", 0) or 0)
        fallback_snapshot = calibration.get("fallback_count_at_calibration")
        verification_snapshot = calibration.get("verification_count_at_calibration")

        if fallback_snapshot is not None and verification_snapshot is not None:
            fallback_count = max(
                0, cumulative_fallbacks - int(fallback_snapshot or 0)
            )
            verification_count = max(
                0, cumulative_verifications - int(verification_snapshot or 0)
            )
        elif calibrated_at_for_checks is not None and not verification_is_current:
            # Legacy calibration created before per-calibration counter snapshots:
            # historical fallback totals cannot be attributed to the new baseline.
            fallback_count = 0
            verification_count = 0
        else:
            fallback_count = cumulative_fallbacks
            verification_count = cumulative_verifications

        total_checks = fallback_count + verification_count
        fallback_rate = fallback_count / total_checks if total_checks else 0.0
        if verification_is_current:
            matched = bool(last_verification.get("matched"))
            difference = abs(self._number(last_verification.get("difference_percent")))
            if not matched:
                factors.append(
                    self._factor(
                        "LAST_VERIFICATION_MISMATCH",
                        "Last quick verification",
                        "high",
                        min(0.85, 0.55 + difference / 100),
                        (
                            f"The most recent quick verification did not match "
                            f"the saved baseline ({difference:.1f}% difference)."
                        ),
                        observed=f"{difference:.1f}% difference",
                        reference="matched verification",
                    )
                )
            else:
                factors.append(
                    self._factor(
                        "LAST_VERIFICATION_MATCH",
                        "Last quick verification",
                        "ok",
                        min(0.18, difference / 100),
                        (
                            f"The most recent quick verification matched with "
                            f"{difference:.1f}% difference."
                        ),
                        observed=f"{difference:.1f}% difference",
                        reference="matched verification",
                    )
                )

        if total_checks >= 3:
            if fallback_rate >= 0.35:
                severity, score = "high", 0.68
            elif fallback_rate >= 0.18:
                severity, score = "watch", 0.42
            else:
                severity, score = "ok", min(0.18, fallback_rate)
            factors.append(
                self._factor(
                    "VERIFICATION_FALLBACK_RATE",
                    "Verification fallback rate",
                    severity,
                    score,
                    (
                        f"{fallback_count} of {total_checks} profile checks "
                        "required full-calibration fallback."
                    ),
                    observed=round(fallback_rate, 4),
                    reference="below 18% watch threshold",
                )
            )

        if current_camera != baseline_camera:
            factors.append(
                self._factor(
                    "CAMERA_CONFIGURATION_CHANGED",
                    "Camera configuration changed",
                    "watch",
                    0.46,
                    (
                        f"Passport baseline was recorded on camera {baseline_camera}, "
                        f"while Settings currently use camera {current_camera}."
                    ),
                    observed=current_camera,
                    reference=baseline_camera,
                )
            )

        calibrated_at = self._parse_time(calibration.get("updated_at"))
        calibration_age_days = None
        if calibrated_at is not None:
            calibration_age_days = max(
                0.0,
                (datetime.now(timezone.utc) - calibrated_at).total_seconds()
                / 86400.0,
            )
            if calibration_age_days >= 90:
                severity, score = "high", 0.66
            elif calibration_age_days >= 45:
                severity, score = "watch", 0.40
            else:
                severity, score = "ok", min(0.18, calibration_age_days / 250)
            factors.append(
                self._factor(
                    "CALIBRATION_AGE",
                    "Calibration age",
                    severity,
                    score,
                    f"The saved calibration is {calibration_age_days:.1f} days old.",
                    observed=round(calibration_age_days, 1),
                    reference="under 45 days preferred",
                )
            )

        recent_sample_count = len(samples)
        if recent_sample_count < 20:
            severity = "watch" if baseline_ear > 0 else "critical"
            score = 0.38 if baseline_ear > 0 else 0.85
            factors.append(
                self._factor(
                    "RECENT_EVIDENCE_LIMITED",
                    "Recent validation evidence",
                    severity,
                    score,
                    (
                        f"Only {recent_sample_count} recent Decision Memory "
                        "samples are available for drift validation."
                    ),
                    observed=recent_sample_count,
                    reference="at least 20 recent samples",
                )
            )

        critical = [item for item in factors if item["severity"] == "critical"]
        high = [item for item in factors if item["severity"] == "high"]
        watch = [item for item in factors if item["severity"] == "watch"]

        weighted_score = max(
            [item["score"] for item in factors if item["severity"] != "ok"]
            or [0.0]
        )
        weighted_score += min(0.18, max(0, len(high) - 1) * 0.07)
        weighted_score += min(0.10, max(0, len(watch) - 1) * 0.03)
        weighted_score = min(1.0, weighted_score)

        if critical or weighted_score >= 0.82 or len(high) >= 3:
            state = "recalibration_recommended"
        elif weighted_score >= 0.62 or len(high) >= 2:
            state = "drift_detected"
        elif weighted_score >= 0.32 or high or watch:
            state = "watch"
        else:
            state = "valid"

        labels = {
            "valid": "VALID",
            "watch": "WATCH",
            "drift_detected": "DRIFT DETECTED",
            "recalibration_recommended": "RECALIBRATION RECOMMENDED",
        }
        action = {
            "valid": "Continue using the saved Passport baseline with normal verification.",
            "watch": "Continue quick verification, but monitor the listed drift factors.",
            "drift_detected": "Treat the Passport as potentially stale and prefer a fresh full calibration soon.",
            "recalibration_recommended": "Run a fresh full calibration before relying on this Passport baseline.",
        }[state]

        important = [
            item for item in factors
            if item["severity"] in {"critical", "high", "watch"}
        ]
        important.sort(
            key=lambda item: (
                {"critical": 3, "high": 2, "watch": 1}.get(item["severity"], 0),
                item["score"],
            ),
            reverse=True,
        )

        return {
            "version": self.VERSION,
            "passport_id": passport.get("passport_id"),
            "profile_id": profile_id,
            "profile_name": profile.get("name"),
            "state": state,
            "label": labels[state],
            "drift_score": round(weighted_score, 4),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "recommended_action": action,
            "summary": (
                f"Passport status is {labels[state]}. "
                f"{len(important)} validation factor(s) currently require attention."
            ),
            "factors": factors,
            "attention_factors": important,
            "recent_evidence": {
                "session_count": len(recent_sessions),
                "sample_count": recent_sample_count,
                "average_perception_confidence": round(average_perception, 4),
                "insufficient_perception_rate": round(insufficient_rate, 4),
                "fallback_rate": round(fallback_rate, 4),
                "calibration_age_days": (
                    round(calibration_age_days, 1)
                    if calibration_age_days is not None
                    else None
                ),
            },
            "automatic_changes_applied": False,
            "safety_boundary": (
                "Passport Validation is advisory. It never rewrites the saved "
                "baseline, model weights, alert thresholds or driver identity."
            ),
        }
