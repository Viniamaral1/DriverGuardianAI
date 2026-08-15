from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
import json
import math
import time


class PredictiveGuardianService:
    """Near-term advisory forecasting over Guardian's existing evidence.

    This is not a medical predictor and does not affect the live safety path.
    It combines a short live trajectory with local Decision Memory patterns,
    Perception Confidence and Passport Validation trust.
    """

    VERSION = "8.9-predictive-guardian-v1"
    ELEVATED_THRESHOLD = 0.65

    def __init__(self, root: Path) -> None:
        self.root = root
        self._live: deque[dict[str, float]] = deque(maxlen=20)

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value if value is not None else default)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

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
            samples = payload.get("samples", []) or []
            if samples:
                result.append(payload)
        result.sort(key=lambda item: str(item.get("started_at") or ""))
        return result

    @staticmethod
    def _period(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"

    def _history(self, profile_name: str, current_period: str) -> dict[str, Any]:
        sessions = self._sessions(profile_name)
        recent = sessions[-20:]
        first_elevated_minutes: list[float] = []
        peak_risks: list[float] = []
        recovery_seconds: list[float] = []
        same_period_peaks: list[float] = []
        alert_sessions = 0

        for session in recent:
            samples = session.get("samples", []) or []
            if not samples:
                continue

            risks = [self._number(s.get("advisory_risk")) for s in samples]
            peak = max(risks, default=0.0)
            peak_risks.append(peak)

            if max(int(self._number(s.get("alert_count"))) for s in samples) > 0:
                alert_sessions += 1

            started = str(session.get("started_at") or "")
            try:
                hour = datetime.fromisoformat(started).hour
                if self._period(hour) == current_period:
                    same_period_peaks.append(peak)
            except ValueError:
                pass

            first_index = next(
                (
                    i for i, sample in enumerate(samples)
                    if self._number(sample.get("advisory_risk"))
                    >= self.ELEVATED_THRESHOLD
                ),
                None,
            )
            if first_index is not None:
                seconds = self._number(
                    samples[first_index].get("elapsed_seconds")
                )
                if seconds >= 0:
                    first_elevated_minutes.append(seconds / 60.0)

                peak_index = max(
                    range(len(samples)),
                    key=lambda i: self._number(
                        samples[i].get("advisory_risk")
                    ),
                )
                peak_risk = self._number(
                    samples[peak_index].get("advisory_risk")
                )
                if peak_risk >= self.ELEVATED_THRESHOLD:
                    for later in samples[peak_index + 1:]:
                        if self._number(later.get("advisory_risk")) <= 0.45:
                            recovery_seconds.append(
                                max(
                                    0.0,
                                    self._number(
                                        later.get("elapsed_seconds")
                                    )
                                    - self._number(
                                        samples[peak_index].get(
                                            "elapsed_seconds"
                                        )
                                    ),
                                )
                            )
                            break

        median_first_elevated_seconds = (
            median(first_elevated_minutes) * 60.0
            if first_elevated_minutes else None
        )

        return {
            "session_count": len(recent),
            "median_first_elevated_seconds": (
                round(median_first_elevated_seconds, 1)
                if median_first_elevated_seconds is not None else None
            ),
            "median_first_elevated_minutes": (
                round(median_first_elevated_seconds / 60.0, 3)
                if median_first_elevated_seconds is not None else None
            ),
            "median_peak_risk": (
                round(median(peak_risks), 4) if peak_risks else 0.0
            ),
            "same_period_peak_risk": (
                round(median(same_period_peaks), 4)
                if same_period_peaks else 0.0
            ),
            "same_period_session_count": len(same_period_peaks),
            "alert_session_rate": (
                round(alert_sessions / len(recent), 4)
                if recent else 0.0
            ),
            "median_recovery_seconds": (
                round(median(recovery_seconds), 1)
                if recovery_seconds else None
            ),
        }

    def _record_live(
        self,
        *,
        monitoring: bool,
        risk: float,
        session_seconds: float,
    ) -> None:
        if not monitoring:
            self._live.clear()
            return
        now = time.monotonic()
        if self._live and session_seconds < self._live[-1]["session_seconds"]:
            self._live.clear()
        if (
            not self._live
            or session_seconds - self._live[-1]["session_seconds"] >= 1.0
            or now - self._live[-1]["clock"] >= 1.0
        ):
            self._live.append(
                {
                    "clock": now,
                    "session_seconds": session_seconds,
                    "risk": max(0.0, min(1.0, risk)),
                }
            )

    def _trajectory(self) -> dict[str, Any]:
        points = list(self._live)
        if len(points) < 3:
            return {
                "direction": "uncertain",
                "slope_per_minute": 0.0,
                "live_points": len(points),
            }

        recent = points[-8:]
        x0 = recent[0]["session_seconds"]
        xs = [p["session_seconds"] - x0 for p in recent]
        ys = [p["risk"] for p in recent]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        slope_per_second = (
            sum(
                (x - x_mean) * (y - y_mean)
                for x, y in zip(xs, ys)
            ) / denom
            if denom > 0 else 0.0
        )
        slope = slope_per_second * 60.0

        if slope >= 0.08:
            direction = "rising"
        elif slope <= -0.08:
            direction = "falling"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "slope_per_minute": round(slope, 4),
            "live_points": len(points),
        }

    def snapshot(
        self,
        *,
        metrics: dict[str, Any],
        decision_engine: dict[str, Any],
        perception: dict[str, Any],
        passport_validation: dict[str, Any] | None,
        history_insights: dict[str, Any],
    ) -> dict[str, Any]:
        monitoring = bool(metrics.get("monitoring"))
        calibrated = bool(metrics.get("calibration_complete"))
        session_seconds = self._number(metrics.get("session_seconds"))
        session_minutes = session_seconds / 60.0
        risk = self._number(
            decision_engine.get(
                "risk_score",
                metrics.get("fatigue_probability"),
            )
        )
        profile_name = str(metrics.get("driver_profile_name") or "Guest")
        profile_id = metrics.get("driver_profile_id")

        self._record_live(
            monitoring=monitoring,
            risk=risk,
            session_seconds=session_seconds,
        )
        trajectory = self._trajectory()

        # A short, steep live change can be real, but extrapolating that raw
        # slope several minutes forward creates implausible 0%→100% jumps.
        # Preserve the measured slope for explanation/direction while using a
        # bounded slope only for the scenario projection.
        raw_slope = self._number(trajectory.get("slope_per_minute"))
        projection_slope = max(-0.08, min(0.08, raw_slope))
        trajectory["projection_slope_per_minute"] = round(
            projection_slope, 4
        )
        trajectory["projection_slope_limited"] = (
            abs(raw_slope - projection_slope) > 1e-9
        )

        period = self._period(datetime.now().hour)
        historical = self._history(profile_name, period)

        perception_state = str(
            perception.get("state") or "standby"
        ).lower()
        perception_score = self._number(perception.get("score"))

        passport_state = str(
            (passport_validation or {}).get("state") or "unavailable"
        ).lower()
        passport_drift = self._number(
            (passport_validation or {}).get("drift_score")
        )

        withheld_reasons: list[str] = []
        if not monitoring:
            withheld_reasons.append("Monitoring is not active.")
        elif not calibrated:
            withheld_reasons.append(
                "Personal calibration is not complete."
            )

        if perception_state == "insufficient":
            withheld_reasons.append(
                "Perception is INSUFFICIENT; current visual evidence is "
                "not reliable enough for a personalised forecast."
            )

        if passport_state in {
            "drift_detected",
            "recalibration_recommended",
        }:
            withheld_reasons.append(
                "The Personal AI Calibration Passport is not trusted enough "
                "for personalised forecasting."
            )

        if profile_id and historical["session_count"] < 2:
            withheld_reasons.append(
                "Fewer than two historical Decision Memory sessions are "
                "available for this profile."
            )
        if session_seconds < 20 and monitoring:
            withheld_reasons.append(
                "The current session is too short to establish a live trajectory."
            )

        confidence = 0.34
        confidence += min(0.20, historical["session_count"] / 20 * 0.20)
        confidence += min(
            0.16,
            trajectory["live_points"] / 10 * 0.16,
        )
        confidence += min(0.12, perception_score * 0.12)
        if passport_state == "valid":
            confidence += 0.14
        elif passport_state == "watch":
            confidence += 0.06
        elif passport_state in {
            "drift_detected",
            "recalibration_recommended",
        }:
            confidence -= 0.18
        confidence = max(0.05, min(0.94, confidence))

        if passport_state == "watch":
            confidence = min(confidence, 0.72)
        if perception_state == "degraded":
            confidence = min(confidence, 0.65)

        historical_window = historical["median_first_elevated_minutes"]
        time_to_elevated = None
        if (
            trajectory["direction"] == "rising"
            and projection_slope > 0
            and risk < self.ELEVATED_THRESHOLD
        ):
            estimate = (
                self.ELEVATED_THRESHOLD - risk
            ) / projection_slope
            if 0 < estimate <= 45:
                time_to_elevated = round(estimate, 1)

        if historical_window is not None and historical_window > session_minutes:
            historical_remaining = round(
                historical_window - session_minutes, 1
            )
            if time_to_elevated is None:
                time_to_elevated = historical_remaining
            else:
                time_to_elevated = round(
                    min(time_to_elevated, historical_remaining), 1
                )

        direction = trajectory["direction"]
        forecast_risk = risk
        horizon_minutes = 10
        if direction in {"rising", "falling"}:
            forecast_risk += projection_slope * min(
                horizon_minutes, 5
            )
        forecast_risk = max(0.0, min(1.0, forecast_risk))

        same_period = historical["same_period_peak_risk"]
        if historical["same_period_session_count"] >= 2:
            forecast_risk = (
                forecast_risk * 0.78 + same_period * 0.22
            )

        if withheld_reasons:
            status = "withheld"
            forecast_state = "uncertain"
            horizon = "Forecast withheld"
            summary = withheld_reasons[0]
            confidence = min(confidence, 0.35)
        else:
            status = "available"
            forecast_state = direction
            if risk >= 0.82:
                horizon = "Immediate safety attention"
            elif time_to_elevated is not None:
                horizon = (
                    f"Elevated-risk window in ~{time_to_elevated:g} min"
                )
            else:
                horizon = "Next 10–15 minutes"

            if direction == "rising":
                summary = (
                    "Recent advisory risk is rising. Guardian is combining "
                    "the live trajectory with the driver's local history."
                )
            elif direction == "falling":
                summary = (
                    "Recent advisory risk is falling relative to the preceding "
                    "live samples."
                )
            else:
                summary = (
                    "Recent advisory risk is broadly stable; no strong live "
                    "escalation trajectory is present."
                )

        factors: list[dict[str, Any]] = [
            {
                "label": "Current advisory risk",
                "value": round(risk, 4),
                "detail": f"Current V8 advisory risk is {risk:.1%}.",
            },
            {
                "label": "Live trajectory",
                "value": round(
                    min(
                        1.0,
                        abs(trajectory["slope_per_minute"]) / 0.30,
                    ),
                    4,
                ),
                "detail": (
                    f"Measured risk slope is "
                    f"{trajectory['slope_per_minute']:+.1%} per minute."
                    + (
                        f" Forecast projection is bounded to "
                        f"{projection_slope:+.1%} per minute to avoid "
                        "over-extrapolating short spikes."
                        if trajectory["projection_slope_limited"]
                        else ""
                    )
                ),
            },
        ]
        if historical["median_first_elevated_minutes"] is not None:
            factors.append(
                {
                    "label": "Historical escalation timing",
                    "value": min(
                        1.0,
                        session_minutes
                        / max(
                            1.0,
                            historical["median_first_elevated_minutes"],
                        ),
                    ),
                    "detail": (
                        "Historical sessions reached elevated advisory risk "
                        + (
                            f"at a median "
                            f"{historical['median_first_elevated_seconds']:.0f} seconds."
                            if (
                                historical.get("median_first_elevated_seconds")
                                is not None
                                and historical["median_first_elevated_seconds"] < 60
                            )
                            else (
                                f"at a median "
                                f"{historical['median_first_elevated_minutes']:.1f} minutes."
                            )
                        )
                    ),
                }
            )
        factors.append(
            {
                "label": "Perception trust",
                "value": round(perception_score, 4),
                "detail": (
                    f"Perception is {perception_state.upper()} "
                    f"at {perception_score:.1%}."
                ),
            }
        )
        if passport_validation:
            factors.append(
                {
                    "label": "Passport trust",
                    "value": round(1.0 - passport_drift, 4),
                    "detail": (
                        f"Passport validation is "
                        f"{str(passport_validation.get('label') or passport_state).upper()}."
                    ),
                }
            )

        projection = []
        for minute in (0, 5, 10, 15):
            projected = risk + projection_slope * min(minute, 5)
            if historical["same_period_session_count"] >= 2:
                weight = min(0.28, minute / 15 * 0.28)
                projected = (
                    projected * (1.0 - weight)
                    + same_period * weight
                )
            projection.append(
                {
                    "minute": minute,
                    "risk": round(
                        max(0.0, min(1.0, projected)),
                        4,
                    ),
                }
            )

        recommended_action = (
            "Use Guardian's existing live monitoring and alerts as the "
            "authoritative safety path."
        )
        if status == "withheld":
            recommended_action = (
                "Do not rely on a personalised forecast until the stated "
                "trust limitation is resolved. Live monitoring remains active."
            )
        elif risk >= 0.82 or (
            direction == "rising"
            and time_to_elevated is not None
            and time_to_elevated <= 10
        ):
            recommended_action = (
                "Plan a safe break now rather than waiting for further "
                "forecast escalation."
            )
        elif direction == "rising":
            recommended_action = (
                "Continue close monitoring and prepare for a safe break if "
                "the rising trajectory persists."
            )

        return {
            "version": self.VERSION,
            "status": status,
            "forecast_state": forecast_state,
            "direction": direction,
            "forecast_risk": round(forecast_risk, 4),
            "confidence": round(confidence, 4),
            "horizon": horizon,
            "prediction_horizon_minutes": horizon_minutes,
            "time_to_elevated_minutes": time_to_elevated,
            "summary": summary,
            "recommended_action": recommended_action,
            "withheld_reasons": withheld_reasons,
            "trajectory": trajectory,
            "historical_pattern": historical,
            "current_period": period,
            "factors": factors,
            "projection": projection,
            "method": (
                "Transparent advisory forecast using the recent live V8 risk "
                "trajectory, local Decision Memory timing patterns, Perception "
                "Confidence and Passport Validation. It is not a medical "
                "prediction and does not change Guardian alerts."
            ),
            "safety_boundary": (
                "Predictive Guardian is advisory only. The trained model, "
                "V8 decision engine and existing monitoring/alert path remain "
                "authoritative."
            ),
        }
