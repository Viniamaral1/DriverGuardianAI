from __future__ import annotations

from statistics import median
from typing import Any, Callable


class PersistentCalibrationAdapter:
    """Compatibility wrapper around the existing V3 PersonalCalibration.

    A known profile receives a short verification period. A mismatch switches
    to the original full calibration. The wrapped V3 risk-fusion method remains
    unchanged.
    """

    def __init__(
        self,
        calibration_factory: Callable[..., Any],
        *,
        profile: dict[str, Any] | None,
        profile_service: Any | None,
        event_callback: Callable[[str, str, str], None],
        camera_index: int,
        quick_seconds: float = 3.0,
        quick_minimum_samples: int = 20,
        match_tolerance: float = 0.12,
        full_seconds: float = 10.0,
        full_minimum_samples: int = 80,
    ) -> None:
        self._factory = calibration_factory
        self._profile = profile
        self._profile_service = profile_service
        self._event_callback = event_callback
        self._camera_index = int(camera_index)
        self._quick_seconds = float(quick_seconds)
        self._quick_minimum_samples = int(quick_minimum_samples)
        self._match_tolerance = float(match_tolerance)
        self._full_seconds = float(full_seconds)
        self._full_minimum_samples = int(full_minimum_samples)

        self._delegate = self._factory(
            required_seconds=self._full_seconds,
            minimum_samples=self._full_minimum_samples,
        )
        self._mode = "full"
        self._status = "NEW PROFILE CALIBRATION"
        self._fallback_reason: str | None = None
        self._saved = False

        self._quick_started_at: float | None = None
        self._quick_ear: list[float] = []
        self._quick_yawn: list[float] = []
        self._quick_tilt: list[float] = []

        calibration = (profile or {}).get("calibration") or {}
        saved_camera_index = calibration.get("camera_index")
        camera_matches = (
            saved_camera_index is None
            or int(saved_camera_index) == self._camera_index
        )
        if profile and calibration and camera_matches:
            self._mode = "quick"
            self._status = "VERIFYING SAVED PROFILE"
            self._saved_baseline = {
                "ear": float(calibration.get("baseline_ear", 0.25) or 0.25),
                "yawn": float(calibration.get("baseline_yawn", 0.0) or 0.0),
                "tilt": float(calibration.get("baseline_tilt", 0.0) or 0.0),
            }
        else:
            self._saved_baseline = None
            if profile and calibration and not camera_matches:
                self._status = "CAMERA CHANGED — FULL CALIBRATION"
                self._fallback_reason = "The saved baseline belongs to a different camera index"

    @property
    def complete(self) -> bool:
        return bool(self._delegate.complete)

    @property
    def baseline_ear(self) -> float:
        return float(self._delegate.baseline_ear)

    @property
    def baseline_yawn(self) -> float:
        return float(self._delegate.baseline_yawn)

    @property
    def baseline_tilt(self) -> float:
        return float(self._delegate.baseline_tilt)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def status(self) -> str:
        return self._status

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def profile_id(self) -> str | None:
        return (self._profile or {}).get("id")

    @property
    def profile_name(self) -> str:
        return str((self._profile or {}).get("name") or "Guest")

    def reset(self) -> None:
        self._delegate.reset()
        self._quick_started_at = None
        self._quick_ear.clear()
        self._quick_yawn.clear()
        self._quick_tilt.clear()
        self._saved = False

    def _switch_to_full(self, reason: str) -> None:
        self._mode = "full"
        self._status = "FULL RECALIBRATION"
        self._fallback_reason = reason
        self._delegate = self._factory(
            required_seconds=self._full_seconds,
            minimum_samples=self._full_minimum_samples,
        )
        self._event_callback(
            "PROFILE",
            f"{self.profile_name}: saved baseline did not match; full calibration started.",
            "warning",
        )

    def _accept_saved_profile(
        self,
        observed_ear: float,
        observed_yawn: float,
        observed_tilt: float,
    ) -> None:
        assert self._saved_baseline is not None
        # Mostly preserve the stored baseline while allowing a small adjustment
        # to the present camera position.
        self._delegate.baseline_ear = (
            self._saved_baseline["ear"] * 0.75 + observed_ear * 0.25
        )
        self._delegate.baseline_yawn = (
            self._saved_baseline["yawn"] * 0.75 + observed_yawn * 0.25
        )
        self._delegate.baseline_tilt = (
            self._saved_baseline["tilt"] * 0.75 + observed_tilt * 0.25
        )
        self._delegate.complete = True
        self._mode = "reused"
        self._status = "SAVED PROFILE VERIFIED"
        self._event_callback(
            "PROFILE",
            f"{self.profile_name}: saved baseline verified in {self._quick_seconds:.0f} seconds.",
            "success",
        )

    def update(self, features: dict[str, float], now: float) -> None:
        if self.complete:
            return

        if self._mode == "quick":
            if not features.get("face_detected"):
                return
            if self._quick_started_at is None:
                self._quick_started_at = now

            ear = float(features.get("ear", 0.0) or 0.0)
            if 0.08 <= ear <= 0.45:
                self._quick_ear.append(ear)
                self._quick_yawn.append(float(features.get("yawn_score", 0.0) or 0.0))
                self._quick_tilt.append(float(features.get("head_tilt", 0.0) or 0.0))

            elapsed = now - self._quick_started_at
            if (
                elapsed >= self._quick_seconds
                and len(self._quick_ear) >= self._quick_minimum_samples
            ):
                observed_ear = float(median(self._quick_ear))
                observed_yawn = float(median(self._quick_yawn))
                observed_tilt = float(median(self._quick_tilt))
                saved_ear = max(float(self._saved_baseline["ear"]), 1e-6)
                difference = abs(observed_ear - saved_ear) / saved_ear
                tilt_difference = abs(
                    observed_tilt - float(self._saved_baseline["tilt"])
                )
                matched = (
                    difference <= self._match_tolerance
                    and tilt_difference <= 8.0
                )

                if self._profile_service and self.profile_id:
                    self._profile_service.record_verification(
                        self.profile_id,
                        matched=matched,
                        observed_ear=observed_ear,
                        difference_percent=difference * 100.0,
                    )

                if matched:
                    self._accept_saved_profile(
                        observed_ear,
                        observed_yawn,
                        observed_tilt,
                    )
                else:
                    reason_parts = [
                        f"EAR differed by {difference * 100.0:.1f}%"
                    ]
                    if tilt_difference > 8.0:
                        reason_parts.append(
                            f"head angle differed by {tilt_difference:.1f} degrees"
                        )
                    self._switch_to_full("; ".join(reason_parts))
            return

        was_complete = self._delegate.complete
        self._delegate.update(features, now)

        if self._delegate.complete and not was_complete and not self._saved:
            self._saved = True
            self._mode = "full_complete"
            self._status = "FULL CALIBRATION SAVED"
            if self._profile_service and self.profile_id:
                self._profile_service.save_calibration(
                    self.profile_id,
                    baseline_ear=self.baseline_ear,
                    baseline_yawn=self.baseline_yawn,
                    baseline_tilt=self.baseline_tilt,
                    camera_index=self._camera_index,
                    sample_count=len(self._delegate.ear_samples),
                    source="full_calibration",
                )
                self._event_callback(
                    "PROFILE",
                    f"{self.profile_name}: personal baseline saved locally.",
                    "success",
                )

    def elapsed(self, now: float) -> float:
        if self._mode == "quick":
            if self._quick_started_at is None:
                return 0.0
            return now - self._quick_started_at
        return float(self._delegate.elapsed(now))

    def remaining(self, now: float) -> float:
        if self.complete:
            return 0.0
        if self._mode == "quick":
            return max(0.0, self._quick_seconds - self.elapsed(now))
        return float(self._delegate.remaining(now))

    def calculate_fused_probability(
        self,
        model_probability: float,
        features: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        return self._delegate.calculate_fused_probability(
            model_probability,
            features,
        )
