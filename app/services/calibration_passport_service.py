from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import statistics
import uuid


class CalibrationPassportService:
    """Portable, local-only summary of a driver's calibration context.

    The passport contains behavioural baselines and derived reliability summaries.
    It intentionally excludes raw video, face embeddings and identity templates.
    """

    SCHEMA = "guardian-calibration-passport-v1"
    VERSION = "8.7-passport-v1"

    def __init__(self, root: Path, profile_service, settings_provider) -> None:
        self.root = root
        self.profile_service = profile_service
        self.settings_provider = settings_provider
        self.path = root / "guardian_data" / "calibration_passports.json"
        self._metadata = self._load_metadata()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_metadata(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return {"version": self.VERSION, "profiles": {}}

    def _save_metadata(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata["version"] = self.VERSION
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _profile_meta(self, profile_id: str) -> dict[str, Any]:
        profiles = self._metadata.setdefault("profiles", {})
        meta = profiles.setdefault(
            profile_id,
            {
                "passport_id": f"gcp_{uuid.uuid4().hex[:16]}",
                "created_at": self._now(),
                "updated_at": self._now(),
                "origin": "local",
                "privacy": {
                    "allow_export": True,
                    "include_perception_history": True,
                    "local_only": True,
                },
            },
        )
        return meta

    def _decision_sessions_for(self, profile_name: str) -> list[dict[str, Any]]:
        directory = self.root / "guardian_data" / "decision_memory"
        sessions: list[dict[str, Any]] = []
        if not directory.exists():
            return sessions
        wanted = str(profile_name or "").strip().casefold()
        for path in directory.glob("decision_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = str(payload.get("driver_profile") or "").strip().casefold()
            if wanted and name != wanted:
                continue
            sessions.append(payload)
        sessions.sort(key=lambda item: str(item.get("started_at") or ""))
        return sessions

    def _history_summary(self, profile_name: str) -> dict[str, Any]:
        sessions = self._decision_sessions_for(profile_name)
        perception_states: Counter[str] = Counter()
        reason_codes: Counter[str] = Counter()
        occlusions: Counter[str] = Counter()
        eye_visibility: list[float] = []
        perception_scores: list[float] = []
        baseline_values: list[float] = []
        head_tilt_values: list[float] = []

        for session in sessions:
            for sample in session.get("samples", []) or []:
                state = str(sample.get("perception_state") or "").lower()
                if state in {"trusted", "degraded", "insufficient"}:
                    perception_states[state] += 1
                score = self._number(sample.get("perception_confidence"))
                if score > 0:
                    perception_scores.append(score)
                eye = self._number(sample.get("automatic_eye_visibility"))
                if eye <= 0:
                    eye = self._number(sample.get("eye_visibility"))
                if eye > 0:
                    eye_visibility.append(eye)
                baseline = self._number(sample.get("baseline_ear"))
                if baseline > 0:
                    baseline_values.append(baseline)
                tilt = abs(self._number(sample.get("head_tilt")))
                if tilt > 0:
                    head_tilt_values.append(tilt)

                raw_reasons = sample.get("perception_reason_codes") or []
                if isinstance(raw_reasons, str):
                    raw_reasons = [raw_reasons]
                for reason in raw_reasons:
                    if isinstance(reason, dict):
                        code = str(reason.get("code") or "").strip()
                    else:
                        code = str(reason or "").strip()
                    if code:
                        reason_codes[code] += 1

                occ = str(
                    sample.get("automatic_occlusion")
                    or sample.get("raw_automatic_occlusion")
                    or ""
                ).strip().lower()
                if occ and occ not in {"none", "standby"}:
                    occlusions[occ] += 1

        total_perception = sum(perception_states.values())
        def state_rate(key: str) -> float:
            return (
                perception_states[key] / total_perception
                if total_perception
                else 0.0
            )

        return {
            "session_count": len(sessions),
            "sample_count": sum(
                len(item.get("samples", []) or []) for item in sessions
            ),
            "perception": {
                "average_confidence": round(
                    statistics.fmean(perception_scores), 4
                ) if perception_scores else 0.0,
                "trusted_rate": round(state_rate("trusted"), 4),
                "degraded_rate": round(state_rate("degraded"), 4),
                "insufficient_rate": round(state_rate("insufficient"), 4),
                "average_eye_visibility": round(
                    statistics.fmean(eye_visibility), 4
                ) if eye_visibility else 0.0,
                "common_limitations": [
                    {"code": key, "count": count}
                    for key, count in reason_codes.most_common(5)
                ],
            },
            "visibility": {
                "known_conditions": [
                    {"condition": key, "count": count}
                    for key, count in occlusions.most_common(5)
                ],
                "sunglasses_seen": any(
                    "sunglass" in key for key in occlusions
                )
                or any(
                    "EYE_REGION_OCCLUDED" in key.upper()
                    for key in reason_codes
                ),
            },
            "baseline_history": {
                "historical_baseline_ear": round(
                    statistics.median(baseline_values), 6
                ) if baseline_values else 0.0,
                "typical_abs_head_tilt": round(
                    statistics.median(head_tilt_values), 4
                ) if head_tilt_values else 0.0,
            },
        }

    def build(self, profile_id: str) -> dict[str, Any]:
        profile = self.profile_service.get(profile_id)
        if not profile:
            raise KeyError("Driver profile was not found.")

        meta = self._profile_meta(profile_id)
        calibration = profile.get("calibration") or {}
        history = self._history_summary(str(profile.get("name") or ""))
        settings = self.settings_provider() or {}
        privacy = dict(meta.get("privacy") or {})
        privacy.update(
            {
                "local_only": True,
                "cloud_sync": False,
                "raw_video_required": False,
                "face_identity_template": False,
                "visual_evidence_enabled": bool(
                    settings.get("visual_evidence_enabled", False)
                ),
                "evidence_deletion_independent": True,
            }
        )

        perception = history["perception"]
        visibility = history["visibility"]
        if not bool(privacy.get("include_perception_history", True)):
            perception = {
                "included_in_export": False,
                "note": "Perception-history export disabled by the user.",
            }
            visibility = {
                "included_in_export": False,
                "note": "Visibility-history export disabled by the user.",
            }

        return {
            "schema": self.SCHEMA,
            "passport_version": self.VERSION,
            "passport_id": meta["passport_id"],
            "created_at": meta.get("created_at"),
            "updated_at": self._now(),
            "origin": meta.get("origin", "local"),
            "identity": {
                "profile_id": profile["id"],
                "profile_name": profile["name"],
                "anonymous_identifier": meta["passport_id"],
                "manual_profile_selection": True,
                "biometric_identity_matching": False,
            },
            "personal_visual_baseline": {
                "baseline_ear": self._number(calibration.get("baseline_ear")),
                "baseline_yawn": self._number(calibration.get("baseline_yawn")),
                "baseline_head_tilt": self._number(
                    calibration.get("baseline_tilt")
                ),
                "camera_index": int(calibration.get("camera_index", 0) or 0),
                "observation_count": int(
                    calibration.get("observation_count", 0) or 0
                ),
                "last_calibration_source": calibration.get("last_source"),
                "last_calibrated_at": calibration.get("updated_at"),
                "historical_baseline_ear": history[
                    "baseline_history"
                ]["historical_baseline_ear"],
                "typical_abs_head_tilt": history[
                    "baseline_history"
                ]["typical_abs_head_tilt"],
            },
            "visibility_profile": visibility,
            "perception_reliability_profile": perception,
            "calibration_history": {
                "full_calibration_count": int(
                    profile.get("full_calibration_count", 0) or 0
                ),
                "verification_count": int(
                    profile.get("verification_count", 0) or 0
                ),
                "fallback_count": int(profile.get("fallback_count", 0) or 0),
                "last_verification": profile.get("last_verification"),
                "decision_memory_sessions": history["session_count"],
                "decision_memory_samples": history["sample_count"],
            },
            "privacy_and_retention": privacy,
            "portability": {
                "json_export": bool(privacy.get("allow_export", True)),
                "json_import": True,
                "raw_biometric_media_required": False,
                "cross_device_validation_required": True,
            },
            "safety_boundary": (
                "The Calibration Passport describes personal baseline and "
                "observability context. Importing it does not alter model "
                "weights, alert thresholds or biometric identity."
            ),
        }

    def update_privacy(
        self,
        profile_id: str,
        *,
        allow_export: bool,
        include_perception_history: bool,
    ) -> dict[str, Any]:
        if not self.profile_service.get(profile_id):
            raise KeyError("Driver profile was not found.")
        meta = self._profile_meta(profile_id)
        meta["privacy"] = {
            "allow_export": bool(allow_export),
            "include_perception_history": bool(include_perception_history),
            "local_only": True,
        }
        meta["updated_at"] = self._now()
        self._save_metadata()
        return self.build(profile_id)

    def reset(self, profile_id: str) -> dict[str, Any]:
        if not self.profile_service.get(profile_id):
            raise KeyError("Driver profile was not found.")
        self._metadata.setdefault("profiles", {}).pop(profile_id, None)
        self._save_metadata()
        return self.build(profile_id)

    def export_payload(self, profile_id: str) -> dict[str, Any]:
        passport = self.build(profile_id)
        if not bool(
            passport.get("privacy_and_retention", {}).get(
                "allow_export", True
            )
        ):
            raise PermissionError("Passport export is disabled for this profile.")
        passport["exported_at"] = self._now()
        return passport

    def validate_import(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Passport JSON must be an object.")
        if payload.get("schema") != self.SCHEMA:
            raise ValueError(
                "Unsupported passport schema. Expected "
                f"{self.SCHEMA}."
            )
        baseline = payload.get("personal_visual_baseline")
        if not isinstance(baseline, dict):
            raise ValueError("Passport does not contain a visual baseline.")
        ear = self._number(baseline.get("baseline_ear"))
        if not 0.05 <= ear <= 0.6:
            raise ValueError("Imported baseline EAR is outside a safe range.")
        return payload

    def import_into(
        self,
        profile_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        validated = self.validate_import(payload)
        if not self.profile_service.get(profile_id):
            raise KeyError("Driver profile was not found.")

        baseline = validated["personal_visual_baseline"]
        self.profile_service.import_calibration(
            profile_id,
            baseline_ear=self._number(baseline.get("baseline_ear")),
            baseline_yawn=self._number(baseline.get("baseline_yawn")),
            baseline_tilt=self._number(
                baseline.get("baseline_head_tilt")
            ),
            camera_index=int(baseline.get("camera_index", 0) or 0),
            observation_count=max(
                1, int(baseline.get("observation_count", 1) or 1)
            ),
            source_passport_id=str(
                validated.get("passport_id") or "external"
            ),
        )

        meta = self._profile_meta(profile_id)
        privacy = validated.get("privacy_and_retention") or {}
        meta["origin"] = "imported"
        meta["imported_from_passport_id"] = validated.get("passport_id")
        meta["imported_at"] = self._now()
        meta["privacy"] = {
            "allow_export": bool(privacy.get("allow_export", True)),
            "include_perception_history": bool(
                privacy.get("include_perception_history", True)
            ),
            "local_only": True,
        }
        meta["updated_at"] = self._now()
        self._save_metadata()
        return self.build(profile_id)
