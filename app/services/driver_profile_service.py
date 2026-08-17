from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DriverProfileService:
    """Local driver profiles and persistent calibration baselines.

    Profiles are selected manually. No face embeddings or biometric identity
    templates are stored.
    """

    VERSION = 1

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = root / "guardian_data"
        self.path = self.data_dir / "driver_profiles.json"
        self._lock = threading.RLock()
        self._data: dict[str, Any] = self._empty()
        self._load()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _empty(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "updated_at": self._now(),
            "active_profile_id": None,
            "profiles": {},
        }

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                data = self._empty()
                data.update(payload)
                data["profiles"] = payload.get("profiles", {}) or {}
                self._data = data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._data = self._empty()

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = self._now()
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _clean_name(name: str) -> str:
        clean = re.sub(r"\s+", " ", str(name or "").strip())
        if not clean:
            raise ValueError("Profile name cannot be empty.")
        return clean[:40]

    @staticmethod
    def _normalise_name(name: str) -> str:
        return " ".join(str(name or "").casefold().split())

    def _profile_with_name(self, name: str) -> dict[str, Any] | None:
        wanted = self._normalise_name(name)
        for profile in self._data.get("profiles", {}).values():
            if self._normalise_name(profile.get("name", "")) == wanted:
                return profile
        return None

    def ensure_default_profile(self, preferred_name: str) -> dict[str, Any] | None:
        name = str(preferred_name or "").strip()
        if not name:
            return None

        with self._lock:
            profiles = self._data.setdefault("profiles", {})
            if profiles:
                return self.active_profile()

            profile = self.create(name)
            self.set_active(profile["id"])
            return self.active_profile()

    def create(self, name: str) -> dict[str, Any]:
        clean = self._clean_name(name)
        with self._lock:
            existing = self._profile_with_name(clean)
            if existing is not None:
                raise FileExistsError(
                    f'A driver profile named “{existing.get("name", clean)}” already exists.'
                )

            profile_id = uuid.uuid4().hex[:12]
            profile = {
                "id": profile_id,
                "name": clean,
                "created_at": self._now(),
                "updated_at": self._now(),
                "calibration": None,
                "verification_count": 0,
                "full_calibration_count": 0,
                "fallback_count": 0,
                "last_used_at": None,
            }
            self._data.setdefault("profiles", {})[profile_id] = profile
            if not self._data.get("active_profile_id"):
                self._data["active_profile_id"] = profile_id
            self._save()
            return dict(profile)

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            active_id = self._data.get("active_profile_id")
            profiles = []
            for profile in self._data.get("profiles", {}).values():
                item = dict(profile)
                item["active"] = item.get("id") == active_id
                item["has_calibration"] = bool(item.get("calibration"))
                profiles.append(item)
            profiles.sort(key=lambda item: str(item.get("name", "")).lower())
            return profiles

    def get(self, profile_id: str | None) -> dict[str, Any] | None:
        if not profile_id:
            return None
        with self._lock:
            profile = self._data.get("profiles", {}).get(profile_id)
            return dict(profile) if profile else None

    def active_profile(self) -> dict[str, Any] | None:
        return self.get(self._data.get("active_profile_id"))

    def set_active(self, profile_id: str | None) -> dict[str, Any] | None:
        with self._lock:
            if profile_id in {None, "", "guest"}:
                self._data["active_profile_id"] = None
                self._save()
                return None
            if profile_id not in self._data.get("profiles", {}):
                raise KeyError("Driver profile was not found.")
            self._data["active_profile_id"] = profile_id
            profile = self._data["profiles"][profile_id]
            profile["last_used_at"] = self._now()
            profile["updated_at"] = self._now()
            self._save()
            return dict(profile)

    def delete(self, profile_id: str) -> bool:
        with self._lock:
            profiles = self._data.get("profiles", {})
            if profile_id not in profiles:
                return False
            profiles.pop(profile_id)
            if self._data.get("active_profile_id") == profile_id:
                self._data["active_profile_id"] = None
            self._save()
            return True

    def reset_calibration(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = self._data.get("profiles", {}).get(profile_id)
            if not profile:
                raise KeyError("Driver profile was not found.")
            profile["calibration"] = None
            # A verification result belongs to the calibration it evaluated.
            # Once that calibration is explicitly reset, the result must not be
            # carried forward as evidence against the next baseline.
            profile["last_verification"] = None
            profile["updated_at"] = self._now()
            self._save()
            return dict(profile)

    def save_calibration(
        self,
        profile_id: str,
        *,
        baseline_ear: float,
        baseline_yawn: float,
        baseline_tilt: float,
        camera_index: int,
        sample_count: int,
        source: str,
    ) -> dict[str, Any]:
        with self._lock:
            profile = self._data.get("profiles", {}).get(profile_id)
            if not profile:
                raise KeyError("Driver profile was not found.")

            previous = profile.get("calibration") or {}
            previous_count = max(0, int(previous.get("observation_count", 0) or 0))
            new_count = max(1, int(sample_count or 1))
            total_count = previous_count + new_count

            def weighted(key: str, current: float) -> float:
                if previous_count <= 0:
                    return float(current)
                previous_value = float(previous.get(key, current) or current)
                return (
                    previous_value * previous_count + float(current) * new_count
                ) / total_count

            profile["calibration"] = {
                "baseline_ear": round(weighted("baseline_ear", baseline_ear), 6),
                "baseline_yawn": round(weighted("baseline_yawn", baseline_yawn), 6),
                "baseline_tilt": round(weighted("baseline_tilt", baseline_tilt), 4),
                "camera_index": int(camera_index),
                "observation_count": total_count,
                "last_source": str(source),
                "verification_count_at_calibration": int(
                    profile.get("verification_count", 0) or 0
                ),
                "fallback_count_at_calibration": int(
                    profile.get("fallback_count", 0) or 0
                ),
                "updated_at": self._now(),
            }
            # Any quick-verification mismatch that triggered this full
            # calibration has now been resolved by the new baseline.
            profile["last_verification"] = None
            profile["full_calibration_count"] = (
                int(profile.get("full_calibration_count", 0) or 0) + 1
            )
            profile["last_used_at"] = self._now()
            profile["updated_at"] = self._now()
            self._save()
            return dict(profile)

    def import_calibration(
        self,
        profile_id: str,
        *,
        baseline_ear: float,
        baseline_yawn: float,
        baseline_tilt: float,
        camera_index: int,
        observation_count: int,
        source_passport_id: str,
    ) -> dict[str, Any]:
        """Explicitly replace the saved baseline from an imported passport.

        This does not count as a completed local full calibration.
        """
        with self._lock:
            profile = self._data.get("profiles", {}).get(profile_id)
            if not profile:
                raise KeyError("Driver profile was not found.")
            profile["calibration"] = {
                "baseline_ear": round(float(baseline_ear), 6),
                "baseline_yawn": round(float(baseline_yawn), 6),
                "baseline_tilt": round(float(baseline_tilt), 4),
                "camera_index": int(camera_index),
                "observation_count": max(1, int(observation_count or 1)),
                "last_source": "passport_import",
                "source_passport_id": str(source_passport_id),
                "verification_count_at_calibration": int(
                    profile.get("verification_count", 0) or 0
                ),
                "fallback_count_at_calibration": int(
                    profile.get("fallback_count", 0) or 0
                ),
                "updated_at": self._now(),
            }
            profile["last_verification"] = None
            profile["last_used_at"] = self._now()
            profile["updated_at"] = self._now()
            self._save()
            return dict(profile)

    def record_verification(
        self,
        profile_id: str,
        *,
        matched: bool,
        observed_ear: float,
        difference_percent: float,
    ) -> None:
        with self._lock:
            profile = self._data.get("profiles", {}).get(profile_id)
            if not profile:
                return
            if matched:
                profile["verification_count"] = (
                    int(profile.get("verification_count", 0) or 0) + 1
                )
            else:
                profile["fallback_count"] = (
                    int(profile.get("fallback_count", 0) or 0) + 1
                )
            profile["last_verification"] = {
                "matched": bool(matched),
                "observed_ear": round(float(observed_ear), 6),
                "difference_percent": round(float(difference_percent), 2),
                "at": self._now(),
            }
            profile["last_used_at"] = self._now()
            profile["updated_at"] = self._now()
            self._save()

    def snapshot(self) -> dict[str, Any]:
        active = self.active_profile()
        return {
            "available": True,
            "active_profile": active,
            "active_profile_id": self._data.get("active_profile_id"),
            "guest_mode": active is None,
            "profiles": self.list_profiles(),
            "storage_path": str(self.path),
            "privacy": {
                "manual_selection": True,
                "face_recognition_used": False,
                "biometric_identity_template_stored": False,
                "raw_video_stored": False,
            },
        }
