from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from app.services.automatic_context_service import AutomaticContextService


class EdgeMemoryService:
    """Persistent, offline-first memory derived from completed session reports.

    The service stores summaries rather than raw camera frames. It can rebuild
    itself from reports/v3 at any time, so the memory file is a convenience
    cache rather than the only copy of a session.
    """

    VERSION = 1

    def __init__(self, root: Path) -> None:
        self.root = root
        self.report_dir = root / "reports" / "v3"
        self.data_dir = root / "guardian_data"
        self.path = self.data_dir / "edge_memory.json"
        self._lock = threading.RLock()
        self.automatic_context_service = AutomaticContextService()
        self._data: dict[str, Any] = self._empty()
        self._load()
        self.refresh_from_reports()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _empty(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "updated_at": self._now(),
            "sessions": {},
            "sync_queue": [],
            "context": {
                "automatic_enabled": True,
                "location": "",
                "manual_override": False,
                "weather": "unknown",
                "external_light": "unknown",
                "cabin_light": "unknown",
                "occlusion": "none",
                "road_condition": "unknown",
                "notes": "",
                "updated_at": None,
            },
        }

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                base = self._empty()
                base.update(payload)
                base["sessions"] = payload.get("sessions", {}) or {}
                base["sync_queue"] = payload.get("sync_queue", []) or []
                base["context"].update(payload.get("context", {}) or {})
                self._data = base
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
    def _number(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(payload.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _integer(payload: dict[str, Any], key: str, default: int = 0) -> int:
        try:
            return int(payload.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def _session_summary(self, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        report_id = path.stem
        generated_at = payload.get("generated_at")
        if not generated_at:
            generated_at = datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()

        return {
            "id": report_id,
            "generated_at": generated_at,
            "duration_seconds": self._number(payload, "duration_seconds"),
            "average_risk": self._number(payload, "average_smoothed_probability"),
            "maximum_risk": self._number(payload, "maximum_smoothed_probability"),
            "alert_count": self._integer(payload, "alert_count"),
            "warning_episodes": self._integer(payload, "warning_episodes"),
            "critical_episodes": self._integer(payload, "critical_episodes"),
            "dominant_signal": str(payload.get("dominant_risk_signal", "unknown")),
            "baseline_ear": self._number(payload, "baseline_ear"),
            "average_ear": self._number(payload, "average_ear"),
            "minimum_ear": self._number(payload, "minimum_ear"),
            "maximum_yawn_score": self._number(payload, "maximum_yawn_score"),
            "maximum_head_tilt": self._number(payload, "maximum_head_tilt"),
            "state_percentages": payload.get("state_percentages", {}) or {},
            "report_file": path.name,
            "modified": path.stat().st_mtime,
        }

    def refresh_from_reports(self) -> dict[str, Any]:
        with self._lock:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            sessions = self._data.setdefault("sessions", {})
            queue = self._data.setdefault("sync_queue", [])
            queued_ids = {
                item.get("session_id")
                for item in queue
                if item.get("status") == "pending"
            }
            imported = 0

            for path in sorted(self.report_dir.glob("session_report_*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue

                summary = self._session_summary(path, payload)
                report_id = summary["id"]
                previous = sessions.get(report_id)
                sessions[report_id] = summary

                if previous is None:
                    imported += 1
                    if report_id not in queued_ids:
                        queue.append(
                            {
                                "session_id": report_id,
                                "status": "pending",
                                "created_at": self._now(),
                                "attempts": 0,
                            }
                        )
                        queued_ids.add(report_id)

            self._save()
            return {
                "imported": imported,
                "session_count": len(sessions),
                "pending_sync": self.pending_count(),
            }

    def set_context(self, updates: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "automatic_enabled",
            "location",
            "manual_override",
            "weather",
            "external_light",
            "cabin_light",
            "occlusion",
            "road_condition",
            "notes",
        }
        with self._lock:
            context = self._data.setdefault("context", {})
            for key, value in updates.items():
                if key in allowed and value is not None:
                    if key in {"automatic_enabled", "manual_override"}:
                        context[key] = bool(value)
                    else:
                        context[key] = str(value).strip()[:240]
            context["updated_at"] = self._now()
            self._save()
            return dict(context)


    def clear_manual_context(self) -> dict[str, Any]:
        """Clear optional manual context without touching automatic settings."""
        with self._lock:
            context = self._data.setdefault("context", {})
            context.update(
                {
                    "manual_override": False,
                    "weather": "unknown",
                    "road_condition": "unknown",
                    "external_light": "unknown",
                    "cabin_light": "unknown",
                    "occlusion": "none",
                    "notes": "",
                    "updated_at": self._now(),
                }
            )
            self._save()
            return dict(context)

    def resolved_context(self, force_weather: bool = False) -> dict[str, Any]:
        with self._lock:
            manual = dict(self._data.get("context", {}))

        automatic_enabled = bool(manual.get("automatic_enabled", True))
        manual_override = bool(manual.get("manual_override", False))
        location = str(manual.get("location", "") or "").strip()
        automatic = (
            self.automatic_context_service.snapshot(location, force=force_weather)
            if automatic_enabled
            else self.automatic_context_service.unknown(
                "Automatic weather is disabled.", location=location
            )
        )

        now = datetime.now()
        period = (
            "morning" if 5 <= now.hour < 12 else
            "afternoon" if 12 <= now.hour < 17 else
            "evening" if 17 <= now.hour < 21 else "night"
        )
        time_light = "daylight" if 7 <= now.hour < 18 else "dusk" if 5 <= now.hour < 21 else "night"

        def value(name: str, fallback: str = "unknown") -> dict[str, Any]:
            manual_value = str(manual.get(name, fallback) or fallback)
            auto_value = str(automatic.get(name, fallback) or fallback)
            if manual_override and manual_value not in {"", "unknown"}:
                return {"value": manual_value, "source": "Manual override", "confidence": 1.0, "updated_at": manual.get("updated_at"), "fresh": True}
            if automatic_enabled and automatic.get("available") and auto_value != "unknown":
                return {"value": auto_value, "source": automatic.get("source", "Automatic"), "confidence": automatic.get("confidence", 0.0), "updated_at": automatic.get("updated_at"), "fresh": automatic.get("fresh", False)}
            return {"value": fallback, "source": "Unknown", "confidence": 0.0, "updated_at": None, "fresh": False}

        weather = value("weather")
        road = value("road_condition")
        external = value("external_light")
        if external["value"] == "unknown":
            external = {"value": time_light, "source": "Local clock estimate", "confidence": 0.65, "updated_at": self._now(), "fresh": True}

        cabin = {"value": str(manual.get("cabin_light", "unknown") or "unknown"), "source": "Manual context", "confidence": 1.0 if manual.get("cabin_light") not in {None, "unknown"} else 0.0, "updated_at": manual.get("updated_at"), "fresh": bool(manual.get("updated_at"))}
        occlusion = {"value": str(manual.get("occlusion", "none") or "none"), "source": "Manual context", "confidence": 1.0, "updated_at": manual.get("updated_at"), "fresh": bool(manual.get("updated_at"))}

        return {
            "automatic_enabled": automatic_enabled,
            "manual_override": manual_override,
            "location": location,
            "automatic_weather": automatic,
            "local_period": {"value": period, "source": "Local clock", "confidence": 1.0, "updated_at": self._now(), "fresh": True},
            "weather": weather,
            "road_condition": road,
            "external_light": external,
            "cabin_light": cabin,
            "occlusion": occlusion,
            "notes": manual.get("notes", ""),
            "manual_context": manual,
        }
    def pending_count(self) -> int:
        return sum(
            1
            for item in self._data.get("sync_queue", [])
            if item.get("status") == "pending"
        )

    def mark_synced(self, session_ids: list[str] | None = None) -> int:
        with self._lock:
            selected = set(session_ids or [])
            count = 0
            for item in self._data.get("sync_queue", []):
                if item.get("status") != "pending":
                    continue
                if selected and item.get("session_id") not in selected:
                    continue
                item["status"] = "synced"
                item["synced_at"] = self._now()
                count += 1
            self._save()
            return count

    @staticmethod
    def _hour_from_timestamp(value: Any) -> int | None:
        if not value:
            return None
        try:
            cleaned = str(value).replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned).hour
        except ValueError:
            return None

    def insights(self) -> dict[str, Any]:
        sessions = list(self._data.get("sessions", {}).values())
        sessions.sort(key=lambda item: str(item.get("generated_at", "")), reverse=True)

        if not sessions:
            return {
                "session_count": 0,
                "total_duration_seconds": 0,
                "average_duration_seconds": 0,
                "average_risk": 0,
                "highest_risk": 0,
                "total_alerts": 0,
                "alert_session_rate": 0,
                "average_baseline_ear": 0,
                "highest_risk_period": "not enough data",
                "latest_session": None,
                "summary": "Complete a session to begin building local driving patterns.",
            }

        durations = [float(item.get("duration_seconds", 0) or 0) for item in sessions]
        average_risks = [float(item.get("average_risk", 0) or 0) for item in sessions]
        maximum_risks = [float(item.get("maximum_risk", 0) or 0) for item in sessions]
        baselines = [
            float(item.get("baseline_ear", 0) or 0)
            for item in sessions
            if float(item.get("baseline_ear", 0) or 0) > 0
        ]
        total_alerts = sum(int(item.get("alert_count", 0) or 0) for item in sessions)
        alert_sessions = sum(
            1 for item in sessions if int(item.get("alert_count", 0) or 0) > 0
        )

        periods = {
            "morning": [],
            "afternoon": [],
            "evening": [],
            "night": [],
        }
        for item in sessions:
            hour = self._hour_from_timestamp(item.get("generated_at"))
            if hour is None:
                continue
            risk = float(item.get("maximum_risk", 0) or 0)
            if 5 <= hour < 12:
                periods["morning"].append(risk)
            elif 12 <= hour < 17:
                periods["afternoon"].append(risk)
            elif 17 <= hour < 22:
                periods["evening"].append(risk)
            else:
                periods["night"].append(risk)

        period_scores = {
            period: mean(values)
            for period, values in periods.items()
            if values
        }
        highest_period = (
            max(period_scores, key=period_scores.get)
            if period_scores
            else "not enough data"
        )

        avg_risk = mean(average_risks)
        highest_risk = max(maximum_risks)
        if highest_risk >= 0.82 or total_alerts:
            summary = (
                "Local history contains at least one high-attention session. "
                "Commander can compare new sessions against these records."
            )
        elif highest_risk >= 0.65:
            summary = (
                "Local history contains warning-level fatigue evidence. "
                "Current-session signals should remain the primary safety input."
            )
        else:
            summary = (
                "Recorded sessions have generally remained below the warning threshold."
            )

        return {
            "session_count": len(sessions),
            "total_duration_seconds": sum(durations),
            "average_duration_seconds": mean(durations),
            "average_risk": avg_risk,
            "highest_risk": highest_risk,
            "total_alerts": total_alerts,
            "alert_session_rate": alert_sessions / len(sessions),
            "average_baseline_ear": mean(baselines) if baselines else 0,
            "highest_risk_period": highest_period,
            "latest_session": sessions[0],
            "summary": summary,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            insights = self.insights()
            sessions = list(self._data.get("sessions", {}).values())
            sessions.sort(
                key=lambda item: str(item.get("generated_at", "")),
                reverse=True,
            )
            return {
                "available": True,
                "offline_ready": True,
                "updated_at": self._data.get("updated_at"),
                "context": self.resolved_context(),
                "manual_context": dict(self._data.get("context", {})),
                "insights": insights,
                "recent_sessions": sessions[:12],
                "pending_sync": self.pending_count(),
                "sync_queue": list(self._data.get("sync_queue", []))[-20:],
                "storage_path": str(self.path),
            }

    def export_bundle(self) -> dict[str, Any]:
        with self._lock:
            return {
                "exported_at": self._now(),
                "guardian_edge_version": self.VERSION,
                "privacy": {
                    "raw_video_included": False,
                    "contains": [
                        "session summaries",
                        "risk metrics",
                        "behavioural signal summaries",
                        "journey context",
                        "sync queue metadata",
                    ],
                },
                **self.snapshot(),
            }
