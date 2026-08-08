from __future__ import annotations

import csv
import io
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class DecisionMemoryService:
    """Persistent research trace for Guardian's advisory Intelligence layer.

    Samples are captured when the Intelligence endpoint is polled. This keeps
    the stable Monitoring/camera/alert implementation untouched.

    Important boundary:
    if the Intelligence page is not open, no advisory samples are collected.
    The service reports this as observed coverage rather than implying complete
    journey telemetry.
    """

    CSV_FIELDS = [
        "timestamp",
        "elapsed_seconds",
        "driver_profile",
        "state",
        "alert_count",
        "ear",
        "baseline_ear",
        "yawn_score",
        "head_tilt",
        "raw_model_probability",
        "existing_decision_probability",
        "existing_smoothed_probability",
        "advisory_risk",
        "risk_band",
        "decision_confidence",
        "confidence_level",
        "signal_quality",
        "image_quality",
        "weather",
        "road_condition",
        "external_light",
        "occlusion",
        "context_caution",
        "dominant_evidence",
        "recommended_action",
    ]

    def __init__(self, root: Path) -> None:
        self.root = root
        self.memory_dir = root / "guardian_data" / "decision_memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active_id: str | None = None
        self._active: dict[str, Any] | None = None
        self._last_monitoring = False

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            return float(value or default)
        except (TypeError, ValueError):
            return default

    def _new_session(
        self,
        metrics: dict[str, Any],
        intelligence: dict[str, Any],
    ) -> dict[str, Any]:
        now = self._now()
        session_id = f"decision_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return {
            "schema_version": "8.1-decision-memory-v1",
            "id": session_id,
            "started_at": now.isoformat(timespec="seconds"),
            "ended_at": None,
            "driver_profile": str(metrics.get("driver_profile_name") or "Guest"),
            "source": "Guardian Intelligence polling",
            "coverage_note": (
                "Samples are captured while the Intelligence endpoint is being polled. "
                "This is an advisory research trace, not complete camera telemetry."
            ),
            "sample_interval_seconds_nominal": 4,
            "samples": [],
            "summary": {},
        }

    def _sample(
        self,
        metrics: dict[str, Any],
        intelligence: dict[str, Any],
    ) -> dict[str, Any]:
        engine = intelligence.get("decision_engine", {}) or {}
        confidence = engine.get("confidence", {}) or {}
        caution = engine.get("context_caution", {}) or {}
        context = intelligence.get("context", {}) or {}
        quality = intelligence.get("signal_quality", {}) or {}
        environment = intelligence.get("environment", {}) or {}
        evidence = engine.get("evidence", []) or []
        strongest = max(
            evidence,
            key=lambda item: self._number(item.get("contribution")),
            default={},
        )

        legacy = engine.get("legacy_reference", {}) or {}
        return {
            "timestamp": self._now().isoformat(timespec="milliseconds"),
            "elapsed_seconds": int(self._number(metrics.get("session_seconds"))),
            "driver_profile": str(metrics.get("driver_profile_name") or "Guest"),
            "state": str(metrics.get("state") or "READY"),
            "alert_count": int(self._number(metrics.get("alert_count"))),
            "ear": round(self._number(metrics.get("ear")), 6),
            "baseline_ear": round(self._number(metrics.get("baseline_ear")), 6),
            "yawn_score": round(self._number(metrics.get("yawn_score")), 6),
            "head_tilt": round(self._number(metrics.get("head_tilt")), 6),
            "raw_model_probability": round(
                self._number(
                    legacy.get(
                        "raw_model_probability",
                        metrics.get("raw_probability"),
                    )
                ),
                6,
            ),
            "existing_decision_probability": round(
                self._number(
                    legacy.get(
                        "existing_personalized_probability",
                        metrics.get("decision_probability"),
                    )
                ),
                6,
            ),
            "existing_smoothed_probability": round(
                self._number(
                    legacy.get(
                        "existing_smoothed_probability",
                        metrics.get("fatigue_probability"),
                    )
                ),
                6,
            ),
            "advisory_risk": round(self._number(engine.get("risk_score")), 6),
            "risk_band": str(engine.get("risk_band") or "standby"),
            "decision_confidence": round(
                self._number(confidence.get("score")), 6
            ),
            "confidence_level": str(confidence.get("level") or "standby"),
            "signal_quality": round(self._number(quality.get("score")), 6),
            "image_quality": round(
                self._number(environment.get("quality_score")), 6
            ),
            "weather": str(context.get("weather") or "unknown"),
            "road_condition": str(context.get("road_condition") or "unknown"),
            "external_light": str(context.get("external_light") or "unknown"),
            "occlusion": str(context.get("occlusion") or "none"),
            "context_caution": str(caution.get("level") or "normal"),
            "dominant_evidence": str(strongest.get("label") or "none"),
            "recommended_action": str(engine.get("action") or ""),
            "evidence": evidence,
        }

    @staticmethod
    def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {
                "sample_count": 0,
                "observed_seconds": 0,
                "maximum_advisory_risk": 0.0,
                "average_advisory_risk": 0.0,
                "average_confidence": 0.0,
                "minimum_confidence": 0.0,
                "average_ear": 0.0,
                "alert_count": 0,
                "dominant_evidence": "none",
            }

        risks = [float(row.get("advisory_risk", 0) or 0) for row in samples]
        confidence = [
            float(row.get("decision_confidence", 0) or 0)
            for row in samples
        ]
        ears = [float(row.get("ear", 0) or 0) for row in samples if row.get("ear")]
        evidence_counts: dict[str, int] = {}
        for row in samples:
            label = str(row.get("dominant_evidence") or "none")
            evidence_counts[label] = evidence_counts.get(label, 0) + 1

        start = int(samples[0].get("elapsed_seconds", 0) or 0)
        end = int(samples[-1].get("elapsed_seconds", 0) or 0)
        return {
            "sample_count": len(samples),
            "observed_seconds": max(0, end - start),
            "maximum_advisory_risk": round(max(risks), 6),
            "average_advisory_risk": round(sum(risks) / len(risks), 6),
            "average_confidence": round(sum(confidence) / len(confidence), 6),
            "minimum_confidence": round(min(confidence), 6),
            "average_ear": round(sum(ears) / len(ears), 6) if ears else 0.0,
            "alert_count": max(
                int(row.get("alert_count", 0) or 0)
                for row in samples
            ),
            "dominant_evidence": max(evidence_counts, key=evidence_counts.get),
        }

    def observe(
        self,
        metrics: dict[str, Any],
        intelligence: dict[str, Any],
    ) -> dict[str, Any]:
        monitoring = bool(metrics.get("monitoring"))

        with self._lock:
            if monitoring and not self._last_monitoring:
                self._active = self._new_session(metrics, intelligence)
                self._active_id = self._active["id"]

            if monitoring:
                if self._active is None:
                    self._active = self._new_session(metrics, intelligence)
                    self._active_id = self._active["id"]
                sample = self._sample(metrics, intelligence)
                self._active["samples"].append(sample)
                self._active["summary"] = self._summary(self._active["samples"])
                self._save(self._active)

            if not monitoring and self._last_monitoring and self._active:
                self._active["ended_at"] = self._now().isoformat(timespec="seconds")
                self._active["summary"] = self._summary(self._active["samples"])
                self._save(self._active)
                self._active = None
                self._active_id = None

            self._last_monitoring = monitoring

            return {
                "recording": monitoring and self._active is not None,
                "active_session_id": self._active_id,
                "sample_count": (
                    len(self._active.get("samples", []))
                    if self._active
                    else 0
                ),
                "coverage_note": (
                    "Decision Memory records while Guardian Intelligence is open."
                ),
            }

    def _save(self, payload: dict[str, Any]) -> None:
        path = self.memory_dir / f"{payload['id']}.json"
        path.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    def _read(self, session_id: str) -> dict[str, Any]:
        path = self.resolve(session_id, ".json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid Decision Memory file.")
        return payload

    def list_sessions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(
            self.memory_dir.glob("decision_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            summary = payload.get("summary", {}) or {}
            rows.append({
                "id": payload.get("id", path.stem),
                "started_at": payload.get("started_at"),
                "ended_at": payload.get("ended_at"),
                "driver_profile": payload.get("driver_profile", "Guest"),
                "sample_count": int(summary.get("sample_count", 0) or 0),
                "observed_seconds": int(summary.get("observed_seconds", 0) or 0),
                "maximum_advisory_risk": float(
                    summary.get("maximum_advisory_risk", 0) or 0
                ),
                "average_advisory_risk": float(
                    summary.get("average_advisory_risk", 0) or 0
                ),
                "average_confidence": float(
                    summary.get("average_confidence", 0) or 0
                ),
                "dominant_evidence": summary.get("dominant_evidence", "none"),
                "active": payload.get("id") == self._active_id,
            })
        return rows

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._read(session_id)

    def comparison(
        self,
        first_id: str,
        second_id: str,
    ) -> dict[str, Any]:
        first = self._read(first_id)
        second = self._read(second_id)
        a = first.get("summary", {}) or {}
        b = second.get("summary", {}) or {}

        def delta(key: str) -> float:
            return round(
                float(b.get(key, 0) or 0) - float(a.get(key, 0) or 0),
                6,
            )

        return {
            "first": {
                "id": first["id"],
                "started_at": first.get("started_at"),
                "driver_profile": first.get("driver_profile"),
                "summary": a,
            },
            "second": {
                "id": second["id"],
                "started_at": second.get("started_at"),
                "driver_profile": second.get("driver_profile"),
                "summary": b,
            },
            "delta_second_minus_first": {
                "average_advisory_risk": delta("average_advisory_risk"),
                "maximum_advisory_risk": delta("maximum_advisory_risk"),
                "average_confidence": delta("average_confidence"),
                "minimum_confidence": delta("minimum_confidence"),
                "average_ear": delta("average_ear"),
                "alert_count": int(b.get("alert_count", 0) or 0)
                - int(a.get("alert_count", 0) or 0),
            },
        }

    def resolve(self, session_id: str, suffix: str) -> Path:
        if not session_id.startswith("decision_"):
            raise ValueError("Invalid Decision Memory ID.")
        candidate = (self.memory_dir / f"{session_id}{suffix}").resolve()
        if self.memory_dir.resolve() not in candidate.parents:
            raise ValueError("Invalid Decision Memory path.")
        if suffix == ".json":
            if not candidate.exists():
                raise FileNotFoundError(session_id)
            return candidate
        raise ValueError("Unsupported Decision Memory suffix.")

    def csv_bytes(self, session_id: str) -> bytes:
        payload = self._read(session_id)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.CSV_FIELDS)
        writer.writeheader()
        for sample in payload.get("samples", []):
            writer.writerow({
                key: sample.get(key, "")
                for key in self.CSV_FIELDS
            })
        return output.getvalue().encode("utf-8-sig")
