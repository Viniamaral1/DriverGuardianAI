from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class ResearchLabService:
    """Dataset research and explainability service.

    This service is isolated from live monitoring. It never changes the camera,
    fatigue model, calibration, alert thresholds or driver profiles.
    """

    NUMERIC_FEATURES = [
        "ear",
        "yawn_score",
        "head_tilt",
        "hands_detected",
        "low_light",
        "face_confidence",
        "blink_count",
    ]
    REQUIRED_COLUMNS = {
        "participant_id", "session_id", "state", "condition",
        "ear", "yawn_score", "head_tilt",
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_dir = root / "guardian_data" / "research_lab"
        self.reference_path = (
            Path(__file__).resolve().parents[2]
            / "guardian_data"
            / "research_audit_reference.json"
        )
        self._lock = threading.RLock()
        self._last: dict[str, Any] | None = None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def candidate_paths(self) -> list[str]:
        candidates = [
            self.root / "data" / "processed" / "driver_guardian_v2_clean.csv",
            self.root.parent / "DriverGuardianAI" / "data" / "processed" / "driver_guardian_v2_clean.csv",
            Path.home() / "DriverGuardianAI" / "data" / "processed" / "driver_guardian_v2_clean.csv",
        ]
        return [str(path) for path in candidates]

    def reference(self) -> dict[str, Any]:
        try:
            return json.loads(self.reference_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"available": False}

    @staticmethod
    def _safe_number(value: Any) -> float | None:
        try:
            number = float(value)
            return number if math.isfinite(number) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _counts(series: Any, limit: int = 25) -> list[dict[str, Any]]:
        counts = series.fillna("Missing").astype(str).value_counts().head(limit)
        total = max(1, int(counts.sum()))
        return [
            {
                "label": str(label),
                "count": int(count),
                "share": round(int(count) / total, 6),
            }
            for label, count in counts.items()
        ]

    def _feature_association(self, frame: Any, target: Any) -> list[dict[str, Any]]:
        import pandas as pd
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder

        available = [name for name in self.NUMERIC_FEATURES if name in frame.columns]
        if not available:
            return []

        X = frame[available].copy()
        for name in available:
            if X[name].dtype == bool:
                X[name] = X[name].astype(int)
            else:
                X[name] = pd.to_numeric(X[name], errors="coerce")
            median = X[name].median()
            X[name] = X[name].fillna(0.0 if pd.isna(median) else median)

        y = LabelEncoder().fit_transform(target.fillna("Missing").astype(str))
        if len(set(y)) < 2:
            return []

        scores = mutual_info_classif(X, y, random_state=42)
        result = [
            {"feature": name, "score": round(float(score), 6)}
            for name, score in zip(available, scores)
        ]
        result.sort(key=lambda item: item["score"], reverse=True)
        return result

    def _condition_profile(self, frame: Any) -> list[dict[str, Any]]:
        if "condition" not in frame.columns:
            return []

        rows: list[dict[str, Any]] = []
        for condition, group in frame.groupby("condition", dropna=False):
            row: dict[str, Any] = {
                "condition": str(condition),
                "rows": int(len(group)),
                "participants": int(group["participant_id"].nunique())
                if "participant_id" in group else 0,
                "sessions": int(group["session_id"].nunique())
                if "session_id" in group else 0,
            }
            for name in ["ear", "yawn_score", "head_tilt", "face_confidence"]:
                if name in group:
                    values = group[name]
                    row[f"mean_{name}"] = round(float(values.mean()), 5)
                    row[f"median_{name}"] = round(float(values.median()), 5)
            if "state" in group:
                state_counts = group["state"].astype(str).value_counts(normalize=True)
                row["drowsy_share"] = round(float(state_counts.get("drowsy", 0.0)), 6)
            rows.append(row)

        rows.sort(key=lambda item: item["rows"], reverse=True)
        return rows

    def analyse(self, dataset_path: str) -> dict[str, Any]:
        import pandas as pd

        path = Path(dataset_path).expanduser()
        if not path.is_absolute():
            path = (self.root / path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Dataset was not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError("Research Lab currently accepts CSV datasets only.")

        frame = pd.read_csv(path, low_memory=False)
        missing_required = sorted(self.REQUIRED_COLUMNS - set(frame.columns))
        if missing_required:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(missing_required)
            )

        timestamp_summary: dict[str, Any] = {}
        if "timestamp" in frame:
            parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
            valid = parsed.dropna()
            timestamp_summary = {
                "valid_rows": int(valid.size),
                "start": valid.min().isoformat() if not valid.empty else None,
                "end": valid.max().isoformat() if not valid.empty else None,
            }

        state_association = self._feature_association(frame, frame["state"])
        fatigue_association = (
            self._feature_association(frame, frame["fatigue_level"])
            if "fatigue_level" in frame else []
        )

        result = {
            "available": True,
            "generated_at": self._now(),
            "dataset": {
                "path": str(path),
                "name": path.name,
                "bytes": path.stat().st_size,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "column_names": [str(column) for column in frame.columns],
                "participants": int(frame["participant_id"].nunique()),
                "sessions": int(frame["session_id"].nunique()),
                "duplicate_rows": int(frame.duplicated().sum()),
                "missing_cells": int(frame.isna().sum().sum()),
                "timestamp": timestamp_summary,
            },
            "distributions": {
                "participants": self._counts(frame["participant_id"]),
                "sessions": self._counts(frame["session_id"], limit=15),
                "states": self._counts(frame["state"]),
                "conditions": self._counts(frame["condition"]),
                "fatigue_levels": (
                    self._counts(frame["fatigue_level"])
                    if "fatigue_level" in frame else []
                ),
            },
            "condition_profile": self._condition_profile(frame),
            "feature_association": {
                "state": state_association,
                "fatigue_level": fatigue_association,
                "interpretation": (
                    "Mutual information describes association in this dataset. "
                    "It is not the trained model's feature importance and must "
                    "not be interpreted as causal influence."
                ),
            },
            "readiness": {
                "participant_aware_evaluation": (
                    "ready" if frame["participant_id"].nunique() >= 5 else "limited"
                ),
                "condition_robustness": (
                    "ready" if frame["condition"].nunique() >= 2 else "limited"
                ),
                "sequence_research": (
                    "ready"
                    if "timestamp" in frame and frame["session_id"].nunique() >= 10
                    else "limited"
                ),
                "image_occlusion_model": "needs_images_or_video_frames",
                "forecast_model": (
                    "candidate_for_research"
                    if "timestamp" in frame and frame["session_id"].nunique() >= 20
                    else "limited"
                ),
            },
            "boundaries": [
                "Condition labels support robustness analysis.",
                "Tabular condition labels alone do not train a camera image classifier.",
                "Feature association is not trained-model explainability.",
                "Participant and session groups must remain separated during evaluation.",
            ],
        }

        with self._lock:
            self._last = result
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output = self.output_dir / "latest_research_audit.json"
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["export_path"] = str(output)

        return result

    def deployment_mode(self) -> dict[str, Any]:
        mode = str(os.getenv("GUARDIAN_DEPLOYMENT_MODE", "research")).strip().lower()
        enabled_value = str(os.getenv("GUARDIAN_RESEARCH_ENABLED", "true")).strip().lower()
        enabled = enabled_value not in {"0", "false", "no", "off"}
        return {
            "mode": mode if mode in {"research", "production"} else "research",
            "research_enabled": enabled,
            "recommended_public_setting": {
                "GUARDIAN_DEPLOYMENT_MODE": "production",
                "GUARDIAN_RESEARCH_ENABLED": "false",
            },
            "public_boundary": (
                "Research Lab should be disabled on a public driver-facing deployment. "
                "Expose aggregate model-card information only; never publish raw rows, "
                "participant identifiers, session identifiers or local dataset paths."
            ),
        }

    def model_lifecycle(self) -> dict[str, Any]:
        model_path = (
            self.root / "models" / "v2" / "ablation"
            / "driver_guardian_core_behaviour.joblib"
        )
        split_root = self.root / "data" / "splits" / "v2"
        split_status = {
            name: (split_root / f"{name}.csv").exists()
            for name in ("train", "calibration", "test")
        }
        return {
            "model_card": {
                "name": "Driver Guardian Core Behaviour",
                "family": "Histogram Gradient Boosting classifier",
                "task": "Binary fatigue-risk classification",
                "features": ["ear", "yawn_score", "head_tilt"],
                "target_mapping": {
                    "Alert": 0,
                    "Mild Fatigue": 1,
                    "Moderate Fatigue": 1,
                },
                "expected_path": str(model_path),
                "artifact_available": model_path.exists(),
                "packaging_note": (
                    "Model artifacts are intentionally excluded from clean source ZIPs."
                    if not model_path.exists() else
                    "The expected trained model artifact is available locally."
                ),
            },
            "training_protocol": {
                "participant_and_session_aware": True,
                "train_only_preprocessing": True,
                "threshold_selected_on_calibration_split": True,
                "untouched_test_split": True,
                "deployment_transfer_evaluation": True,
                "split_files_available": split_status,
                "source_script": "src/v2/train_ablation_models_v2.py",
            },
            "live_decision_layer": {
                "personal_calibration": True,
                "temporal_smoothing": True,
                "controlled_alert_state_machine": True,
                "explanation": (
                    "The trained model produces risk evidence. Personal baseline "
                    "calibration and deterministic temporal logic then create the "
                    "final operational decision."
                ),
            },
        }

    def deployment_evidence(self) -> dict[str, Any]:
        reports_dir = self.root / "reports" / "v3"
        reports: list[dict[str, Any]] = []
        if reports_dir.exists():
            for path in sorted(reports_dir.glob("session_report_*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    reports.append(payload)
                except (OSError, json.JSONDecodeError):
                    continue

        if not reports:
            return {
                "session_count": 0,
                "total_alerts": 0,
                "total_duration_seconds": 0.0,
                "highest_smoothed_risk": 0.0,
                "dominant_signals": [],
                "summary": "No local deployment reports are available in this export.",
            }

        signal_counts: dict[str, int] = {}
        for report in reports:
            signal = str(report.get("dominant_risk_signal") or "Unknown")
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        dominant = sorted(
            ({"label": key, "count": value} for key, value in signal_counts.items()),
            key=lambda item: item["count"],
            reverse=True,
        )
        return {
            "session_count": len(reports),
            "total_alerts": sum(int(item.get("alert_count", 0) or 0) for item in reports),
            "total_duration_seconds": round(sum(float(item.get("duration_seconds", 0) or 0) for item in reports), 2),
            "highest_smoothed_risk": round(max(float(item.get("maximum_smoothed_probability", 0) or 0) for item in reports), 6),
            "average_baseline_ear": round(sum(float(item.get("baseline_ear", 0) or 0) for item in reports) / len(reports), 6),
            "dominant_signals": dominant,
            "summary": (
                "Aggregate local deployment evidence from generated session reports. "
                "These observations are not a substitute for held-out model validation."
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            last = self._last
        return {
            "available": True,
            "last_audit": last,
            "reference": self.reference(),
            "candidate_paths": self.candidate_paths(),
            "deployment": self.deployment_mode(),
            "lifecycle": self.model_lifecycle(),
            "deployment_evidence": self.deployment_evidence(),
        }
