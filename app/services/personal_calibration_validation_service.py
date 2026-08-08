from __future__ import annotations

import importlib
import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PersonalCalibrationValidationService:
    """Controlled replay of Guardian's real V3 personal-calibration fusion."""

    TARGET_CANDIDATES = ("fatigue_level", "binary_target", "target")
    PARTICIPANT_CANDIDATES = ("participant_id", "participant")
    SESSION_CANDIDATES = ("session_id", "session")
    CONDITION_CANDIDATES = ("condition",)
    STATE_CANDIDATES = ("state",)

    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_dir = root / "guardian_data" / "research_lab"
        self.output_path = self.output_dir / "latest_calibration_validation.json"
        self._lock = threading.RLock()
        self._last: dict[str, Any] | None = None
        self._load_last()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        try:
            import numpy as np
        except ImportError:
            np = None
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        if np is not None and isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value

    def _load_last(self) -> None:
        try:
            payload = json.loads(
                self.output_path.read_text(encoding="utf-8"),
                parse_constant=lambda _value: None,
            )
            if isinstance(payload, dict):
                self._last = self._json_safe(payload)
        except (OSError, json.JSONDecodeError):
            self._last = None

    def candidate_paths(self) -> dict[str, list[str]]:
        roots = [
            self.root,
            self.root.parent / "DriverGuardianAI",
            Path.home() / "DriverGuardianAI",
        ]
        return {
            "model": [
                str(root / "models" / "v2" / "ablation" / "driver_guardian_core_behaviour.joblib")
                for root in roots
            ],
            "test": [
                str(root / "data" / "splits" / "v2" / "test.csv")
                for root in roots
            ],
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "available": True,
            "candidate_paths": self.candidate_paths(),
            "last_validation": self._last,
            "export_available": self.output_path.exists(),
        }

    def _resolve(self, value: str, label: str, suffix: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (self.root / path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{label} was not found: {path}")
        if path.suffix.lower() != suffix:
            raise ValueError(f"{label} must be a {suffix} file.")
        return path

    @staticmethod
    def _column(frame: Any, candidates: tuple[str, ...]) -> str | None:
        lookup = {str(c).lower(): str(c) for c in frame.columns}
        for candidate in candidates:
            if candidate.lower() in lookup:
                return lookup[candidate.lower()]
        return None

    @staticmethod
    def _target(series: Any, saved_mapping: dict[str, Any]) -> Any:
        import pandas as pd
        if saved_mapping:
            mapping = {str(k).strip().casefold(): int(v) for k, v in saved_mapping.items()}
            text = series.fillna("").astype(str).str.strip().str.casefold()
            unknown = sorted(set(text.unique()) - set(mapping))
            if unknown:
                raise ValueError(
                    "Labels not present in the model target contract: "
                    + ", ".join(repr(v) for v in unknown[:10])
                )
            return text.map(mapping).astype(int)

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and set(numeric.astype(int).unique()).issubset({0, 1}):
            return numeric.astype(int)

        mapping = {
            "alert": 0,
            "mild fatigue": 1,
            "moderate fatigue": 1,
            "severe fatigue": 1,
        }
        text = series.fillna("").astype(str).str.strip().str.casefold()
        unknown = sorted(set(text.unique()) - set(mapping))
        if unknown:
            raise ValueError("Unsupported target labels: " + ", ".join(unknown[:10]))
        return text.map(mapping).astype(int)

    @staticmethod
    def _positive_probability(model: Any, features: Any) -> Any:
        import numpy as np
        if hasattr(model, "predict_proba"):
            values = np.asarray(model.predict_proba(features), dtype=float)
            classes = list(getattr(model, "classes_", [0, 1]))
            index = classes.index(1) if 1 in classes else 1
            return values[:, index]
        if hasattr(model, "decision_function"):
            values = np.asarray(model.decision_function(features), dtype=float)
            return 1.0 / (1.0 + np.exp(-values))
        return np.asarray(model.predict(features), dtype=float)

    @classmethod
    def _metrics(cls, y_true: Any, probabilities: Any, threshold: float) -> dict[str, Any]:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, confusion_matrix,
            f1_score, precision_score, recall_score, roc_auc_score,
        )
        y_true = np.asarray(y_true, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        y_pred = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in matrix.ravel())
        result = {
            "rows": int(len(y_true)),
            "threshold": round(float(threshold), 6),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
            "confusion_matrix": {
                "true_negative": tn, "false_positive": fp,
                "false_negative": fn, "true_positive": tp,
            },
        }
        try:
            auc = float(roc_auc_score(y_true, probabilities))
            result["roc_auc"] = round(auc, 6) if math.isfinite(auc) else None
        except ValueError:
            result["roc_auc"] = None
        return cls._json_safe(result)

    @staticmethod
    def _features(row: Any) -> dict[str, float | bool]:
        return {
            "ear": float(row["ear"]),
            "yawn_score": float(row["yawn_score"]),
            "head_tilt": float(row["head_tilt"]),
            "face_detected": True,
        }

    @staticmethod
    def _elapsed_seconds(frame: Any) -> Any:
        import pandas as pd
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if timestamps.notna().all():
            return (timestamps - timestamps.iloc[0]).dt.total_seconds()
        return pd.Series(
            [index / 12.0 for index in range(len(frame))],
            index=frame.index,
        )

    def validate(
        self,
        *,
        model_path: str,
        test_path: str,
        required_seconds: float = 10.0,
        minimum_samples: int = 80,
    ) -> dict[str, Any]:
        import joblib
        import pandas as pd

        model_file = self._resolve(model_path, "Model", ".joblib")
        test_file = self._resolve(test_path, "Held-out test split", ".csv")

        try:
            v3 = importlib.import_module("realtime_driver_guardian_v3_alerts")
        except ImportError as error:
            raise RuntimeError(
                "The standalone V3 module could not be imported. Run Guardian "
                "from the same project/environment where Monitoring works."
            ) from error

        calibration_class = getattr(v3, "PersonalCalibration", None)
        if calibration_class is None:
            raise RuntimeError("V3 does not expose PersonalCalibration.")

        bundle = joblib.load(model_file)
        if not isinstance(bundle, dict):
            raise ValueError("Expected Guardian's saved model bundle.")

        model = bundle.get("pipeline") or bundle.get("model")
        if model is None:
            raise ValueError("Model bundle does not contain a pipeline or model.")

        feature_columns = list(bundle.get("feature_columns", []))
        if not feature_columns:
            feature_columns = list(getattr(model, "feature_names_in_", []))
        expected = ["ear", "yawn_score", "head_tilt"]
        if feature_columns != expected:
            raise ValueError(
                "This validator is for the core-behaviour model. "
                f"Found features: {feature_columns}"
            )

        threshold = float(bundle.get("fatigue_threshold", bundle.get("threshold", 0.5)))
        target_mapping = bundle.get("target_mapping", {}) or {}
        saved_target = str(bundle.get("target_column", "") or "").strip()

        frame = pd.read_csv(test_file, low_memory=False)
        participant_col = self._column(frame, self.PARTICIPANT_CANDIDATES)
        session_col = self._column(frame, self.SESSION_CANDIDATES)
        condition_col = self._column(frame, self.CONDITION_CANDIDATES)
        state_col = self._column(frame, self.STATE_CANDIDATES)
        target_col = saved_target if saved_target in frame.columns else self._column(
            frame, self.TARGET_CANDIDATES
        )

        columns = {
            "participant": participant_col, "session": session_col,
            "condition": condition_col, "state": state_col, "target": target_col,
        }
        missing = [name for name, value in columns.items() if not value]
        if missing:
            raise ValueError(
                "Calibration replay needs test columns: " + ", ".join(missing)
            )

        frame = frame.copy()
        frame["_target"] = self._target(frame[target_col], target_mapping)
        frame["_raw_probability"] = self._positive_probability(
            model, frame[feature_columns]
        )

        pair_results: list[dict[str, Any]] = []
        all_y: list[int] = []
        raw_probabilities: list[float] = []
        fused_probabilities: list[float] = []

        for (participant, condition), group in frame.groupby(
            [participant_col, condition_col], sort=True
        ):
            sessions = []
            for session_id, session_frame in group.groupby(session_col, sort=False):
                states = (
                    session_frame[state_col].fillna("").astype(str)
                    .str.strip().str.casefold()
                )
                majority = states.value_counts().index[0] if not states.empty else ""
                sessions.append((session_id, majority, session_frame.copy()))

            normal_sessions = [item for item in sessions if item[1] == "normal"]
            drowsy_sessions = [item for item in sessions if item[1] == "drowsy"]
            if not normal_sessions or not drowsy_sessions:
                continue

            normal_id, _, normal = normal_sessions[0]
            drowsy_id, _, drowsy = drowsy_sessions[0]
            normal = normal.sort_values("timestamp").copy()
            drowsy = drowsy.sort_values("timestamp").copy()

            calibration = calibration_class(
                required_seconds=float(required_seconds),
                minimum_samples=int(minimum_samples),
            )
            elapsed = self._elapsed_seconds(normal)
            calibration_end = None

            for position, (_, row) in enumerate(normal.iterrows()):
                calibration.update(self._features(row), float(elapsed.iloc[position]))
                if bool(getattr(calibration, "complete", False)):
                    calibration_end = position
                    break

            if calibration_end is None:
                pair_results.append({
                    "participant": str(participant),
                    "condition": str(condition),
                    "status": "calibration_incomplete",
                    "normal_session": str(normal_id),
                    "drowsy_session": str(drowsy_id),
                })
                continue

            scored_normal = normal.iloc[calibration_end + 1 :].copy()
            evaluation = pd.concat([scored_normal, drowsy], ignore_index=True)

            pair_y, pair_raw, pair_fused = [], [], []
            for _, row in evaluation.iterrows():
                raw = float(row["_raw_probability"])
                fused, _evidence = calibration.calculate_fused_probability(
                    raw, self._features(row)
                )
                pair_y.append(int(row["_target"]))
                pair_raw.append(raw)
                pair_fused.append(float(fused))

            if not pair_y:
                continue

            generic = self._metrics(pair_y, pair_raw, threshold)
            personalized = self._metrics(pair_y, pair_fused, threshold)

            all_y.extend(pair_y)
            raw_probabilities.extend(pair_raw)
            fused_probabilities.extend(pair_fused)

            pair_results.append({
                "participant": str(participant),
                "condition": str(condition),
                "status": "evaluated",
                "normal_session": str(normal_id),
                "drowsy_session": str(drowsy_id),
                "calibration_rows": int(calibration_end + 1),
                "scored_normal_rows": int(len(scored_normal)),
                "scored_drowsy_rows": int(len(drowsy)),
                "baseline_ear": round(float(calibration.baseline_ear), 6),
                "baseline_yawn": round(float(calibration.baseline_yawn), 6),
                "baseline_tilt": round(float(calibration.baseline_tilt), 6),
                "generic": generic,
                "personalized_fusion": personalized,
                "delta_balanced_accuracy": round(
                    personalized["balanced_accuracy"] - generic["balanced_accuracy"], 6
                ),
                "delta_false_positive": (
                    personalized["confusion_matrix"]["false_positive"]
                    - generic["confusion_matrix"]["false_positive"]
                ),
                "delta_false_negative": (
                    personalized["confusion_matrix"]["false_negative"]
                    - generic["confusion_matrix"]["false_negative"]
                ),
            })

        evaluated = [row for row in pair_results if row.get("status") == "evaluated"]
        if not evaluated:
            raise ValueError(
                "No held-out normal/drowsy participant-condition pairs completed calibration."
            )

        generic = self._metrics(all_y, raw_probabilities, threshold)
        personalized = self._metrics(all_y, fused_probabilities, threshold)

        result = {
            "available": True,
            "protocol_version": "7.3-calibration-replay-v1",
            "generated_at": self._now(),
            "model": {
                "path": str(model_file),
                "name": str(bundle.get("variant_name", bundle.get("model_name", model_file.stem))),
                "features": feature_columns,
                "threshold": threshold,
                "target_column": target_col,
            },
            "test_split": {
                "path": str(test_file),
                "rows": int(len(frame)),
                "participants": int(frame[participant_col].nunique()),
                "conditions": sorted(frame[condition_col].astype(str).unique().tolist()),
            },
            "protocol": {
                "baseline_source": (
                    "first held-out normal session for the same participant and condition"
                ),
                "baseline_seconds": float(required_seconds),
                "minimum_samples": int(minimum_samples),
                "baseline_rows_excluded_from_scoring": True,
                "paired_drowsy_session_scored": True,
                "uses_exact_v3_personal_calibration_fusion": True,
                "temporal_state_engine_included": False,
                "alert_manager_included": False,
                "interpretation": (
                    "Controlled replay of personal risk fusion; not final live alert accuracy."
                ),
            },
            "generic_model": generic,
            "personalized_fusion": personalized,
            "comparison": {
                "balanced_accuracy_delta": round(
                    personalized["balanced_accuracy"] - generic["balanced_accuracy"], 6
                ),
                "accuracy_delta": round(
                    personalized["accuracy"] - generic["accuracy"], 6
                ),
                "f1_delta": round(personalized["f1"] - generic["f1"], 6),
                "false_positive_delta": (
                    personalized["confusion_matrix"]["false_positive"]
                    - generic["confusion_matrix"]["false_positive"]
                ),
                "false_negative_delta": (
                    personalized["confusion_matrix"]["false_negative"]
                    - generic["confusion_matrix"]["false_negative"]
                ),
            },
            "pairs": pair_results,
            "boundaries": [
                "The trained model is unchanged.",
                "Normal-session state is used only to define the research baseline protocol.",
                "Calibration rows are excluded from scoring.",
                "Temporal smoothing and alert cooldown are not included.",
                "Calibration is not assumed to improve every group; report the measured delta.",
            ],
        }

        result = self._json_safe(result)
        with self._lock:
            self._last = result
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(
                json.dumps(result, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        return result
