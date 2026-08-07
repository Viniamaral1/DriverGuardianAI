from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelEvaluationService:
    """Reproducible held-out evaluation for Guardian's saved model bundle."""

    TARGET_CANDIDATES = ("fatigue_level", "binary_target", "target", "state")
    PARTICIPANT_CANDIDATES = ("participant_id", "participant")
    SESSION_CANDIDATES = ("session_id", "session")
    CONDITION_CANDIDATES = ("condition",)

    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_dir = root / "guardian_data" / "research_lab"
        self.output_path = self.output_dir / "latest_model_evaluation.json"
        self._lock = threading.RLock()
        self._last: dict[str, Any] | None = None
        self._load_last()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Convert NumPy values and non-finite floats into strict JSON values."""
        try:
            import numpy as np
        except ImportError:
            np = None

        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
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
                payload = self._json_safe(payload)
                target_column = str(
                    (payload.get("test_split") or {}).get("target_column") or ""
                ).strip().casefold()
                self._last = None if target_column == "state" else payload
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
            "calibration": [
                str(root / "data" / "splits" / "v2" / "calibration.csv")
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
            "last_evaluation": self._last,
            "export_available": self._last is not None and self.output_path.exists(),
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
        lower = {str(column).lower(): str(column) for column in frame.columns}
        for candidate in candidates:
            if candidate.lower() in lower:
                return lower[candidate.lower()]
        return None

    @staticmethod
    def _normalise_target(
        series: Any,
        saved_mapping: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, int]]:
        import pandas as pd

        if saved_mapping:
            clean_mapping = {
                str(label).strip().casefold(): int(value)
                for label, value in saved_mapping.items()
            }
            text = series.fillna("").astype(str).str.strip().str.casefold()
            unknown = sorted(set(text.unique()) - set(clean_mapping))
            if not unknown:
                return text.map(clean_mapping).astype(int), {
                    str(label): int(value)
                    for label, value in saved_mapping.items()
                }

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            unique = sorted(set(int(value) for value in numeric.unique()))
            if set(unique).issubset({0, 1}):
                return numeric.astype(int), {
                    str(value): int(value) for value in unique
                }

        mapping = {
            "alert": 0,
            "normal": 0,
            "awake": 0,
            "non-fatigue": 0,
            "non fatigue": 0,
            "0": 0,
            "mild fatigue": 1,
            "moderate fatigue": 1,
            "severe fatigue": 1,
            "drowsy": 1,
            "fatigue": 1,
            "1": 1,
        }
        text = series.fillna("").astype(str).str.strip().str.casefold()
        unknown = sorted(set(text.unique()) - set(mapping))
        if unknown:
            raise ValueError(
                "Unsupported target labels: "
                + ", ".join(repr(value) for value in unknown[:12])
            )
        return text.map(mapping).astype(int), mapping

    @staticmethod
    def _positive_probability(pipeline: Any, features: Any) -> Any:
        import numpy as np

        if hasattr(pipeline, "predict_proba"):
            probabilities = np.asarray(pipeline.predict_proba(features), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                raise ValueError("Model predict_proba output does not contain two classes.")
            classes = list(getattr(pipeline, "classes_", [0, 1]))
            positive_index = classes.index(1) if 1 in classes else 1
            return probabilities[:, positive_index]

        if hasattr(pipeline, "decision_function"):
            values = np.asarray(pipeline.decision_function(features), dtype=float)
            return 1.0 / (1.0 + np.exp(-values))

        predictions = np.asarray(pipeline.predict(features), dtype=float)
        return predictions

    @classmethod
    def _metrics(cls, y_true: Any, probabilities: Any, threshold: float) -> dict[str, Any]:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = (np.asarray(probabilities) >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        result = {
            "rows": int(len(y_true)),
            "threshold": round(float(threshold), 6),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "positive_share": round(float(sum(y_true) / max(1, len(y_true))), 6),
            "predicted_positive_share": round(float(sum(y_pred) / max(1, len(y_pred))), 6),
        }
        try:
            auc_value = float(roc_auc_score(y_true, probabilities))
            result["roc_auc"] = round(auc_value, 6) if math.isfinite(auc_value) else None
        except ValueError:
            result["roc_auc"] = None
        return cls._json_safe(result)

    @classmethod
    def _calibration_bins(cls, y_true: Any, probabilities: Any, bins: int = 10) -> list[dict[str, Any]]:
        import numpy as np

        probabilities = np.asarray(probabilities, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        edges = np.linspace(0.0, 1.0, bins + 1)
        rows: list[dict[str, Any]] = []
        for index in range(bins):
            lower = float(edges[index])
            upper = float(edges[index + 1])
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper if index == bins - 1 else probabilities < upper)
            )
            if not mask.any():
                continue
            rows.append(
                {
                    "lower": round(lower, 3),
                    "upper": round(upper, 3),
                    "rows": int(mask.sum()),
                    "mean_probability": round(float(probabilities[mask].mean()), 6),
                    "observed_positive_rate": round(float(y_true[mask].mean()), 6),
                }
            )
        return cls._json_safe(rows)

    def _group_metrics(
        self,
        frame: Any,
        column: str | None,
        y_true: Any,
        probabilities: Any,
        threshold: float,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        if not column:
            return []
        rows: list[dict[str, Any]] = []
        values = frame[column].fillna("Missing").astype(str)
        for label in values.value_counts().head(limit).index:
            mask = values == label
            if int(mask.sum()) < 2:
                continue
            metrics = self._metrics(y_true[mask], probabilities[mask], threshold)
            rows.append({"label": str(label), **metrics})
        return rows

    def _threshold_sweep(self, y_true: Any, probabilities: Any) -> list[dict[str, Any]]:
        rows = []
        for index in range(5, 96, 5):
            threshold = index / 100.0
            metrics = self._metrics(y_true, probabilities, threshold)
            rows.append(
                {
                    "threshold": threshold,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            )
        return rows

    def evaluate(
        self,
        *,
        model_path: str,
        test_path: str,
        calibration_path: str | None = None,
    ) -> dict[str, Any]:
        import joblib
        import pandas as pd

        model_file = self._resolve(model_path, "Model", ".joblib")
        test_file = self._resolve(test_path, "Test split", ".csv")
        calibration_file = (
            self._resolve(calibration_path, "Calibration split", ".csv")
            if calibration_path and calibration_path.strip()
            else None
        )

        bundle = joblib.load(model_file)
        if isinstance(bundle, dict):
            pipeline = bundle.get("pipeline") or bundle.get("model")
            threshold = float(bundle.get("fatigue_threshold", bundle.get("threshold", 0.5)))
            feature_columns = list(bundle.get("feature_columns", []))
            class_names = list(bundle.get("class_names", ["Alert", "Fatigue"]))
            saved_metrics = bundle.get("test_metrics", {}) or {}
            saved_target_mapping = bundle.get("target_mapping", {}) or {}
            saved_target_column = str(bundle.get("target_column", "") or "").strip()
            model_name = str(bundle.get("model_name", bundle.get("variant_name", model_file.stem)))
        else:
            pipeline = bundle
            threshold = 0.5
            feature_columns = []
            class_names = ["Alert", "Fatigue"]
            saved_metrics = {}
            saved_target_mapping = {}
            saved_target_column = ""
            model_name = model_file.stem

        if pipeline is None:
            raise ValueError("Model bundle does not contain a pipeline or model.")
        if not feature_columns:
            feature_columns = list(getattr(pipeline, "feature_names_in_", []))
        if not feature_columns:
            raise ValueError("Feature columns could not be determined from the model bundle.")

        test = pd.read_csv(test_file, low_memory=False)
        missing_features = sorted(set(feature_columns) - set(test.columns))
        if missing_features:
            raise ValueError("Test split is missing model features: " + ", ".join(missing_features))

        target_column = None
        if saved_target_column and saved_target_column in test.columns:
            target_column = saved_target_column
        elif saved_target_mapping:
            expected_labels = {
                str(label).strip().casefold()
                for label in saved_target_mapping
            }
            for candidate in self.TARGET_CANDIDATES:
                column = self._column(test, (candidate,))
                if not column:
                    continue
                observed = set(
                    test[column].dropna().astype(str).str.strip().str.casefold().unique()
                )
                if observed and observed.issubset(expected_labels):
                    target_column = column
                    break
        if not target_column:
            target_column = self._column(test, self.TARGET_CANDIDATES)
        if not target_column:
            raise ValueError(
                "Test split needs one target column: "
                + ", ".join(self.TARGET_CANDIDATES)
            )

        y_test, target_mapping = self._normalise_target(
            test[target_column],
            saved_target_mapping,
        )
        test_probabilities = self._positive_probability(pipeline, test[feature_columns])
        test_metrics = self._metrics(y_test, test_probabilities, threshold)

        participant_column = self._column(test, self.PARTICIPANT_CANDIDATES)
        session_column = self._column(test, self.SESSION_CANDIDATES)
        condition_column = self._column(test, self.CONDITION_CANDIDATES)

        calibration_result = None
        threshold_sweep = []
        if calibration_file:
            calibration = pd.read_csv(calibration_file, low_memory=False)
            missing_calibration = sorted(set(feature_columns) - set(calibration.columns))
            if missing_calibration:
                raise ValueError(
                    "Calibration split is missing model features: "
                    + ", ".join(missing_calibration)
                )
            calibration_target = (
                target_column
                if target_column in calibration.columns
                else self._column(calibration, self.TARGET_CANDIDATES)
            )
            if not calibration_target:
                raise ValueError("Calibration split does not contain a supported target column.")
            y_calibration, _ = self._normalise_target(
                calibration[calibration_target],
                saved_target_mapping,
            )
            calibration_probabilities = self._positive_probability(
                pipeline,
                calibration[feature_columns],
            )
            calibration_result = self._metrics(
                y_calibration,
                calibration_probabilities,
                threshold,
            )
            threshold_sweep = self._threshold_sweep(
                y_calibration,
                calibration_probabilities,
            )

        result = {
            "available": True,
            "evaluation_contract_version": "7.2.2-fatigue-level",
            "generated_at": self._now(),
            "model": {
                "name": model_name,
                "path": str(model_file),
                "bytes": model_file.stat().st_size,
                "features": feature_columns,
                "class_names": class_names,
                "saved_threshold": threshold,
                "saved_test_metrics": saved_metrics,
                "target_column": target_column,
                "target_mapping": target_mapping,
                "bundle_type": type(bundle).__name__,
                "pipeline_type": type(pipeline).__name__,
            },
            "test_split": {
                "path": str(test_file),
                "rows": int(len(test)),
                "target_column": target_column,
                "target_mapping": target_mapping,
                "participants": int(test[participant_column].nunique())
                if participant_column else None,
                "sessions": int(test[session_column].nunique())
                if session_column else None,
                "conditions": int(test[condition_column].nunique())
                if condition_column else None,
            },
            "metrics": test_metrics,
            "calibration_split_metrics": calibration_result,
            "threshold_sweep": threshold_sweep,
            "probability_calibration": self._calibration_bins(
                y_test,
                test_probabilities,
            ),
            "condition_performance": self._group_metrics(
                test,
                condition_column,
                y_test,
                test_probabilities,
                threshold,
            ),
            "participant_performance": self._group_metrics(
                test,
                participant_column,
                y_test,
                test_probabilities,
                threshold,
            ),
            "session_performance": self._group_metrics(
                test,
                session_column,
                y_test,
                test_probabilities,
                threshold,
                limit=60,
            ),
            "error_analysis": {
                "false_positive_rate": round(
                    test_metrics["confusion_matrix"]["false_positive"]
                    / max(
                        1,
                        test_metrics["confusion_matrix"]["false_positive"]
                        + test_metrics["confusion_matrix"]["true_negative"],
                    ),
                    6,
                ),
                "false_negative_rate": round(
                    test_metrics["confusion_matrix"]["false_negative"]
                    / max(
                        1,
                        test_metrics["confusion_matrix"]["false_negative"]
                        + test_metrics["confusion_matrix"]["true_positive"],
                    ),
                    6,
                ),
            },
            "boundaries": [
                "Metrics describe the supplied held-out split and saved model artifact.",
                "Condition metrics evaluate robustness; they do not prove visual condition recognition.",
                "Participant-level rows with very few examples should not be over-interpreted.",
                "Live calibrated decisions also include personal baseline and temporal logic not reproduced here.",
            ],
        }


        saved_balanced = saved_metrics.get("balanced_accuracy")
        saved_accuracy = saved_metrics.get("accuracy")
        tolerance = 0.000001
        result["reconciliation"] = {
            "target_contract_source": (
                "model_bundle" if saved_target_mapping else "inferred"
            ),
            "historical_balanced_accuracy": saved_balanced,
            "reproduced_balanced_accuracy": test_metrics["balanced_accuracy"],
            "balanced_accuracy_delta": (
                round(test_metrics["balanced_accuracy"] - float(saved_balanced), 6)
                if saved_balanced is not None else None
            ),
            "historical_accuracy": saved_accuracy,
            "reproduced_accuracy": test_metrics["accuracy"],
            "accuracy_delta": (
                round(test_metrics["accuracy"] - float(saved_accuracy), 6)
                if saved_accuracy is not None else None
            ),
            "matches_saved_evidence": bool(
                saved_balanced is not None
                and abs(test_metrics["balanced_accuracy"] - float(saved_balanced)) <= tolerance
                and saved_accuracy is not None
                and abs(test_metrics["accuracy"] - float(saved_accuracy)) <= tolerance
            ),
            "explanation": (
                "The evaluator used the target mapping stored in the model bundle."
            ),
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
