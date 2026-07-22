"""
Calibrated classical predictor for DriverGuardianAI.

This module loads the Histogram Gradient Boosting model produced by
Experiment 13 and provides production-style single-sample and batch
prediction methods.

The saved model bundle contains:
- fitted preprocessing pipeline;
- fitted Histogram Gradient Boosting classifier;
- calibrated Fatigue probability threshold;
- expected feature order;
- class names;
- participant split metadata.

Binary classes
--------------
0 = Alert
1 = Fatigue

The predictor does not retrain or modify the saved model.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = Path(
    "models/driver_guardian_calibrated_hgb.joblib"
)

DEFAULT_FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count",
]

DEFAULT_CLASS_NAMES = [
    "Alert",
    "Fatigue",
]


FeatureInput = Union[
    Mapping[str, Any],
    pd.Series,
    pd.DataFrame,
]


class ClassicalPredictor:
    """
    Production predictor for the calibrated classical model.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Load the saved model bundle.

        Parameters
        ----------
        model_path:
            Path to the joblib bundle produced by Experiment 13.

        threshold:
            Optional manual Fatigue probability threshold.
            When omitted, the calibrated threshold stored in the
            model bundle is used.
        """

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Calibrated classical model was not found: "
                f"{self.model_path}"
            )

        bundle = joblib.load(
            self.model_path
        )

        if not isinstance(bundle, dict):
            raise TypeError(
                "The saved model file must contain a dictionary bundle."
            )

        required_keys = {
            "pipeline",
            "fatigue_threshold",
            "feature_columns",
            "class_names",
        }

        missing_keys = required_keys.difference(
            bundle.keys()
        )

        if missing_keys:
            raise KeyError(
                "The saved model bundle is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        self.pipeline = bundle[
            "pipeline"
        ]

        self.feature_columns = list(
            bundle[
                "feature_columns"
            ]
        )

        self.class_names = list(
            bundle[
                "class_names"
            ]
        )

        self.training_participants = list(
            bundle.get(
                "training_participants",
                [],
            )
        )

        self.calibration_participants = list(
            bundle.get(
                "calibration_participants",
                [],
            )
        )

        self.test_participants = list(
            bundle.get(
                "test_participants",
                [],
            )
        )

        stored_threshold = float(
            bundle[
                "fatigue_threshold"
            ]
        )

        self.threshold = (
            stored_threshold
            if threshold is None
            else float(threshold)
        )

        self._validate_loaded_bundle()

        print(
            "Classical model loaded successfully."
        )

        print(
            "Model type: Histogram Gradient Boosting"
        )

        print(
            "Expected features "
            f"({len(self.feature_columns)}): "
            f"{self.feature_columns}"
        )

        print(
            "Fatigue threshold: "
            f"{self.threshold:.2f}"
        )

    def _validate_loaded_bundle(self) -> None:
        """
        Validate model metadata and prediction capabilities.
        """

        if not hasattr(
            self.pipeline,
            "predict_proba",
        ):
            raise AttributeError(
                "The saved pipeline does not provide predict_proba()."
            )

        if self.feature_columns != DEFAULT_FEATURE_COLUMNS:
            raise ValueError(
                "Unexpected feature order in the saved model.\n"
                f"Expected: {DEFAULT_FEATURE_COLUMNS}\n"
                f"Received: {self.feature_columns}"
            )

        if self.class_names != DEFAULT_CLASS_NAMES:
            raise ValueError(
                "Unexpected class names in the saved model.\n"
                f"Expected: {DEFAULT_CLASS_NAMES}\n"
                f"Received: {self.class_names}"
            )

        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                "The Fatigue threshold must be between 0 and 1."
            )

    @staticmethod
    def _normalise_boolean(
        value: Any,
    ) -> float:
        """
        Convert common boolean forms to 0.0 or 1.0.
        """

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            return float(
                bool(value)
            )

        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            numeric_value = float(value)

            if numeric_value in {
                0.0,
                1.0,
            }:
                return numeric_value

        text = str(value).strip().lower()

        mapping = {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "1": 1.0,
            "0": 0.0,
            "1.0": 1.0,
            "0.0": 0.0,
        }

        if text not in mapping:
            raise ValueError(
                "low_light must be a boolean or a value equivalent "
                f"to 0 or 1. Received: {value!r}"
            )

        return mapping[
            text
        ]

    @staticmethod
    def _normalise_condition(
        value: Any,
    ) -> str:
        """
        Validate and normalise the recording condition.
        """

        condition = str(
            value
        ).strip().lower()

        allowed_conditions = {
            "none",
            "glasses",
            "hat",
            "dark",
        }

        if condition not in allowed_conditions:
            raise ValueError(
                "condition must be one of "
                f"{sorted(allowed_conditions)}. "
                f"Received: {value!r}"
            )

        return condition

    @staticmethod
    def _normalise_numeric(
        value: Any,
        feature_name: str,
    ) -> float:
        """
        Convert one feature to a finite float.
        """

        try:
            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{feature_name} must be numeric. "
                f"Received: {value!r}"
            ) from error

        if not np.isfinite(
            numeric_value
        ):
            raise ValueError(
                f"{feature_name} must be finite. "
                f"Received: {value!r}"
            )

        return numeric_value

    def _prepare_mapping(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate and standardise one feature dictionary.
        """

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in features
        ]

        unexpected_features = [
            feature
            for feature in features
            if feature not in self.feature_columns
        ]

        if missing_features:
            raise ValueError(
                "Missing required features: "
                f"{missing_features}"
            )

        if unexpected_features:
            raise ValueError(
                "Unexpected features were supplied: "
                f"{unexpected_features}"
            )

        prepared = {
            "ear": self._normalise_numeric(
                features["ear"],
                "ear",
            ),
            "yawn_score": self._normalise_numeric(
                features["yawn_score"],
                "yawn_score",
            ),
            "head_tilt": self._normalise_numeric(
                features["head_tilt"],
                "head_tilt",
            ),
            "hands_detected": self._normalise_numeric(
                features["hands_detected"],
                "hands_detected",
            ),
            "condition": self._normalise_condition(
                features["condition"]
            ),
            "low_light": self._normalise_boolean(
                features["low_light"]
            ),
            "face_confidence": self._normalise_numeric(
                features["face_confidence"],
                "face_confidence",
            ),
            "blink_count": self._normalise_numeric(
                features["blink_count"],
                "blink_count",
            ),
        }

        return prepared

    def _prepare_dataframe(
        self,
        features: FeatureInput,
    ) -> pd.DataFrame:
        """
        Convert supported inputs into a validated DataFrame.
        """

        if isinstance(
            features,
            pd.DataFrame,
        ):
            raw_dataframe = features.copy()

        elif isinstance(
            features,
            pd.Series,
        ):
            raw_dataframe = pd.DataFrame(
                [
                    features.to_dict()
                ]
            )

        elif isinstance(
            features,
            Mapping,
        ):
            raw_dataframe = pd.DataFrame(
                [
                    dict(features)
                ]
            )

        else:
            raise TypeError(
                "features must be a dictionary, pandas Series, "
                "or pandas DataFrame."
            )

        if raw_dataframe.empty:
            raise ValueError(
                "No prediction samples were provided."
            )

        prepared_rows = []

        for row_index, row in raw_dataframe.iterrows():
            try:
                prepared_rows.append(
                    self._prepare_mapping(
                        row.to_dict()
                    )
                )

            except Exception as error:
                raise ValueError(
                    "Invalid feature data at row "
                    f"{row_index}: {error}"
                ) from error

        return pd.DataFrame(
            prepared_rows,
            columns=self.feature_columns,
        )

    def _build_result(
        self,
        alert_probability: float,
        fatigue_probability: float,
    ) -> Dict[str, Any]:
        """
        Build one production prediction dictionary.
        """

        predicted_class = int(
            fatigue_probability
            >= self.threshold
        )

        prediction = self.class_names[
            predicted_class
        ]

        confidence = (
            fatigue_probability
            if predicted_class == 1
            else alert_probability
        )

        probability_margin = abs(
            fatigue_probability
            - self.threshold
        )

        return {
            "prediction": prediction,
            "predicted_class": predicted_class,
            "confidence": float(
                confidence
            ),
            "fatigue_probability": float(
                fatigue_probability
            ),
            "alert_probability": float(
                alert_probability
            ),
            "threshold": float(
                self.threshold
            ),
            "probability_margin": float(
                probability_margin
            ),
            "probabilities": {
                "Alert": float(
                    alert_probability
                ),
                "Fatigue": float(
                    fatigue_probability
                ),
            },
            "model_type": (
                "hist_gradient_boosting"
            ),
        }

    def predict(
        self,
        features: FeatureInput,
    ) -> Dict[str, Any]:
        """
        Predict one sample.

        Parameters
        ----------
        features:
            Dictionary, Series, or one-row DataFrame containing the
            eight expected features.

        Returns
        -------
        dict
            Prediction, confidence, probabilities, and threshold.
        """

        dataframe = self._prepare_dataframe(
            features
        )

        if len(dataframe) != 1:
            raise ValueError(
                "predict() accepts exactly one sample. "
                "Use predict_batch() for multiple rows."
            )

        probabilities = self.pipeline.predict_proba(
            dataframe
        )[0]

        alert_probability = float(
            probabilities[0]
        )

        fatigue_probability = float(
            probabilities[1]
        )

        return self._build_result(
            alert_probability=alert_probability,
            fatigue_probability=fatigue_probability,
        )

    def predict_batch(
        self,
        features: FeatureInput,
    ) -> List[Dict[str, Any]]:
        """
        Predict one or more samples.

        Parameters
        ----------
        features:
            DataFrame or another supported feature input.

        Returns
        -------
        list of dict
            One prediction dictionary per input row.
        """

        dataframe = self._prepare_dataframe(
            features
        )

        probability_matrix = (
            self.pipeline.predict_proba(
                dataframe
            )
        )

        results = []

        for probabilities in probability_matrix:
            results.append(
                self._build_result(
                    alert_probability=float(
                        probabilities[0]
                    ),
                    fatigue_probability=float(
                        probabilities[1]
                    ),
                )
            )

        return results

    def set_threshold(
        self,
        threshold: float,
    ) -> None:
        """
        Update the in-memory decision threshold.

        This does not change or overwrite the saved model file.
        """

        threshold = float(
            threshold
        )

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.threshold = threshold

    def model_information(
        self,
    ) -> Dict[str, Any]:
        """
        Return model metadata for diagnostics or an API endpoint.
        """

        return {
            "model_path": str(
                self.model_path
            ),
            "model_type": (
                "hist_gradient_boosting"
            ),
            "feature_columns": (
                self.feature_columns.copy()
            ),
            "class_names": (
                self.class_names.copy()
            ),
            "fatigue_threshold": float(
                self.threshold
            ),
            "training_participants": (
                self.training_participants.copy()
            ),
            "calibration_participants": (
                self.calibration_participants.copy()
            ),
            "test_participants": (
                self.test_participants.copy()
            ),
        }