"""
Logging agent for DriverGuardianAI.

Records:
- Raw model predictions
- Confidence scores
- Temporal decisions
- Alert state
- Input features
- Timestamps

Logs are written to CSV for later analysis.
"""

import csv
import os

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class LoggingAgent:
    """
    CSV-based operational logger for DriverGuardianAI.
    """

    DEFAULT_FIELDS = [
        "timestamp",
        "prediction",
        "predicted_class",
        "confidence",
        "temporal_state",
        "alert_level",
        "trigger_alert",
        "reason",
        "moderate_ratio",
        "mild_or_higher_ratio",
        "average_confidence",
        "consecutive_moderate",
        "history_size",
        "ear",
        "yawn_score",
        "head_tilt",
        "hands_detected",
        "condition",
        "low_light",
        "face_confidence",
        "blink_count"
    ]

    def __init__(
        self,
        log_directory="logs",
        filename=None
    ):
        """
        Initialise the logger.

        Parameters
        ----------
        log_directory:
            Directory where CSV logs are stored.

        filename:
            Optional custom filename. If omitted, a timestamped
            filename is created automatically.
        """

        self.log_directory = Path(
            log_directory
        )

        self.log_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        if filename is None:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = (
                f"driver_guardian_session_{timestamp}.csv"
            )

        self.filepath = self.log_directory / filename

        self.fieldnames = self.DEFAULT_FIELDS.copy()

        self._create_file_if_required()

        print(
            f"Logging Agent ready: {self.filepath}"
        )

    def _create_file_if_required(self):
        """
        Create the CSV file and header when it does not exist.
        """

        if self.filepath.exists():
            return

        with self.filepath.open(
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames
            )

            writer.writeheader()

    def log(
        self,
        features: Dict,
        prediction_result: Dict,
        decision_result
    ) -> str:
        """
        Save one inference and decision event.

        Parameters
        ----------
        features:
            Raw eight-feature dictionary passed to Predictor.

        prediction_result:
            Dictionary returned by Predictor.predict().

        decision_result:
            DecisionResult returned by DecisionAgent.update().

        Returns
        -------
        str
            Path to the CSV log file.
        """

        if hasattr(
            decision_result,
            "to_dict"
        ):
            decision_data = decision_result.to_dict()

        elif isinstance(
            decision_result,
            dict
        ):
            decision_data = decision_result

        else:
            raise TypeError(
                "decision_result must be a DecisionResult "
                "or dictionary."
            )

        row = {
            "timestamp": datetime.now().isoformat(
                timespec="milliseconds"
            ),
            "prediction": prediction_result.get(
                "prediction"
            ),
            "predicted_class": prediction_result.get(
                "predicted_class"
            ),
            "confidence": prediction_result.get(
                "confidence"
            ),
            "temporal_state": decision_data.get(
                "state"
            ),
            "alert_level": decision_data.get(
                "alert_level"
            ),
            "trigger_alert": decision_data.get(
                "trigger_alert"
            ),
            "reason": decision_data.get(
                "reason"
            ),
            "moderate_ratio": decision_data.get(
                "moderate_ratio"
            ),
            "mild_or_higher_ratio": decision_data.get(
                "mild_or_higher_ratio"
            ),
            "average_confidence": decision_data.get(
                "average_confidence"
            ),
            "consecutive_moderate": decision_data.get(
                "consecutive_moderate"
            ),
            "history_size": decision_data.get(
                "history_size"
            ),
            "ear": features.get(
                "ear"
            ),
            "yawn_score": features.get(
                "yawn_score"
            ),
            "head_tilt": features.get(
                "head_tilt"
            ),
            "hands_detected": features.get(
                "hands_detected"
            ),
            "condition": features.get(
                "condition"
            ),
            "low_light": features.get(
                "low_light"
            ),
            "face_confidence": features.get(
                "face_confidence"
            ),
            "blink_count": features.get(
                "blink_count"
            )
        }

        with self.filepath.open(
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames
            )

            writer.writerow(row)

        return str(
            self.filepath
        )