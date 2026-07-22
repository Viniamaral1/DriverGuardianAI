"""
Binary fatigue service for DriverGuardianAI.

This service connects:

- ClassicalPredictor
- Binary temporal decision logic
- LoggingAgent

Production pipeline
-------------------
Eight live features
    -> calibrated Histogram Gradient Boosting predictor
    -> temporal fatigue decision
    -> CSV logging

The neural-network FatigueService remains available as an experiment.
This file provides the production candidate based on Experiment 13.
"""

from collections import deque
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Deque, Dict, Optional, Union
from pathlib import Path

from src.classical_predictor import ClassicalPredictor
from src.logging_agent import LoggingAgent


@dataclass
class BinaryDecisionResult:
    """
    Temporal decision returned by BinaryDecisionAgent.
    """

    state: str
    alert_level: str
    trigger_alert: bool
    reason: str
    fatigue_ratio: float
    average_fatigue_probability: float
    consecutive_fatigue: int
    history_size: int
    model_threshold: float

    def to_dict(self) -> Dict:
        """
        Convert the result into a dictionary.
        """

        return asdict(self)


class BinaryDecisionAgent:
    """
    Temporal reasoning for binary Alert-versus-Fatigue predictions.

    A single model prediction does not immediately trigger an alarm.
    The agent examines a recent window and requires sustained evidence.

    States
    ------
    Monitoring:
        Not enough predictions have been collected.

    Alert:
        Recent evidence does not indicate sustained fatigue.

    Possible Fatigue:
        Fatigue evidence is increasing but is not yet critical.

    Fatigue:
        Sustained fatigue evidence has crossed the critical rule.
    """

    def __init__(
        self,
        window_size: int = 12,
        minimum_history: int = 5,
        warning_ratio_threshold: float = 0.50,
        critical_ratio_threshold: float = 0.70,
        strong_fatigue_probability: float = 0.89,
        consecutive_fatigue_required: int = 3,
        alert_cooldown_seconds: float = 10.0,
    ) -> None:
        """
        Initialise temporal decision settings.
        """

        if window_size < 1:
            raise ValueError(
                "window_size must be at least 1."
            )

        if minimum_history < 1:
            raise ValueError(
                "minimum_history must be at least 1."
            )

        if minimum_history > window_size:
            raise ValueError(
                "minimum_history cannot exceed window_size."
            )

        for name, value in {
            "warning_ratio_threshold": warning_ratio_threshold,
            "critical_ratio_threshold": critical_ratio_threshold,
            "strong_fatigue_probability": strong_fatigue_probability,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if (
            critical_ratio_threshold
            < warning_ratio_threshold
        ):
            raise ValueError(
                "critical_ratio_threshold must be greater than "
                "or equal to warning_ratio_threshold."
            )

        if consecutive_fatigue_required < 1:
            raise ValueError(
                "consecutive_fatigue_required must be at least 1."
            )

        if alert_cooldown_seconds < 0:
            raise ValueError(
                "alert_cooldown_seconds cannot be negative."
            )

        self.window_size = int(
            window_size
        )

        self.minimum_history = int(
            minimum_history
        )

        self.warning_ratio_threshold = float(
            warning_ratio_threshold
        )

        self.critical_ratio_threshold = float(
            critical_ratio_threshold
        )

        self.strong_fatigue_probability = float(
            strong_fatigue_probability
        )

        self.consecutive_fatigue_required = int(
            consecutive_fatigue_required
        )

        self.alert_cooldown_seconds = float(
            alert_cooldown_seconds
        )

        self.history: Deque[Dict] = deque(
            maxlen=self.window_size
        )

        self.last_alert_time: Optional[float] = None

    def reset(self) -> None:
        """
        Reset temporal history and alert cooldown.
        """

        self.history.clear()
        self.last_alert_time = None

    def _count_consecutive_fatigue(self) -> int:
        """
        Count strong Fatigue predictions from the end of history.
        """

        count = 0

        for item in reversed(
            self.history
        ):
            if (
                item["prediction"] == "Fatigue"
                and item["fatigue_probability"]
                >= self.strong_fatigue_probability
            ):
                count += 1
            else:
                break

        return count

    def _can_trigger_alert(self) -> bool:
        """
        Return True when the alert cooldown has expired.
        """

        if self.last_alert_time is None:
            return True

        return (
            monotonic() - self.last_alert_time
            >= self.alert_cooldown_seconds
        )

    def update(
        self,
        prediction_result: Dict,
    ) -> BinaryDecisionResult:
        """
        Add one model prediction and produce a temporal decision.
        """

        required_keys = {
            "prediction",
            "fatigue_probability",
            "threshold",
        }

        missing_keys = required_keys.difference(
            prediction_result.keys()
        )

        if missing_keys:
            raise KeyError(
                "Prediction result is missing keys: "
                f"{sorted(missing_keys)}"
            )

        prediction = str(
            prediction_result["prediction"]
        )

        if prediction not in {
            "Alert",
            "Fatigue",
        }:
            raise ValueError(
                "BinaryDecisionAgent expects prediction "
                "'Alert' or 'Fatigue'."
            )

        fatigue_probability = float(
            prediction_result[
                "fatigue_probability"
            ]
        )

        threshold = float(
            prediction_result["threshold"]
        )

        self.history.append(
            {
                "prediction": prediction,
                "fatigue_probability": (
                    fatigue_probability
                ),
            }
        )

        history_size = len(
            self.history
        )

        fatigue_count = sum(
            item["prediction"] == "Fatigue"
            for item in self.history
        )

        fatigue_ratio = (
            fatigue_count / history_size
            if history_size > 0
            else 0.0
        )

        average_fatigue_probability = sum(
            item["fatigue_probability"]
            for item in self.history
        ) / history_size

        consecutive_fatigue = (
            self._count_consecutive_fatigue()
        )

        if history_size < self.minimum_history:
            return BinaryDecisionResult(
                state="Monitoring",
                alert_level="none",
                trigger_alert=False,
                reason=(
                    "Collecting more predictions before making "
                    "a temporal decision."
                ),
                fatigue_ratio=fatigue_ratio,
                average_fatigue_probability=(
                    average_fatigue_probability
                ),
                consecutive_fatigue=(
                    consecutive_fatigue
                ),
                history_size=history_size,
                model_threshold=threshold,
            )

        critical_by_consecutive = (
            consecutive_fatigue
            >= self.consecutive_fatigue_required
        )

        critical_by_window = (
            fatigue_ratio
            >= self.critical_ratio_threshold
            and average_fatigue_probability
            >= self.strong_fatigue_probability
        )

        if (
            critical_by_consecutive
            or critical_by_window
        ):
            trigger_alert = (
                self._can_trigger_alert()
            )

            if trigger_alert:
                self.last_alert_time = monotonic()

            if critical_by_consecutive:
                reason = (
                    f"{consecutive_fatigue} consecutive strong "
                    "Fatigue predictions detected."
                )
            else:
                reason = (
                    "Sustained Fatigue predictions and high average "
                    "Fatigue probability detected."
                )

            return BinaryDecisionResult(
                state="Fatigue",
                alert_level="critical",
                trigger_alert=trigger_alert,
                reason=reason,
                fatigue_ratio=fatigue_ratio,
                average_fatigue_probability=(
                    average_fatigue_probability
                ),
                consecutive_fatigue=(
                    consecutive_fatigue
                ),
                history_size=history_size,
                model_threshold=threshold,
            )

        if (
            fatigue_ratio
            >= self.warning_ratio_threshold
        ):
            return BinaryDecisionResult(
                state="Possible Fatigue",
                alert_level="warning",
                trigger_alert=False,
                reason=(
                    "A substantial proportion of recent predictions "
                    "indicate Fatigue, but the critical rule has not "
                    "been reached."
                ),
                fatigue_ratio=fatigue_ratio,
                average_fatigue_probability=(
                    average_fatigue_probability
                ),
                consecutive_fatigue=(
                    consecutive_fatigue
                ),
                history_size=history_size,
                model_threshold=threshold,
            )

        return BinaryDecisionResult(
            state="Alert",
            alert_level="none",
            trigger_alert=False,
            reason=(
                "Recent predictions do not indicate sustained Fatigue."
            ),
            fatigue_ratio=fatigue_ratio,
            average_fatigue_probability=(
                average_fatigue_probability
            ),
            consecutive_fatigue=(
                consecutive_fatigue
            ),
            history_size=history_size,
            model_threshold=threshold,
        )


class BinaryFatigueService:
    """
    End-to-end binary fatigue inference service.
    """

    def __init__(
        self,
        model_path: Union[
            str,
            Path,
        ] = (
            "models/"
            "driver_guardian_calibrated_hgb.joblib"
        ),
        threshold: Optional[float] = None,
        enable_logging: bool = True,
        log_directory: Union[
            str,
            Path,
        ] = "logs",
        window_size: int = 12,
        minimum_history: int = 5,
        warning_ratio_threshold: float = 0.50,
        critical_ratio_threshold: float = 0.70,
        consecutive_fatigue_required: int = 3,
        alert_cooldown_seconds: float = 10.0,
    ) -> None:
        """
        Initialise predictor, temporal decision agent, and logger.
        """

        self.predictor = ClassicalPredictor(
            model_path=model_path,
            threshold=threshold,
        )

        self.decision_agent = (
            BinaryDecisionAgent(
                window_size=window_size,
                minimum_history=minimum_history,
                warning_ratio_threshold=(
                    warning_ratio_threshold
                ),
                critical_ratio_threshold=(
                    critical_ratio_threshold
                ),
                strong_fatigue_probability=(
                    self.predictor.threshold
                ),
                consecutive_fatigue_required=(
                    consecutive_fatigue_required
                ),
                alert_cooldown_seconds=(
                    alert_cooldown_seconds
                ),
            )
        )

        self.logging_agent: Optional[
            LoggingAgent
        ] = None

        if enable_logging:
            self.logging_agent = LoggingAgent(
                log_directory=str(
                    log_directory
                )
            )

    def reset_session(self) -> None:
        """
        Reset temporal decision memory.
        """

        self.decision_agent.reset()

    @staticmethod
    def _convert_decision_for_logger(
        decision: BinaryDecisionResult,
    ) -> Dict:
        """
        Convert the binary decision into fields compatible with the
        existing LoggingAgent.

        Some field names are retained for compatibility with old logs:
        - moderate_ratio stores the binary fatigue ratio;
        - mild_or_higher_ratio also stores the fatigue ratio;
        - average_confidence stores average Fatigue probability;
        - consecutive_moderate stores consecutive Fatigue predictions.
        """

        decision_data = decision.to_dict()

        return {
            "state": decision_data["state"],
            "alert_level": (
                decision_data["alert_level"]
            ),
            "trigger_alert": (
                decision_data["trigger_alert"]
            ),
            "reason": decision_data["reason"],
            "moderate_ratio": (
                decision_data["fatigue_ratio"]
            ),
            "mild_or_higher_ratio": (
                decision_data["fatigue_ratio"]
            ),
            "average_confidence": (
                decision_data[
                    "average_fatigue_probability"
                ]
            ),
            "consecutive_moderate": (
                decision_data[
                    "consecutive_fatigue"
                ]
            ),
            "history_size": (
                decision_data["history_size"]
            ),
        }

    def process(
        self,
        features: Dict,
    ) -> Dict:
        """
        Process one feature dictionary through the full binary system.
        """

        prediction = self.predictor.predict(
            features
        )

        decision = self.decision_agent.update(
            prediction
        )

        log_path = None

        if self.logging_agent is not None:
            logging_decision = (
                self._convert_decision_for_logger(
                    decision
                )
            )

            log_path = self.logging_agent.log(
                features=features,
                prediction_result=prediction,
                decision_result=logging_decision,
            )

        return {
            "prediction": prediction,
            "decision": decision.to_dict(),
            "log_path": log_path,
        }

    def model_information(self) -> Dict:
        """
        Return predictor and decision configuration.
        """

        return {
            "predictor": (
                self.predictor.model_information()
            ),
            "decision_agent": {
                "window_size": (
                    self.decision_agent.window_size
                ),
                "minimum_history": (
                    self.decision_agent.minimum_history
                ),
                "warning_ratio_threshold": (
                    self.decision_agent
                    .warning_ratio_threshold
                ),
                "critical_ratio_threshold": (
                    self.decision_agent
                    .critical_ratio_threshold
                ),
                "consecutive_fatigue_required": (
                    self.decision_agent
                    .consecutive_fatigue_required
                ),
                "alert_cooldown_seconds": (
                    self.decision_agent
                    .alert_cooldown_seconds
                ),
            },
            "logging_enabled": (
                self.logging_agent is not None
            ),
        }