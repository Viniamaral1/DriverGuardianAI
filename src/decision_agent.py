"""
Temporal decision agent for DriverGuardianAI.

The DecisionAgent analyses a rolling history of fatigue predictions
instead of reacting to a single frame.

It helps reduce false alarms by considering:

- Recent predicted classes
- Prediction confidence
- Moderate-fatigue frequency
- Consecutive fatigue predictions
- Cooldown between repeated alerts
"""

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Deque, Dict, Optional


@dataclass
class DecisionResult:
    """
    Result returned by the temporal decision agent.
    """

    state: str
    alert_level: str
    trigger_alert: bool
    reason: str
    moderate_ratio: float
    mild_or_higher_ratio: float
    average_confidence: float
    consecutive_moderate: int
    history_size: int

    def to_dict(self) -> Dict:
        """
        Convert the result to a normal dictionary.
        """

        return {
            "state": self.state,
            "alert_level": self.alert_level,
            "trigger_alert": self.trigger_alert,
            "reason": self.reason,
            "moderate_ratio": self.moderate_ratio,
            "mild_or_higher_ratio": self.mild_or_higher_ratio,
            "average_confidence": self.average_confidence,
            "consecutive_moderate": self.consecutive_moderate,
            "history_size": self.history_size
        }


class DecisionAgent:
    """
    Temporal decision engine for fatigue predictions.

    Class IDs
    ---------
    0 : Alert
    1 : Mild Fatigue
    2 : Moderate/Severe Fatigue
    """

    ALERT_CLASS = 0
    MILD_CLASS = 1
    MODERATE_CLASS = 2

    def __init__(
        self,
        window_size: int = 15,
        minimum_history: int = 5,
        moderate_ratio_threshold: float = 0.40,
        warning_ratio_threshold: float = 0.55,
        moderate_confidence_threshold: float = 0.55,
        consecutive_moderate_required: int = 3,
        alert_cooldown_seconds: float = 10.0
    ):
        """
        Initialise the DecisionAgent.

        Parameters
        ----------
        window_size:
            Number of recent predictions kept in memory.

        minimum_history:
            Minimum predictions required before issuing an alert.

        moderate_ratio_threshold:
            Required proportion of Moderate predictions in the window.

        warning_ratio_threshold:
            Required proportion of Mild-or-Moderate predictions.

        moderate_confidence_threshold:
            Minimum confidence for a Moderate prediction to count
            as a strong fatigue signal.

        consecutive_moderate_required:
            Consecutive Moderate predictions required for an
            immediate critical alert.

        alert_cooldown_seconds:
            Minimum time between repeated alert triggers.
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

        self.window_size = window_size
        self.minimum_history = minimum_history

        self.moderate_ratio_threshold = (
            moderate_ratio_threshold
        )

        self.warning_ratio_threshold = (
            warning_ratio_threshold
        )

        self.moderate_confidence_threshold = (
            moderate_confidence_threshold
        )

        self.consecutive_moderate_required = (
            consecutive_moderate_required
        )

        self.alert_cooldown_seconds = (
            alert_cooldown_seconds
        )

        self.history: Deque[Dict] = deque(
            maxlen=window_size
        )

        self.last_alert_time: Optional[float] = None

    def reset(self) -> None:
        """
        Clear all temporal state.
        """

        self.history.clear()
        self.last_alert_time = None

    def _count_consecutive_moderate(self) -> int:
        """
        Count Moderate predictions from the end of the history.
        """

        count = 0

        for item in reversed(self.history):

            is_strong_moderate = (
                item["predicted_class"]
                == self.MODERATE_CLASS
                and item["confidence"]
                >= self.moderate_confidence_threshold
            )

            if not is_strong_moderate:
                break

            count += 1

        return count

    def _cooldown_complete(self) -> bool:
        """
        Return True when another alert may be triggered.
        """

        if self.last_alert_time is None:
            return True

        elapsed = monotonic() - self.last_alert_time

        return elapsed >= self.alert_cooldown_seconds

    def update(
        self,
        prediction_result: Dict
    ) -> DecisionResult:
        """
        Add one prediction and calculate the current driver state.

        Parameters
        ----------
        prediction_result:
            Dictionary returned by Predictor.predict().

        Returns
        -------
        DecisionResult
            Temporal fatigue decision.
        """

        required_keys = {
            "predicted_class",
            "confidence"
        }

        missing_keys = required_keys.difference(
            prediction_result.keys()
        )

        if missing_keys:
            raise KeyError(
                "Prediction result is missing keys: "
                f"{sorted(missing_keys)}"
            )

        predicted_class = int(
            prediction_result["predicted_class"]
        )

        confidence = float(
            prediction_result["confidence"]
        )

        if predicted_class not in {
            self.ALERT_CLASS,
            self.MILD_CLASS,
            self.MODERATE_CLASS
        }:
            raise ValueError(
                f"Unknown predicted class: {predicted_class}"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Prediction confidence must be between 0 and 1."
            )

        self.history.append(
            {
                "predicted_class": predicted_class,
                "confidence": confidence
            }
        )

        history_size = len(self.history)

        moderate_count = sum(
            1
            for item in self.history
            if item["predicted_class"]
            == self.MODERATE_CLASS
        )

        mild_or_higher_count = sum(
            1
            for item in self.history
            if item["predicted_class"]
            in {
                self.MILD_CLASS,
                self.MODERATE_CLASS
            }
        )

        moderate_ratio = (
            moderate_count / history_size
        )

        mild_or_higher_ratio = (
            mild_or_higher_count / history_size
        )

        average_confidence = sum(
            item["confidence"]
            for item in self.history
        ) / history_size

        consecutive_moderate = (
            self._count_consecutive_moderate()
        )

        # Not enough temporal evidence yet.
        if history_size < self.minimum_history:

            return DecisionResult(
                state="Monitoring",
                alert_level="none",
                trigger_alert=False,
                reason=(
                    "Collecting more predictions before "
                    "making a temporal decision."
                ),
                moderate_ratio=moderate_ratio,
                mild_or_higher_ratio=mild_or_higher_ratio,
                average_confidence=average_confidence,
                consecutive_moderate=consecutive_moderate,
                history_size=history_size
            )

        critical_condition = (
            consecutive_moderate
            >= self.consecutive_moderate_required
        )

        sustained_moderate_condition = (
            moderate_ratio
            >= self.moderate_ratio_threshold
            and average_confidence
            >= self.moderate_confidence_threshold
        )

        warning_condition = (
            mild_or_higher_ratio
            >= self.warning_ratio_threshold
        )

        trigger_alert = False

        if critical_condition or sustained_moderate_condition:

            if self._cooldown_complete():

                trigger_alert = True
                self.last_alert_time = monotonic()

            reason = (
                "Sustained Moderate/Severe fatigue detected."
            )

            if critical_condition:
                reason = (
                    f"{consecutive_moderate} consecutive strong "
                    "Moderate/Severe predictions detected."
                )

            return DecisionResult(
                state="Moderate/Severe Fatigue",
                alert_level="critical",
                trigger_alert=trigger_alert,
                reason=reason,
                moderate_ratio=moderate_ratio,
                mild_or_higher_ratio=mild_or_higher_ratio,
                average_confidence=average_confidence,
                consecutive_moderate=consecutive_moderate,
                history_size=history_size
            )

        if warning_condition:

            return DecisionResult(
                state="Mild Fatigue",
                alert_level="warning",
                trigger_alert=False,
                reason=(
                    "The majority of recent predictions indicate "
                    "at least mild fatigue."
                ),
                moderate_ratio=moderate_ratio,
                mild_or_higher_ratio=mild_or_higher_ratio,
                average_confidence=average_confidence,
                consecutive_moderate=consecutive_moderate,
                history_size=history_size
            )

        return DecisionResult(
            state="Alert",
            alert_level="none",
            trigger_alert=False,
            reason=(
                "Recent predictions do not show sustained fatigue."
            ),
            moderate_ratio=moderate_ratio,
            mild_or_higher_ratio=mild_or_higher_ratio,
            average_confidence=average_confidence,
            consecutive_moderate=consecutive_moderate,
            history_size=history_size
        )