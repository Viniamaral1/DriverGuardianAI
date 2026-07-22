"""
DriverGuardianAI fatigue inference service.

This service connects:

- Predictor
- DecisionAgent
- LoggingAgent

It provides one reusable entry point for:

- Real-time webcam inference
- FastAPI
- Dashboard applications
- Future LangGraph orchestration
"""

from typing import Dict, Optional

from src.predict import Predictor
from src.decision_agent import DecisionAgent
from src.logging_agent import LoggingAgent


class FatigueService:
    """
    End-to-end fatigue inference service.
    """

    def __init__(
        self,
        model_path="models/driver_guardian_best.pth",
        preprocessing_path="models/preprocessing.pkl",
        hidden_dims=None,
        dropout=0.30,
        num_classes=3,
        device=None,
        enable_logging=True,
        log_directory="logs"
    ):
        """
        Initialise all production inference components.
        """

        if hidden_dims is None:
            hidden_dims = [
                256,
                128,
                64
            ]

        self.predictor = Predictor(
            model_path=model_path,
            preprocessing_path=preprocessing_path,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
            device=device
        )

        self.decision_agent = DecisionAgent(
            window_size=15,
            minimum_history=5,
            moderate_ratio_threshold=0.40,
            warning_ratio_threshold=0.55,
            moderate_confidence_threshold=0.55,
            consecutive_moderate_required=3,
            alert_cooldown_seconds=10.0
        )

        self.logging_agent: Optional[LoggingAgent] = None

        if enable_logging:
            self.logging_agent = LoggingAgent(
                log_directory=log_directory
            )

    def reset_session(self):
        """
        Reset temporal decision memory.
        """

        self.decision_agent.reset()

    def process(self, features: Dict) -> Dict:
        """
        Process one feature dictionary through the complete system.

        Parameters
        ----------
        features:
            Dictionary containing the eight model features.

        Returns
        -------
        dict
            Model prediction, temporal decision and log information.
        """

        prediction = self.predictor.predict(
            features
        )

        decision = self.decision_agent.update(
            prediction
        )

        log_path = None

        if self.logging_agent is not None:
            log_path = self.logging_agent.log(
                features=features,
                prediction_result=prediction,
                decision_result=decision
            )

        return {
            "prediction": prediction,
            "decision": decision.to_dict(),
            "log_path": log_path
        }