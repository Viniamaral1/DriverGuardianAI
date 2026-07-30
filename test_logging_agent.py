"""
Test the DriverGuardianAI LoggingAgent.

Run in Jupyter with:

    %run test_logging_agent.py
"""

from src.decision_agent import DecisionAgent
from src.logging_agent import LoggingAgent


def create_prediction(
    predicted_class,
    confidence
):
    labels = {
        0: "Alert",
        1: "Mild Fatigue",
        2: "Moderate/Severe Fatigue"
    }

    return {
        "prediction": labels[predicted_class],
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {}
    }


def main():

    decision_agent = DecisionAgent(
        window_size=10,
        minimum_history=5,
        moderate_ratio_threshold=0.40,
        warning_ratio_threshold=0.55,
        moderate_confidence_threshold=0.55,
        consecutive_moderate_required=3,
        alert_cooldown_seconds=10.0
    )

    logging_agent = LoggingAgent(
        log_directory="logs"
    )

    features = {
        "ear": 0.18,
        "yawn_score": 0.72,
        "head_tilt": 14.0,
        "hands_detected": 1,
        "condition": "dark",
        "low_light": True,
        "face_confidence": 0.86,
        "blink_count": 18
    }

    simulated_predictions = [
        create_prediction(0, 0.74),
        create_prediction(1, 0.62),
        create_prediction(1, 0.68),
        create_prediction(2, 0.59),
        create_prediction(2, 0.66),
        create_prediction(2, 0.76)
    ]

    print("=" * 70)
    print("DriverGuardianAI Logging Agent Test")
    print("=" * 70)

    for index, prediction in enumerate(
        simulated_predictions,
        start=1
    ):

        decision = decision_agent.update(
            prediction
        )

        log_path = logging_agent.log(
            features=features,
            prediction_result=prediction,
            decision_result=decision
        )

        print(
            f"Logged event {index}: "
            f"{prediction['prediction']} -> "
            f"{decision.state}"
        )

    print("\nLog file created at:")

    print(
        log_path
    )

    print("\n" + "=" * 70)
    print("Logging Agent test completed.")
    print("=" * 70)


if __name__ == "__main__":

    main()