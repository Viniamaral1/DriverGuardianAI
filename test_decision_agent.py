"""
Test the DriverGuardianAI temporal DecisionAgent.

Run in Jupyter with:

    %run test_decision_agent.py
"""

from src.decision_agent import DecisionAgent


def create_prediction(
    predicted_class,
    confidence
):
    """
    Create a prediction in the same format returned
    by Predictor.predict().
    """

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

    simulated_predictions = [
        create_prediction(0, 0.72),
        create_prediction(0, 0.68),
        create_prediction(1, 0.61),
        create_prediction(1, 0.65),
        create_prediction(1, 0.70),
        create_prediction(2, 0.58),
        create_prediction(2, 0.66),
        create_prediction(2, 0.73),
        create_prediction(2, 0.81)
    ]

    print("=" * 70)
    print("DriverGuardianAI Decision Agent Test")
    print("=" * 70)

    for index, prediction in enumerate(
        simulated_predictions,
        start=1
    ):

        decision = decision_agent.update(
            prediction
        )

        print(
            f"\nPrediction {index}: "
            f"{prediction['prediction']} "
            f"({prediction['confidence']:.0%})"
        )

        print(
            f"Temporal state      : {decision.state}"
        )

        print(
            f"Alert level         : {decision.alert_level}"
        )

        print(
            f"Trigger alert       : {decision.trigger_alert}"
        )

        print(
            f"Moderate ratio      : "
            f"{decision.moderate_ratio:.0%}"
        )

        print(
            f"Mild-or-higher ratio: "
            f"{decision.mild_or_higher_ratio:.0%}"
        )

        print(
            f"Consecutive moderate: "
            f"{decision.consecutive_moderate}"
        )

        print(
            f"Reason              : {decision.reason}"
        )

    print("\n" + "=" * 70)
    print("Decision Agent test completed.")
    print("=" * 70)


if __name__ == "__main__":

    main()