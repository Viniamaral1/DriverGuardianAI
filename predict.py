"""
DriverGuardianAI Prediction Script

Loads the trained model and preprocessing artifacts,
runs inference on a sample feature vector,
and prints the prediction results.
"""

from src.predict import Predictor


def main():

    predictor = Predictor(
        model_path="models/best_fatigue_model_final_aug.pth",
        scaler_path="artifacts/scaler.pkl",
        label_encoder_path="artifacts/label_encoder.pkl"
    )

    # --------------------------------------------------
    # Example feature vector
    #
    # Replace these values with real MediaPipe features
    # later.
    # --------------------------------------------------

    sample = [

        0.24,   # EAR

        0.08,   # Yawn score

        2.5,    # Head tilt

        2,      # Hands detected

        0,      # Weather / condition

        0,      # Low light

        0.95,   # Face confidence

        6,      # Blink count

        0.18    # Fatigue score

    ]

    result = predictor.predict(sample)

    print("\n" + "=" * 50)
    print("DriverGuardianAI Prediction")
    print("=" * 50)

    print(f"\nPredicted Class : {result['prediction']}")

    print(f"Confidence      : {result['confidence']:.2%}")

    print("\nProbabilities")

    for label, probability in result["probabilities"].items():

        print(f"{label:<12}: {probability:.4f}")

    if result["confidence"] >= 0.80:

        print("\n⚠ Driver Alert Recommended")

    else:

        print("\n✓ Driver appears safe")


if __name__ == "__main__":

    main()