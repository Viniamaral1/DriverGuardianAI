"""
DriverGuardianAI V2
Core-Behaviour Real-Time Application
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional

import cv2
import joblib
import numpy as np
import pandas as pd

from realtime_driver_guardian_v2 import (
    ALERT_COOLDOWN_SECONDS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CONSECUTIVE_FATIGUE_REQUIRED,
    HAND_LOWER_FRAME_RATIO,
    INFERENCE_INTERVAL_SECONDS,
    LOW_LIGHT_THRESHOLD,
    MINIMUM_CLOSED_FRAMES,
    MINIMUM_HISTORY,
    RAW_BLINK_EAR_THRESHOLD,
    TEMPORAL_WINDOW_SIZE,
    WARNING_FATIGUE_RATIO,
    CRITICAL_FATIGUE_RATIO,
    PredictionResult,
    RuntimeLoggerV2,
    TemporalDecision,
    TemporalDecisionAgentV2,
    annotate_runtime,
)
from src.v2.vision_agent_v2 import VisionAgentV2

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v2"
    / "ablation"
    / "driver_guardian_core_behaviour.joblib"
)

LOG_DIRECTORY = (
    PROJECT_ROOT
    / "logs"
    / "v2"
    / "core_behaviour"
)

WINDOW_NAME = "DriverGuardianAI V2 - Core Behaviour"
DEFAULT_CONDITION = "none"


class AblationPredictorV2:
    SUPPORTED_RUNTIME_FEATURES = {
        "ear",
        "yawn_score",
        "head_tilt",
        "hands_detected",
        "condition",
        "low_light",
        "face_confidence",
        "blink_count",
    }

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Ablation model was not found: {model_path}"
            )

        self.model_path = model_path
        self.bundle = joblib.load(model_path)

        required_keys = {
            "pipeline",
            "fatigue_threshold",
            "feature_columns",
            "class_names",
        }

        missing = required_keys.difference(
            self.bundle.keys()
        )

        if missing:
            raise KeyError(
                "Model bundle is missing keys: "
                f"{sorted(missing)}"
            )

        self.pipeline = self.bundle["pipeline"]

        self.fatigue_threshold = float(
            self.bundle["fatigue_threshold"]
        )

        self.feature_columns = list(
            self.bundle["feature_columns"]
        )

        self.class_names = list(
            self.bundle["class_names"]
        )

        unsupported = set(
            self.feature_columns
        ).difference(
            self.SUPPORTED_RUNTIME_FEATURES
        )

        if unsupported:
            raise ValueError(
                "Model requests unsupported runtime features: "
                f"{sorted(unsupported)}"
            )

    @staticmethod
    def _convert_feature(
        feature: str,
        value: Any,
    ) -> Any:
        if feature == "condition":
            return str(value).strip().lower()

        if feature == "low_light":
            return float(bool(value))

        return float(value)

    def predict(
        self,
        all_features: Dict[str, Any],
    ) -> PredictionResult:
        missing = [
            feature
            for feature in self.feature_columns
            if feature not in all_features
        ]

        if missing:
            raise ValueError(
                "Vision output is missing model features: "
                f"{missing}"
            )

        prepared = {
            feature: self._convert_feature(
                feature,
                all_features[feature],
            )
            for feature in self.feature_columns
        }

        for feature, value in prepared.items():
            if (
                feature != "condition"
                and not np.isfinite(float(value))
            ):
                raise ValueError(
                    f"{feature} is not finite."
                )

        dataframe = pd.DataFrame(
            [prepared],
            columns=self.feature_columns,
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

        predicted_class = int(
            fatigue_probability
            >= self.fatigue_threshold
        )

        predicted_label = (
            "Fatigue"
            if predicted_class == 1
            else "Alert"
        )

        confidence = (
            fatigue_probability
            if predicted_class == 1
            else alert_probability
        )

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            fatigue_probability=fatigue_probability,
            alert_probability=alert_probability,
            threshold=self.fatigue_threshold,
            threshold_margin=float(
                abs(
                    fatigue_probability
                    - self.fatigue_threshold
                )
            ),
            confidence=float(confidence),
        )

    def information(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "variant_name": self.bundle.get(
                "variant_name",
                "unknown",
            ),
            "model_name": self.bundle.get(
                "model_name",
                "unknown",
            ),
            "feature_columns": self.feature_columns,
            "fatigue_threshold": self.fatigue_threshold,
            "class_names": self.class_names,
            "saved_test_metrics": self.bundle.get(
                "test_metrics",
                {},
            ),
            "saved_live_metrics": self.bundle.get(
                "live_metrics",
                {},
            ),
        }


def print_information(
    predictor: AblationPredictorV2,
    decision_agent: TemporalDecisionAgentV2,
    vision_agent: VisionAgentV2,
    logger: RuntimeLoggerV2,
) -> None:
    print("\n" + "-" * 72)
    print("DriverGuardianAI V2 Core Behaviour")
    print("-" * 72)

    print("\nModel:")
    for key, value in predictor.information().items():
        print(f"{key}: {value}")

    print("\nTemporal decision configuration:")
    for key, value in decision_agent.information().items():
        print(f"{key}: {value}")

    print("\nVision:")
    print(f"condition: {vision_agent.condition}")
    print(
        "raw_blink_ear_threshold: "
        f"{vision_agent.raw_blink_ear_threshold}"
    )
    print(
        "low_light_threshold: "
        f"{vision_agent.low_light_threshold}"
    )

    print("\nLogging:")
    print(f"enabled: {logger.enabled}")
    print(f"path: {logger.filepath}")
    print("-" * 72)


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Core-Behaviour Real-Time Application")
    print("=" * 72)

    print("Q or Esc: quit")
    print("R: reset blink and temporal state")
    print("I: print runtime information")
    print("0: condition none")
    print("1: condition glasses")
    print("2: condition hat")
    print("3: condition dark")
    print("L: toggle logging")

    predictor = AblationPredictorV2(
        MODEL_PATH
    )

    decision_agent = TemporalDecisionAgentV2(
        window_size=TEMPORAL_WINDOW_SIZE,
        minimum_history=MINIMUM_HISTORY,
        warning_ratio_threshold=WARNING_FATIGUE_RATIO,
        critical_ratio_threshold=CRITICAL_FATIGUE_RATIO,
        consecutive_fatigue_required=(
            CONSECUTIVE_FATIGUE_REQUIRED
        ),
        alert_cooldown_seconds=(
            ALERT_COOLDOWN_SECONDS
        ),
    )

    vision_agent = VisionAgentV2(
        low_light_threshold=LOW_LIGHT_THRESHOLD,
        raw_blink_ear_threshold=(
            RAW_BLINK_EAR_THRESHOLD
        ),
        minimum_closed_frames=(
            MINIMUM_CLOSED_FRAMES
        ),
        hand_lower_frame_ratio=(
            HAND_LOWER_FRAME_RATIO
        ),
        condition=DEFAULT_CONDITION,
        automatically_mark_dark_condition=True,
    )

    logger = RuntimeLoggerV2(
        LOG_DIRECTORY
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():
        logger.close()
        vision_agent.close()

        raise RuntimeError(
            "Could not open webcam."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    latest_prediction: Optional[
        PredictionResult
    ] = None

    latest_decision: Optional[
        TemporalDecision
    ] = None

    last_inference_time = 0.0

    frame_counter = 0
    last_fps_time = monotonic()
    current_fps = 0.0

    application_start = monotonic()

    print("\nCore model loaded successfully.")
    print(f"Features: {predictor.feature_columns}")
    print(
        "Fatigue threshold: "
        f"{predictor.fatigue_threshold:.2f}"
    )
    print(f"Logging to: {logger.filepath}")

    print_information(
        predictor,
        decision_agent,
        vision_agent,
        logger,
    )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "Could not read webcam frame."
                )
                break

            frame_counter += 1
            current_time = monotonic()

            fps_elapsed = (
                current_time
                - last_fps_time
            )

            if fps_elapsed >= 1.0:
                current_fps = (
                    frame_counter
                    / fps_elapsed
                )

                frame_counter = 0
                last_fps_time = current_time

            vision_result = (
                vision_agent.process_frame(
                    frame
                )
            )

            if (
                vision_result.face_detected
                and vision_result.features
                is not None
                and current_time
                - last_inference_time
                >= INFERENCE_INTERVAL_SECONDS
            ):
                latest_prediction = (
                    predictor.predict(
                        vision_result.features
                    )
                )

                latest_decision = (
                    decision_agent.update(
                        latest_prediction
                    )
                )

                logger.log(
                    vision_result.features,
                    latest_prediction,
                    latest_decision,
                )

                last_inference_time = (
                    current_time
                )

            displayed_frame = annotate_runtime(
                vision_result.annotated_frame,
                latest_prediction,
                latest_decision,
                (
                    1.0
                    / INFERENCE_INTERVAL_SECONDS
                ),
                current_fps,
                logger.enabled,
            )

            cv2.imshow(
                WINDOW_NAME,
                displayed_frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in {
                ord("q"),
                ord("Q"),
                27,
            }:
                break

            if key in {
                ord("r"),
                ord("R"),
            }:
                decision_agent.reset()
                vision_agent.reset_temporal_state()

                latest_prediction = None
                latest_decision = None
                last_inference_time = 0.0

                print(
                    "Blink count and temporal history reset."
                )

            elif key in {
                ord("i"),
                ord("I"),
            }:
                print_information(
                    predictor,
                    decision_agent,
                    vision_agent,
                    logger,
                )

            elif key == ord("0"):
                vision_agent.set_condition(
                    "none"
                )
                print(
                    "Condition set to none."
                )

            elif key == ord("1"):
                vision_agent.set_condition(
                    "glasses"
                )
                print(
                    "Condition set to glasses."
                )

            elif key == ord("2"):
                vision_agent.set_condition(
                    "hat"
                )
                print(
                    "Condition set to hat."
                )

            elif key == ord("3"):
                vision_agent.set_condition(
                    "dark"
                )
                print(
                    "Condition set to dark."
                )

            elif key in {
                ord("l"),
                ord("L"),
            }:
                logger.set_enabled(
                    not logger.enabled
                )

                print(
                    "Logging "
                    + (
                        "enabled."
                        if logger.enabled
                        else "disabled."
                    )
                )

    except KeyboardInterrupt:
        print(
            "\nKeyboard interruption received."
        )

    finally:
        camera.release()
        vision_agent.close()
        logger.close()
        cv2.destroyAllWindows()

        print(
            "\nCore-behaviour application stopped safely."
        )

        print(
            "Runtime: "
            f"{monotonic() - application_start:.1f} seconds"
        )

        print(
            f"Log file: {logger.filepath}"
        )


if __name__ == "__main__":
    main()