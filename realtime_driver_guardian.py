"""
Real-time DriverGuardianAI application.

Production candidate pipeline
-----------------------------
Webcam
    -> VisionAgent
    -> eight engineered features
    -> calibrated Histogram Gradient Boosting model
    -> BinaryDecisionAgent
    -> LoggingAgent
    -> real-time display

The previous neural-network real-time service remains available as an
experimental implementation. This script uses the calibrated classical
model selected through participant-aware evaluation.

Controls
--------
Q or Esc
    Close the application safely.

R
    Reset temporal decision history.

I
    Print current model and service information in the terminal.

Important
---------
The webcam continues to run at its natural frame rate, but model
inference runs at a controlled interval. This prevents consecutive
video frames from being interpreted as several independent seconds of
fatigue evidence.
"""

from time import monotonic
from typing import Any, Dict, Optional

import cv2

from src.binary_fatigue_service import BinaryFatigueService
from src.vision_agent import VisionAgent


# ============================================================
# APPLICATION SETTINGS
# ============================================================

WINDOW_NAME = "DriverGuardianAI"

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Two model predictions per second.
INFERENCE_INTERVAL_SECONDS = 0.5

MODEL_PATH = (
    "models/"
    "driver_guardian_calibrated_hgb.joblib"
)


# ============================================================
# COLOURS
#
# OpenCV uses BGR rather than RGB.
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
GREY = (180, 180, 180)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to float without breaking the display.
    """

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return float(default)


def draw_text_with_background(
    frame,
    text: str,
    position,
    font_scale: float = 0.60,
    text_colour=WHITE,
    background_colour=BLACK,
    thickness: int = 2,
    padding: int = 5,
):
    """
    Draw readable text with a filled background rectangle.
    """

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    top_left = (
        x - padding,
        y - text_height - padding,
    )

    bottom_right = (
        x + text_width + padding,
        y + baseline + padding,
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        background_colour,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (
            x,
            y,
        ),
        font,
        font_scale,
        text_colour,
        thickness,
        cv2.LINE_AA,
    )


def decision_colour(
    state: str,
    alert_level: str,
):
    """
    Select a display colour for the current temporal state.
    """

    if alert_level == "critical":
        return RED

    if alert_level == "warning":
        return ORANGE

    if state == "Alert":
        return GREEN

    if state == "Monitoring":
        return YELLOW

    return WHITE


def draw_probability_bar(
    frame,
    probability: float,
    threshold: float,
    position,
    width: int = 360,
    height: int = 22,
):
    """
    Draw Fatigue probability and the calibrated threshold.
    """

    x, y = position

    probability = max(
        0.0,
        min(
            1.0,
            safe_float(probability),
        ),
    )

    threshold = max(
        0.0,
        min(
            1.0,
            safe_float(threshold),
        ),
    )

    cv2.rectangle(
        frame,
        (
            x,
            y,
        ),
        (
            x + width,
            y + height,
        ),
        WHITE,
        2,
    )

    filled_width = int(
        width * probability
    )

    bar_colour = (
        RED
        if probability >= threshold
        else GREEN
    )

    if filled_width > 0:
        cv2.rectangle(
            frame,
            (
                x + 2,
                y + 2,
            ),
            (
                x + max(
                    2,
                    filled_width - 2,
                ),
                y + height - 2,
            ),
            bar_colour,
            -1,
        )

    threshold_x = (
        x + int(
            width * threshold
        )
    )

    cv2.line(
        frame,
        (
            threshold_x,
            y - 5,
        ),
        (
            threshold_x,
            y + height + 5,
        ),
        YELLOW,
        2,
    )

    label = (
        f"Fatigue probability: {probability:.1%} "
        f"| Threshold: {threshold:.0%}"
    )

    draw_text_with_background(
        frame,
        label,
        (
            x,
            y - 10,
        ),
        font_scale=0.50,
        text_colour=WHITE,
        background_colour=BLACK,
        thickness=1,
    )


def draw_waiting_message(
    frame,
):
    """
    Draw the state before the first model prediction.
    """

    draw_text_with_background(
        frame,
        "Waiting for first model prediction...",
        (
            15,
            270,
        ),
        font_scale=0.65,
        text_colour=YELLOW,
        background_colour=BLACK,
        thickness=2,
    )

    draw_text_with_background(
        frame,
        "Keep your face visible to the camera.",
        (
            15,
            305,
        ),
        font_scale=0.55,
        text_colour=WHITE,
        background_colour=BLACK,
        thickness=1,
    )


def draw_prediction_panel(
    frame,
    service_result: Optional[
        Dict[str, Any]
    ],
):
    """
    Draw model and temporal-decision outputs.
    """

    if service_result is None:
        draw_waiting_message(
            frame
        )

        return frame

    prediction = service_result.get(
        "prediction",
        {},
    )

    decision = service_result.get(
        "decision",
        {},
    )

    predicted_label = str(
        prediction.get(
            "prediction",
            "Unknown",
        )
    )

    fatigue_probability = safe_float(
        prediction.get(
            "fatigue_probability",
            0.0,
        )
    )

    alert_probability = safe_float(
        prediction.get(
            "alert_probability",
            0.0,
        )
    )

    threshold = safe_float(
        prediction.get(
            "threshold",
            0.89,
        ),
        default=0.89,
    )

    temporal_state = str(
        decision.get(
            "state",
            "Monitoring",
        )
    )

    alert_level = str(
        decision.get(
            "alert_level",
            "none",
        )
    )

    trigger_alert = bool(
        decision.get(
            "trigger_alert",
            False,
        )
    )

    fatigue_ratio = safe_float(
        decision.get(
            "fatigue_ratio",
            0.0,
        )
    )

    average_fatigue_probability = safe_float(
        decision.get(
            "average_fatigue_probability",
            0.0,
        )
    )

    consecutive_fatigue = int(
        safe_float(
            decision.get(
                "consecutive_fatigue",
                0,
            )
        )
    )

    history_size = int(
        safe_float(
            decision.get(
                "history_size",
                0,
            )
        )
    )

    reason = str(
        decision.get(
            "reason",
            "",
        )
    )

    state_colour = decision_colour(
        temporal_state,
        alert_level,
    )

    # --------------------------------------------------------
    # Main prediction
    # --------------------------------------------------------

    draw_text_with_background(
        frame,
        (
            f"Model: {predicted_label} "
            f"| Fatigue {fatigue_probability:.1%} "
            f"| Alert {alert_probability:.1%}"
        ),
        (
            15,
            260,
        ),
        font_scale=0.62,
        text_colour=state_colour,
        background_colour=BLACK,
        thickness=2,
    )

    draw_probability_bar(
        frame,
        probability=fatigue_probability,
        threshold=threshold,
        position=(
            15,
            300,
        ),
        width=390,
        height=22,
    )

    # --------------------------------------------------------
    # Temporal decision
    # --------------------------------------------------------

    draw_text_with_background(
        frame,
        f"Temporal state: {temporal_state}",
        (
            15,
            360,
        ),
        font_scale=0.65,
        text_colour=state_colour,
        background_colour=BLACK,
        thickness=2,
    )

    draw_text_with_background(
        frame,
        f"Alert level: {alert_level}",
        (
            15,
            395,
        ),
        font_scale=0.58,
        text_colour=state_colour,
        background_colour=BLACK,
        thickness=2,
    )

    draw_text_with_background(
        frame,
        (
            f"Recent Fatigue ratio: "
            f"{fatigue_ratio:.0%}"
        ),
        (
            15,
            430,
        ),
        font_scale=0.54,
        text_colour=WHITE,
        background_colour=BLACK,
        thickness=1,
    )

    draw_text_with_background(
        frame,
        (
            f"Average Fatigue probability: "
            f"{average_fatigue_probability:.1%}"
        ),
        (
            15,
            460,
        ),
        font_scale=0.54,
        text_colour=WHITE,
        background_colour=BLACK,
        thickness=1,
    )

    draw_text_with_background(
        frame,
        (
            f"Consecutive Fatigue: "
            f"{consecutive_fatigue} "
            f"| History: {history_size}"
        ),
        (
            15,
            490,
        ),
        font_scale=0.54,
        text_colour=WHITE,
        background_colour=BLACK,
        thickness=1,
    )

    # --------------------------------------------------------
    # Alert banner
    # --------------------------------------------------------

    if trigger_alert:
        banner_text = (
            "CRITICAL FATIGUE ALERT — "
            "TAKE A SAFE BREAK"
        )

        frame_width = frame.shape[1]

        cv2.rectangle(
            frame,
            (
                0,
                0,
            ),
            (
                frame_width,
                65,
            ),
            RED,
            -1,
        )

        cv2.putText(
            frame,
            banner_text,
            (
                20,
                43,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.88,
            WHITE,
            2,
            cv2.LINE_AA,
        )

    elif alert_level == "critical":
        draw_text_with_background(
            frame,
            "SUSTAINED FATIGUE DETECTED",
            (
                15,
                535,
            ),
            font_scale=0.70,
            text_colour=RED,
            background_colour=BLACK,
            thickness=2,
        )

    elif alert_level == "warning":
        draw_text_with_background(
            frame,
            "POSSIBLE FATIGUE — CONTINUE MONITORING",
            (
                15,
                535,
            ),
            font_scale=0.62,
            text_colour=ORANGE,
            background_colour=BLACK,
            thickness=2,
        )

    else:
        draw_text_with_background(
            frame,
            "No new alert",
            (
                15,
                535,
            ),
            font_scale=0.58,
            text_colour=GREEN,
            background_colour=BLACK,
            thickness=1,
        )

    # Keep the reason concise enough for the live window.
    if reason:
        maximum_reason_length = 95

        if len(reason) > maximum_reason_length:
            reason = (
                reason[
                    :maximum_reason_length - 3
                ]
                + "..."
            )

        draw_text_with_background(
            frame,
            f"Reason: {reason}",
            (
                15,
                570,
            ),
            font_scale=0.47,
            text_colour=GREY,
            background_colour=BLACK,
            thickness=1,
        )

    return frame


def draw_runtime_information(
    frame,
    fps: float,
):
    """
    Draw FPS, inference rate, and keyboard controls.
    """

    frame_width = frame.shape[1]

    inference_rate = (
        1.0 / INFERENCE_INTERVAL_SECONDS
    )

    right_x = max(
        10,
        frame_width - 255,
    )

    draw_text_with_background(
        frame,
        f"FPS: {fps:.1f}",
        (
            right_x,
            28,
        ),
        font_scale=0.52,
        text_colour=WHITE,
        background_colour=BLACK,
        thickness=1,
    )

    draw_text_with_background(
        frame,
        f"Inference: {inference_rate:.1f} Hz",
        (
            right_x,
            55,
        ),
        font_scale=0.52,
        text_colour=WHITE,
        background_colour=BLACK,
        thickness=1,
    )

    draw_text_with_background(
        frame,
        "Q/Esc: quit | R: reset | I: info",
        (
            15,
            frame.shape[0] - 18,
        ),
        font_scale=0.48,
        text_colour=WHITE,
        background_colour=BLACK,
        thickness=1,
    )


# ============================================================
# SERVICE CREATION
# ============================================================

def create_vision_agent() -> VisionAgent:
    """
    Create the real-time feature-extraction agent.
    """

    return VisionAgent(
        max_num_faces=1,
        max_num_hands=2,
        low_light_threshold=70.0,
        blink_ear_threshold=0.65,
        minimum_closed_frames=2,
        blink_window_seconds=60.0,
    )


def create_fatigue_service() -> BinaryFatigueService:
    """
    Create the calibrated binary fatigue service.
    """

    return BinaryFatigueService(
        model_path=MODEL_PATH,
        threshold=None,
        enable_logging=True,
        log_directory="logs",
        window_size=12,
        minimum_history=5,
        warning_ratio_threshold=0.50,
        critical_ratio_threshold=0.70,
        consecutive_fatigue_required=3,
        alert_cooldown_seconds=10.0,
    )


def print_service_information(
    fatigue_service: BinaryFatigueService,
):
    """
    Print model and decision configuration.
    """

    information = (
        fatigue_service.model_information()
    )

    print("\n" + "-" * 72)
    print("DriverGuardianAI Service Information")
    print("-" * 72)

    predictor_information = information[
        "predictor"
    ]

    print(
        "Model type: "
        f"{predictor_information['model_type']}"
    )

    print(
        "Model path: "
        f"{predictor_information['model_path']}"
    )

    print(
        "Fatigue threshold: "
        f"{predictor_information['fatigue_threshold']:.2f}"
    )

    print(
        "Features: "
        f"{predictor_information['feature_columns']}"
    )

    print(
        "Training participants: "
        f"{predictor_information['training_participants']}"
    )

    print(
        "Calibration participants: "
        f"{predictor_information['calibration_participants']}"
    )

    print(
        "Test participants: "
        f"{predictor_information['test_participants']}"
    )

    print(
        "Decision configuration: "
        f"{information['decision_agent']}"
    )

    print(
        "Logging enabled: "
        f"{information['logging_enabled']}"
    )

    print("-" * 72)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """
    Start the real-time calibrated DriverGuardianAI application.
    """

    print("=" * 72)
    print("DriverGuardianAI Real-Time Application")
    print("Calibrated Histogram Gradient Boosting")
    print("=" * 72)

    print("Press Q or Esc to quit.")
    print("Press R to reset temporal memory.")
    print("Press I to print model information.")

    print(
        "Inference interval: "
        f"{INFERENCE_INTERVAL_SECONDS:.1f} seconds"
    )

    vision_agent = create_vision_agent()

    fatigue_service = create_fatigue_service()

    print_service_information(
        fatigue_service
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():
        vision_agent.close()

        raise RuntimeError(
            "Could not open the webcam. Check whether another "
            "application is currently using it."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    latest_service_result: Optional[
        Dict[str, Any]
    ] = None

    last_inference_time = 0.0

    previous_frame_time = monotonic()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "Unable to read a webcam frame."
                )

                break

            frame = cv2.flip(
                frame,
                1,
            )

            vision_result = (
                vision_agent.process_frame(
                    frame
                )
            )

            display_frame = (
                vision_result.annotated_frame
            )

            current_time = monotonic()

            should_run_inference = (
                current_time
                - last_inference_time
                >= INFERENCE_INTERVAL_SECONDS
            )

            if (
                should_run_inference
                and vision_result.face_detected
                and vision_result.features is not None
            ):
                try:
                    latest_service_result = (
                        fatigue_service.process(
                            vision_result.features
                        )
                    )

                except Exception as error:
                    print(
                        "Prediction error: "
                        f"{error}"
                    )

                last_inference_time = (
                    current_time
                )

            display_frame = draw_prediction_panel(
                display_frame,
                latest_service_result,
            )

            frame_elapsed = (
                current_time
                - previous_frame_time
            )

            fps = (
                1.0 / frame_elapsed
                if frame_elapsed > 0
                else 0.0
            )

            previous_frame_time = current_time

            draw_runtime_information(
                display_frame,
                fps,
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )

            key = cv2.waitKey(
                1
            ) & 0xFF

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
                fatigue_service.reset_session()

                latest_service_result = None

                last_inference_time = 0.0

                print(
                    "Temporal decision history reset."
                )

            if key in {
                ord("i"),
                ord("I"),
            }:
                print_service_information(
                    fatigue_service
                )

    except KeyboardInterrupt:
        print(
            "\nKeyboard interruption received."
        )

    finally:
        camera.release()

        vision_agent.close()

        cv2.destroyAllWindows()

        print(
            "DriverGuardianAI stopped safely."
        )


if __name__ == "__main__":
    main()