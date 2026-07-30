"""
DriverGuardianAI V3
Professional real-time fatigue-monitoring dashboard.

Controls
--------
Q / Esc : Quit
R       : Reset temporal state
L       : Start/stop CSV logging
I       : Print current runtime information

Examples
--------
python realtime_driver_guardian_v3.py
python realtime_driver_guardian_v3.py --camera 0
python realtime_driver_guardian_v3.py --video example.mp4
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v2"
    / "ablation"
    / "driver_guardian_core_behaviour.joblib"
)

LOG_DIRECTORY = PROJECT_ROOT / "logs" / "v3"


# ---------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


# ---------------------------------------------------------------------
# Dashboard colours: BGR format
# ---------------------------------------------------------------------

BACKGROUND = (18, 23, 36)
PANEL = (29, 36, 52)
PANEL_ALT = (38, 46, 65)
BORDER = (74, 87, 110)

WHITE = (245, 245, 245)
MUTED = (165, 175, 194)

CYAN = (235, 190, 55)
GREEN = (95, 215, 135)
AMBER = (65, 185, 245)
RED = (65, 75, 245)


# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------

def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.55,
    colour: tuple[int, int, int] = WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    colour: tuple[int, int, int] = PANEL,
) -> None:
    cv2.rectangle(image, top_left, bottom_right, colour, -1)
    cv2.rectangle(image, top_left, bottom_right, BORDER, 1)


def state_colour(state: str) -> tuple[int, int, int]:
    if state == "CRITICAL":
        return RED

    if state == "WARNING":
        return AMBER

    if state == "MONITORING":
        return GREEN

    return MUTED


def draw_metric_card(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    title: str,
    value: str,
    subtitle: str,
) -> None:
    draw_panel(image, (x, y), (x + width, y + 88), PANEL_ALT)

    draw_text(
        image,
        title.upper(),
        (x + 14, y + 24),
        scale=0.40,
        colour=MUTED,
    )

    draw_text(
        image,
        value,
        (x + 14, y + 55),
        scale=0.78,
        colour=WHITE,
        thickness=2,
    )

    draw_text(
        image,
        subtitle,
        (x + 14, y + 77),
        scale=0.34,
        colour=MUTED,
    )


def draw_probability_gauge(
    image: np.ndarray,
    centre: tuple[int, int],
    radius: int,
    probability: float,
) -> None:
    probability = float(np.clip(probability, 0.0, 1.0))

    start_angle = 145
    end_angle = 395

    cv2.ellipse(
        image,
        centre,
        (radius, radius),
        0,
        start_angle,
        end_angle,
        BORDER,
        14,
        cv2.LINE_AA,
    )

    if probability < 0.55:
        colour = GREEN
    elif probability < 0.80:
        colour = AMBER
    else:
        colour = RED

    current_angle = start_angle + int(
        probability * (end_angle - start_angle)
    )

    cv2.ellipse(
        image,
        centre,
        (radius, radius),
        0,
        start_angle,
        current_angle,
        colour,
        14,
        cv2.LINE_AA,
    )

    percentage = f"{probability * 100:.0f}%"

    text_size = cv2.getTextSize(
        percentage,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        2,
    )[0]

    draw_text(
        image,
        percentage,
        (
            centre[0] - text_size[0] // 2,
            centre[1] + 8,
        ),
        scale=1.1,
        colour=WHITE,
        thickness=2,
    )

    draw_text(
        image,
        "FATIGUE RISK",
        (centre[0] - 59, centre[1] + 38),
        scale=0.40,
        colour=MUTED,
    )


def draw_timeline(
    image: np.ndarray,
    values: deque[float],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    draw_panel(image, (x, y), (x + width, y + height), PANEL)

    draw_text(
        image,
        "FATIGUE PROBABILITY — RECENT HISTORY",
        (x + 18, y + 27),
        scale=0.43,
        colour=WHITE,
    )

    plot_x = x + 18
    plot_y = y + 42
    plot_width = width - 36
    plot_height = height - 60

    cv2.rectangle(
        image,
        (plot_x, plot_y),
        (plot_x + plot_width, plot_y + plot_height),
        PANEL_ALT,
        -1,
    )

    warning_y = plot_y + plot_height - int(plot_height * 0.65)
    critical_y = plot_y + plot_height - int(plot_height * 0.82)

    cv2.line(
        image,
        (plot_x, warning_y),
        (plot_x + plot_width, warning_y),
        AMBER,
        1,
    )

    cv2.line(
        image,
        (plot_x, critical_y),
        (plot_x + plot_width, critical_y),
        RED,
        1,
    )

    history = list(values)

    if len(history) < 2:
        return

    points: list[tuple[int, int]] = []

    for index, value in enumerate(history):
        px = plot_x + int(
            index * plot_width / max(1, len(history) - 1)
        )

        py = (
            plot_y
            + plot_height
            - int(np.clip(value, 0.0, 1.0) * plot_height)
        )

        points.append((px, py))

    cv2.polylines(
        image,
        [np.asarray(points, dtype=np.int32)],
        False,
        CYAN,
        2,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def landmark_point(
    landmarks: list[Any],
    index: int,
    width: int,
    height: int,
) -> np.ndarray:
    point = landmarks[index]

    return np.array(
        [
            point.x * width,
            point.y * height,
        ],
        dtype=np.float32,
    )


def calculate_ear(
    landmarks: list[Any],
    eye_indices: list[int],
    width: int,
    height: int,
) -> float:
    points = [
        landmark_point(landmarks, index, width, height)
        for index in eye_indices
    ]

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal <= 1e-6:
        return 0.0

    return float(
        (vertical_1 + vertical_2)
        / (2.0 * horizontal)
    )


def extract_features(
    frame: np.ndarray,
    face_mesh: Any,
) -> tuple[dict[str, float], Any | None]:
    frame_height, frame_width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    result = face_mesh.process(rgb_frame)

    if not result.multi_face_landmarks:
        return {
            "ear": 0.0,
            "yawn_score": 0.0,
            "head_tilt": 0.0,
            "face_detected": 0.0,
        }, None

    face_landmarks = result.multi_face_landmarks[0]
    landmarks = face_landmarks.landmark

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE,
        frame_width,
        frame_height,
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE,
        frame_width,
        frame_height,
    )

    ear = (left_ear + right_ear) / 2.0

    upper_lip = landmark_point(
        landmarks,
        UPPER_LIP,
        frame_width,
        frame_height,
    )

    lower_lip = landmark_point(
        landmarks,
        LOWER_LIP,
        frame_width,
        frame_height,
    )

    left_mouth = landmark_point(
        landmarks,
        LEFT_MOUTH,
        frame_width,
        frame_height,
    )

    right_mouth = landmark_point(
        landmarks,
        RIGHT_MOUTH,
        frame_width,
        frame_height,
    )

    mouth_vertical = np.linalg.norm(
        upper_lip - lower_lip
    )

    mouth_horizontal = np.linalg.norm(
        left_mouth - right_mouth
    )

    if mouth_horizontal <= 1e-6:
        yawn_score = 0.0
    else:
        yawn_score = float(
            mouth_vertical / mouth_horizontal
        )

    left_eye_outer = landmark_point(
        landmarks,
        LEFT_EYE_OUTER,
        frame_width,
        frame_height,
    )

    right_eye_outer = landmark_point(
        landmarks,
        RIGHT_EYE_OUTER,
        frame_width,
        frame_height,
    )

    eye_difference = (
        right_eye_outer - left_eye_outer
    )

    head_tilt = abs(
        math.degrees(
            math.atan2(
                float(eye_difference[1]),
                float(eye_difference[0]),
            )
        )
    )

    features = {
        "ear": float(np.clip(ear, 0.0, 1.0)),
        "yawn_score": float(
            np.clip(yawn_score, 0.0, 2.0)
        ),
        "head_tilt": float(
            np.clip(head_tilt, 0.0, 90.0)
        ),
        "face_detected": 1.0,
    }

    return features, face_landmarks


# ---------------------------------------------------------------------
# Model loading and prediction
# ---------------------------------------------------------------------

def load_model(
    model_path: Path,
) -> tuple[Any | None, float, str | None]:
    if not model_path.exists():
        return (
            None,
            0.64,
            f"Model not found: {model_path}",
        )

    try:
        bundle = joblib.load(model_path)

        threshold = 0.64
        estimator = bundle

        if isinstance(bundle, dict):
            estimator = (
                bundle.get("model")
                or bundle.get("pipeline")
                or bundle.get("estimator")
                or bundle.get("classifier")
            )

            threshold = float(
                bundle.get(
                    "threshold",
                    bundle.get(
                        "selected_threshold",
                        bundle.get(
                            "fatigue_threshold",
                            threshold,
                        ),
                    ),
                )
            )

        if estimator is None:
            return (
                None,
                threshold,
                "No model or pipeline was found in the bundle.",
            )

        if not hasattr(estimator, "predict_proba"):
            return (
                None,
                threshold,
                "The loaded model does not implement predict_proba().",
            )

        return estimator, threshold, None

    except Exception as error:
        return (
            None,
            0.64,
            f"{type(error).__name__}: {error}",
        )


def heuristic_probability(
    features: dict[str, float],
) -> float:
    """
    Used only if the trained model cannot load.

    This allows the V3 dashboard itself to be tested.
    It is not a replacement for the trained model.
    """

    ear_risk = np.clip(
        (0.27 - features["ear"]) / 0.12,
        0.0,
        1.0,
    )

    yawn_risk = np.clip(
        features["yawn_score"] / 0.45,
        0.0,
        1.0,
    )

    tilt_risk = np.clip(
        features["head_tilt"] / 25.0,
        0.0,
        1.0,
    )

    probability = (
        0.68 * ear_risk
        + 0.22 * yawn_risk
        + 0.10 * tilt_risk
    )

    return float(
        np.clip(probability, 0.0, 1.0)
    )


def predict_probability(
    estimator: Any | None,
    features: dict[str, float],
) -> float:
    if estimator is None:
        return heuristic_probability(features)

    model_input = pd.DataFrame(
        [
            {
                "ear": features["ear"],
                "yawn_score": features["yawn_score"],
                "head_tilt": features["head_tilt"],
            }
        ]
    )

    probabilities = estimator.predict_proba(
        model_input
    )

    return float(
        np.clip(probabilities[0, 1], 0.0, 1.0)
    )


# ---------------------------------------------------------------------
# Temporal state engine
# ---------------------------------------------------------------------

class TemporalStateEngine:
    def __init__(self) -> None:
        self.probability_window: deque[float] = deque(
            maxlen=12
        )

        self.state = "MONITORING"

        self.warning_counter = 0
        self.critical_counter = 0
        self.release_counter = 0

        self.alert_count = 0

    def reset(self) -> None:
        self.probability_window.clear()

        self.state = "MONITORING"

        self.warning_counter = 0
        self.critical_counter = 0
        self.release_counter = 0

        self.alert_count = 0

    def update(
        self,
        probability: float,
        face_detected: bool,
    ) -> tuple[str, float]:
        if not face_detected:
            self.state = "NO FACE"

            self.warning_counter = 0
            self.critical_counter = 0
            self.release_counter = 0

            return self.state, probability

        self.probability_window.append(
            probability
        )

        smoothed_probability = float(
            np.mean(self.probability_window)
        )

        previous_state = self.state

        if smoothed_probability >= 0.82:
            self.critical_counter += 1
            self.warning_counter += 1
            self.release_counter = 0

        elif smoothed_probability >= 0.65:
            self.warning_counter += 1
            self.critical_counter = 0
            self.release_counter = 0

        elif smoothed_probability <= 0.45:
            self.release_counter += 1
            self.warning_counter = 0
            self.critical_counter = 0

        else:
            self.release_counter = 0
            self.critical_counter = 0

        if self.critical_counter >= 12:
            self.state = "CRITICAL"

        elif self.warning_counter >= 6:
            self.state = "WARNING"

        elif self.release_counter >= 8:
            self.state = "MONITORING"

        elif self.state == "NO FACE":
            self.state = "MONITORING"

        if (
            previous_state != "CRITICAL"
            and self.state == "CRITICAL"
        ):
            self.alert_count += 1

        return self.state, smoothed_probability


# ---------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------

class SessionLogger:
    def __init__(self) -> None:
        self.enabled = False
        self.file = None
        self.writer = None
        self.path: Path | None = None

    def start(self) -> None:
        LOG_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.path = (
            LOG_DIRECTORY
            / f"driver_guardian_v3_{timestamp}.csv"
        )

        self.file = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "timestamp",
                "elapsed_seconds",
                "ear",
                "yawn_score",
                "head_tilt",
                "fatigue_probability",
                "smoothed_probability",
                "temporal_state",
            ],
        )

        self.writer.writeheader()

        self.enabled = True

    def stop(self) -> None:
        if self.file is not None:
            self.file.close()

        self.enabled = False
        self.file = None
        self.writer = None

    def toggle(self) -> bool:
        if self.enabled:
            self.stop()
            return False

        self.start()
        return True

    def write(
        self,
        elapsed_seconds: float,
        features: dict[str, float],
        fatigue_probability: float,
        smoothed_probability: float,
        temporal_state: str,
    ) -> None:
        if (
            not self.enabled
            or self.writer is None
        ):
            return

        self.writer.writerow(
            {
                "timestamp": datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                "elapsed_seconds": round(
                    elapsed_seconds,
                    3,
                ),
                "ear": round(
                    features["ear"],
                    6,
                ),
                "yawn_score": round(
                    features["yawn_score"],
                    6,
                ),
                "head_tilt": round(
                    features["head_tilt"],
                    6,
                ),
                "fatigue_probability": round(
                    fatigue_probability,
                    6,
                ),
                "smoothed_probability": round(
                    smoothed_probability,
                    6,
                ),
                "temporal_state": temporal_state,
            }
        )

        self.file.flush()


# ---------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------

def build_dashboard(
    camera_frame: np.ndarray,
    features: dict[str, float],
    raw_probability: float,
    smoothed_probability: float,
    temporal_state: str,
    probability_history: deque[float],
    fps: float,
    elapsed_seconds: float,
    alert_count: int,
    logging_enabled: bool,
    model_loaded: bool,
) -> np.ndarray:
    dashboard = np.full(
        (900, 1440, 3),
        BACKGROUND,
        dtype=np.uint8,
    )

    draw_text(
        dashboard,
        "DRIVERGUARDIAN AI",
        (36, 48),
        scale=1.0,
        colour=WHITE,
        thickness=2,
    )

    draw_text(
        dashboard,
        "V3 REAL-TIME MONITORING DASHBOARD",
        (37, 73),
        scale=0.42,
        colour=CYAN,
    )

    draw_text(
        dashboard,
        (
            f"{fps:.1f} FPS  |  "
            f"{elapsed_seconds:.1f} seconds"
        ),
        (1110, 49),
        scale=0.46,
        colour=MUTED,
    )

    camera_x = 34
    camera_y = 96
    camera_width = 910
    camera_height = 590

    draw_panel(
        dashboard,
        (camera_x, camera_y),
        (
            camera_x + camera_width,
            camera_y + camera_height,
        ),
        PANEL,
    )

    camera_display = cv2.resize(
        camera_frame,
        (
            camera_width - 24,
            camera_height - 24,
        ),
    )

    dashboard[
        camera_y + 12:
        camera_y + 12 + camera_display.shape[0],
        camera_x + 12:
        camera_x + 12 + camera_display.shape[1],
    ] = camera_display

    status_colour = state_colour(
        temporal_state
    )

    cv2.rectangle(
        dashboard,
        (camera_x + 24, camera_y + 24),
        (camera_x + 250, camera_y + 76),
        status_colour,
        -1,
    )

    draw_text(
        dashboard,
        temporal_state,
        (camera_x + 45, camera_y + 60),
        scale=0.72,
        colour=WHITE,
        thickness=2,
    )

    model_text = (
        "TRAINED MODEL"
        if model_loaded
        else "HEURISTIC FALLBACK"
    )

    draw_text(
        dashboard,
        model_text,
        (
            camera_x + camera_width - 205,
            camera_y + 48,
        ),
        scale=0.40,
        colour=GREEN if model_loaded else AMBER,
    )

    right_x = 968
    right_width = 438

    draw_panel(
        dashboard,
        (right_x, 96),
        (right_x + right_width, 686),
        PANEL,
    )

    draw_probability_gauge(
        dashboard,
        (right_x + right_width // 2, 257),
        112,
        smoothed_probability,
    )

    cv2.rectangle(
        dashboard,
        (right_x + 28, 382),
        (right_x + right_width - 28, 438),
        status_colour,
        -1,
    )

    draw_text(
        dashboard,
        f"STATUS: {temporal_state}",
        (right_x + 50, 418),
        scale=0.70,
        colour=WHITE,
        thickness=2,
    )

    card_width = 181

    draw_metric_card(
        dashboard,
        right_x + 28,
        466,
        card_width,
        "EAR",
        f"{features['ear']:.3f}",
        "eye openness",
    )

    draw_metric_card(
        dashboard,
        right_x + 229,
        466,
        card_width,
        "Yawn Score",
        f"{features['yawn_score']:.3f}",
        "mouth opening",
    )

    draw_metric_card(
        dashboard,
        right_x + 28,
        568,
        card_width,
        "Head Tilt",
        f"{features['head_tilt']:.1f} deg",
        "absolute angle",
    )

    draw_metric_card(
        dashboard,
        right_x + 229,
        568,
        card_width,
        "Alerts",
        str(alert_count),
        "critical triggers",
    )

    draw_timeline(
        dashboard,
        probability_history,
        34,
        712,
        1020,
        158,
    )

    draw_panel(
        dashboard,
        (1078, 712),
        (1406, 870),
        PANEL,
    )

    face_text = (
        "DETECTED"
        if features["face_detected"]
        else "NOT DETECTED"
    )

    draw_text(
        dashboard,
        "SESSION",
        (1100, 744),
        scale=0.46,
        colour=WHITE,
    )

    draw_text(
        dashboard,
        f"Face: {face_text}",
        (1100, 779),
        scale=0.45,
        colour=(
            GREEN
            if features["face_detected"]
            else RED
        ),
    )

    draw_text(
        dashboard,
        f"Raw probability: {raw_probability:.3f}",
        (1100, 808),
        scale=0.42,
        colour=MUTED,
    )

    draw_text(
        dashboard,
        (
            "Logging: ON"
            if logging_enabled
            else "Logging: OFF"
        ),
        (1100, 837),
        scale=0.42,
        colour=(
            GREEN
            if logging_enabled
            else MUTED
        ),
    )

    draw_text(
        dashboard,
        "Q quit | R reset | L logging | I info",
        (1100, 860),
        scale=0.32,
        colour=MUTED,
    )

    return dashboard


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DriverGuardianAI V3 "
            "real-time dashboard"
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index",
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional input video file",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model bundle",
    )

    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Hide facial landmark overlay",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    estimator, model_threshold, model_error = (
        load_model(arguments.model)
    )

    if model_error:
        print()
        print("Model warning:")
        print(model_error)
        print()
        print(
            "The dashboard will run using "
            "the heuristic fallback."
        )

    model_loaded = estimator is not None

    if arguments.video is not None:
        capture = cv2.VideoCapture(
            str(arguments.video)
        )

        input_source = str(arguments.video)

    else:
        capture = cv2.VideoCapture(
            arguments.camera,
            cv2.CAP_DSHOW,
        )

        input_source = (
            f"camera {arguments.camera}"
        )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open {input_source}"
        )

    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    face_mesh = (
        mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
    )

    temporal_engine = TemporalStateEngine()
    logger = SessionLogger()

    probability_history: deque[float] = deque(
        maxlen=180
    )

    drawing_utils = (
        mp.solutions.drawing_utils
    )

    drawing_styles = (
        mp.solutions.drawing_styles
    )

    start_time = time.perf_counter()
    previous_time = start_time

    displayed_fps = 0.0

    print()
    print("=" * 72)
    print("DriverGuardianAI V3")
    print("=" * 72)
    print(f"Input source: {input_source}")
    print(f"Model path:   {arguments.model}")
    print(f"Model loaded: {model_loaded}")
    print(f"Threshold:    {model_threshold:.2f}")
    print()
    print(
        "Controls: Q/Esc quit | "
        "R reset | L logging | I info"
    )
    print("=" * 72)

    try:
        while True:
            success, frame = capture.read()

            if not success:
                if arguments.video is not None:
                    break

                continue

            current_time = time.perf_counter()

            elapsed_seconds = (
                current_time - start_time
            )

            frame_duration = max(
                current_time - previous_time,
                1e-6,
            )

            previous_time = current_time

            instantaneous_fps = (
                1.0 / frame_duration
            )

            if displayed_fps == 0.0:
                displayed_fps = instantaneous_fps
            else:
                displayed_fps = (
                    0.90 * displayed_fps
                    + 0.10 * instantaneous_fps
                )

            if arguments.video is None:
                frame = cv2.flip(frame, 1)

            features, face_landmarks = (
                extract_features(
                    frame,
                    face_mesh,
                )
            )

            if (
                face_landmarks is not None
                and not arguments.no_landmarks
            ):
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=(
                        mp.solutions.face_mesh
                        .FACEMESH_CONTOURS
                    ),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                        drawing_styles
                        .get_default_face_mesh_contours_style()
                    ),
                )

            if features["face_detected"]:
                try:
                    raw_probability = (
                        predict_probability(
                            estimator,
                            features,
                        )
                    )

                except Exception as error:
                    print(
                        "Prediction error:",
                        type(error).__name__,
                        error,
                    )

                    raw_probability = (
                        heuristic_probability(
                            features
                        )
                    )

            else:
                raw_probability = 0.0

            (
                temporal_state,
                smoothed_probability,
            ) = temporal_engine.update(
                raw_probability,
                bool(features["face_detected"]),
            )

            probability_history.append(
                smoothed_probability
            )

            logger.write(
                elapsed_seconds,
                features,
                raw_probability,
                smoothed_probability,
                temporal_state,
            )

            dashboard = build_dashboard(
                camera_frame=frame,
                features=features,
                raw_probability=raw_probability,
                smoothed_probability=(
                    smoothed_probability
                ),
                temporal_state=temporal_state,
                probability_history=(
                    probability_history
                ),
                fps=displayed_fps,
                elapsed_seconds=elapsed_seconds,
                alert_count=(
                    temporal_engine.alert_count
                ),
                logging_enabled=logger.enabled,
                model_loaded=model_loaded,
            )

            cv2.imshow(
                "DriverGuardianAI V3",
                dashboard,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("r"):
                temporal_engine.reset()
                probability_history.clear()

                print(
                    "Temporal state reset."
                )

            if key == ord("l"):
                logging_enabled = (
                    logger.toggle()
                )

                if logging_enabled:
                    print(
                        "Logging started:",
                        logger.path,
                    )
                else:
                    print(
                        "Logging stopped."
                    )

            if key == ord("i"):
                print()
                print("-" * 60)
                print(
                    f"FPS: {displayed_fps:.2f}"
                )
                print(
                    f"EAR: {features['ear']:.4f}"
                )
                print(
                    "Yawn score:",
                    f"{features['yawn_score']:.4f}",
                )
                print(
                    "Head tilt:",
                    f"{features['head_tilt']:.2f}",
                )
                print(
                    "Raw probability:",
                    f"{raw_probability:.4f}",
                )
                print(
                    "Smoothed probability:",
                    f"{smoothed_probability:.4f}",
                )
                print(
                    "Temporal state:",
                    temporal_state,
                )
                print(
                    "Critical alerts:",
                    temporal_engine.alert_count,
                )
                print("-" * 60)

    finally:
        logger.stop()
        capture.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()