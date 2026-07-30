"""
DriverGuardianAI V2 real-time application.

Pipeline
--------
Webcam
    -> VisionAgentV2 raw features
    -> saved V2 Histogram Gradient Boosting pipeline
    -> calibrated Fatigue threshold
    -> temporal decision logic
    -> CSV logging
    -> real-time display

Files required
--------------
src/v2/vision_agent_v2.py
models/v2/driver_guardian_hgb_v2.joblib

Controls
--------
Q or Esc
    Quit safely.

R
    Reset blink count and temporal decision history.

I
    Print model and runtime information.

0
    Set condition to none.

1
    Set condition to glasses.

2
    Set condition to hat.

3
    Set condition to dark.

L
    Toggle CSV logging on or off.

Important
---------
This script sends raw V2 features directly into the saved model bundle.
It does not apply the old V1 normalisation formulas.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd

from src.v2.vision_agent_v2 import VisionAgentV2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v2"
    / "driver_guardian_hgb_v2.joblib"
)

FEATURE_CONTRACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "v2"
    / "feature_contract_v2.json"
)

LOG_DIRECTORY = (
    PROJECT_ROOT
    / "logs"
    / "v2"
)


# ============================================================
# CAMERA AND RUNTIME SETTINGS
# ============================================================

WINDOW_NAME = "DriverGuardianAI V2"

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Two model predictions per second.
INFERENCE_INTERVAL_SECONDS = 0.50

# Temporal decision settings.
TEMPORAL_WINDOW_SIZE = 12
MINIMUM_HISTORY = 5

WARNING_FATIGUE_RATIO = 0.50
CRITICAL_FATIGUE_RATIO = 0.70

CONSECUTIVE_FATIGUE_REQUIRED = 3

ALERT_COOLDOWN_SECONDS = 10.0

# Vision settings.
LOW_LIGHT_THRESHOLD = 50.0
RAW_BLINK_EAR_THRESHOLD = 0.20
MINIMUM_CLOSED_FRAMES = 2
HAND_LOWER_FRAME_RATIO = 0.65

# Default condition can be changed at runtime with 0/1/2/3.
DEFAULT_CONDITION = "none"


# ============================================================
# COLOURS
#
# OpenCV uses BGR.
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
GREY = (180, 180, 180)


# ============================================================
# RESULT OBJECTS
# ============================================================

@dataclass
class PredictionResult:
    """
    One model prediction.
    """

    predicted_class: int
    predicted_label: str
    fatigue_probability: float
    alert_probability: float
    threshold: float
    threshold_margin: float
    confidence: float


@dataclass
class TemporalDecision:
    """
    Smoothed decision based on recent model predictions.
    """

    temporal_state: str
    alert_level: str
    trigger_alert: bool
    fatigue_ratio: float
    average_fatigue_probability: float
    consecutive_fatigue: int
    history_size: int
    reason: str


# ============================================================
# MODEL PREDICTOR
# ============================================================

class V2Predictor:
    """
    Load and use the complete saved V2 model bundle.
    """

    REQUIRED_FEATURES = [
        "ear",
        "yawn_score",
        "head_tilt",
        "hands_detected",
        "condition",
        "low_light",
        "face_confidence",
        "blink_count",
    ]

    def __init__(
        self,
        model_path: Path,
    ) -> None:

        if not model_path.exists():
            raise FileNotFoundError(
                "V2 model bundle was not found: "
                f"{model_path}"
            )

        self.model_path = model_path

        bundle = joblib.load(
            model_path
        )

        required_keys = {
            "pipeline",
            "fatigue_threshold",
            "feature_columns",
            "class_names",
        }

        missing_keys = (
            required_keys
            - set(
                bundle.keys()
            )
        )

        if missing_keys:
            raise KeyError(
                "V2 model bundle is missing keys: "
                f"{sorted(missing_keys)}"
            )

        self.bundle = bundle

        self.pipeline = bundle[
            "pipeline"
        ]

        self.fatigue_threshold = float(
            bundle[
                "fatigue_threshold"
            ]
        )

        self.feature_columns = list(
            bundle[
                "feature_columns"
            ]
        )

        self.class_names = list(
            bundle[
                "class_names"
            ]
        )

        if self.feature_columns != self.REQUIRED_FEATURES:
            raise ValueError(
                "Model feature order does not match "
                "the V2 runtime contract.\n"
                f"Expected: {self.REQUIRED_FEATURES}\n"
                f"Found: {self.feature_columns}"
            )

    @staticmethod
    def _prepare_features(
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate and convert one raw feature dictionary.
        """

        missing = [
            feature
            for feature in V2Predictor.REQUIRED_FEATURES
            if feature not in features
        ]

        if missing:
            raise ValueError(
                "Vision Agent output is missing features: "
                f"{missing}"
            )

        prepared = {
            "ear": float(
                features[
                    "ear"
                ]
            ),
            "yawn_score": float(
                features[
                    "yawn_score"
                ]
            ),
            "head_tilt": float(
                features[
                    "head_tilt"
                ]
            ),
            "hands_detected": float(
                features[
                    "hands_detected"
                ]
            ),
            "condition": str(
                features[
                    "condition"
                ]
            ).strip().lower(),
            "low_light": float(
                bool(
                    features[
                        "low_light"
                    ]
                )
            ),
            "face_confidence": float(
                features[
                    "face_confidence"
                ]
            ),
            "blink_count": float(
                features[
                    "blink_count"
                ]
            ),
        }

        for feature in [
            "ear",
            "yawn_score",
            "head_tilt",
            "hands_detected",
            "low_light",
            "face_confidence",
            "blink_count",
        ]:
            if not np.isfinite(
                prepared[
                    feature
                ]
            ):
                raise ValueError(
                    f"{feature} is not finite."
                )

        return prepared

    def predict(
        self,
        features: Dict[str, Any],
    ) -> PredictionResult:
        """
        Predict Alert or Fatigue from one raw feature dictionary.
        """

        prepared = self._prepare_features(
            features
        )

        dataframe = pd.DataFrame(
            [
                prepared
            ],
            columns=self.feature_columns,
        )

        probabilities = (
            self.pipeline.predict_proba(
                dataframe
            )[0]
        )

        alert_probability = float(
            probabilities[
                0
            ]
        )

        fatigue_probability = float(
            probabilities[
                1
            ]
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

        threshold_margin = abs(
            fatigue_probability
            - self.fatigue_threshold
        )

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            fatigue_probability=(
                fatigue_probability
            ),
            alert_probability=(
                alert_probability
            ),
            threshold=(
                self.fatigue_threshold
            ),
            threshold_margin=float(
                threshold_margin
            ),
            confidence=float(
                confidence
            ),
        )

    def information(
        self,
    ) -> Dict[str, Any]:
        """
        Return saved model metadata.
        """

        return {
            "model_path": str(
                self.model_path
            ),
            "model_name": self.bundle.get(
                "model_name",
                "unknown",
            ),
            "project_version": self.bundle.get(
                "project_version",
                "unknown",
            ),
            "fatigue_threshold": (
                self.fatigue_threshold
            ),
            "feature_columns": (
                self.feature_columns
            ),
            "class_names": (
                self.class_names
            ),
            "training_participants": (
                self.bundle.get(
                    "training_participants",
                    [],
                )
            ),
            "calibration_participants": (
                self.bundle.get(
                    "calibration_participants",
                    [],
                )
            ),
            "test_participants": (
                self.bundle.get(
                    "test_participants",
                    [],
                )
            ),
        }


# ============================================================
# TEMPORAL DECISION AGENT
# ============================================================

class TemporalDecisionAgentV2:
    """
    Smooth frame-level predictions over a short temporal window.
    """

    def __init__(
        self,
        window_size: int = TEMPORAL_WINDOW_SIZE,
        minimum_history: int = MINIMUM_HISTORY,
        warning_ratio_threshold: float = (
            WARNING_FATIGUE_RATIO
        ),
        critical_ratio_threshold: float = (
            CRITICAL_FATIGUE_RATIO
        ),
        consecutive_fatigue_required: int = (
            CONSECUTIVE_FATIGUE_REQUIRED
        ),
        alert_cooldown_seconds: float = (
            ALERT_COOLDOWN_SECONDS
        ),
    ) -> None:

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

        self.consecutive_fatigue_required = int(
            consecutive_fatigue_required
        )

        self.alert_cooldown_seconds = float(
            alert_cooldown_seconds
        )

        self.history: Deque[
            Tuple[
                int,
                float,
            ]
        ] = deque(
            maxlen=self.window_size
        )

        self.consecutive_fatigue = 0

        self.last_alert_time = (
            -float(
                "inf"
            )
        )

    def reset(self) -> None:
        """
        Clear all temporal state.
        """

        self.history.clear()

        self.consecutive_fatigue = 0

        self.last_alert_time = (
            -float(
                "inf"
            )
        )

    def update(
        self,
        prediction: PredictionResult,
    ) -> TemporalDecision:
        """
        Add one prediction and calculate the current temporal state.
        """

        self.history.append(
            (
                prediction.predicted_class,
                prediction.fatigue_probability,
            )
        )

        if prediction.predicted_class == 1:
            self.consecutive_fatigue += 1
        else:
            self.consecutive_fatigue = 0

        history_size = len(
            self.history
        )

        fatigue_predictions = sum(
            class_id
            for class_id, _
            in self.history
        )

        fatigue_ratio = (
            fatigue_predictions
            / history_size
        )

        average_probability = float(
            np.mean(
                [
                    probability
                    for _, probability
                    in self.history
                ]
            )
        )

        if history_size < self.minimum_history:

            return TemporalDecision(
                temporal_state="Monitoring",
                alert_level="none",
                trigger_alert=False,
                fatigue_ratio=float(
                    fatigue_ratio
                ),
                average_fatigue_probability=(
                    average_probability
                ),
                consecutive_fatigue=(
                    self.consecutive_fatigue
                ),
                history_size=history_size,
                reason=(
                    "Collecting more predictions before "
                    "making a temporal decision."
                ),
            )

        critical_by_consecutive = (
            self.consecutive_fatigue
            >= self.consecutive_fatigue_required
        )

        critical_by_ratio = (
            fatigue_ratio
            >= self.critical_ratio_threshold
            and average_probability
            >= prediction.threshold
        )

        warning_by_ratio = (
            fatigue_ratio
            >= self.warning_ratio_threshold
        )

        if (
            critical_by_consecutive
            or critical_by_ratio
        ):

            current_time = monotonic()

            trigger_alert = (
                current_time
                - self.last_alert_time
                >= self.alert_cooldown_seconds
            )

            if trigger_alert:
                self.last_alert_time = (
                    current_time
                )

            if critical_by_consecutive:
                reason = (
                    f"{self.consecutive_fatigue} consecutive "
                    "strong Fatigue predictions detected."
                )
            else:
                reason = (
                    "Sustained Fatigue predictions and high "
                    "average Fatigue probability detected."
                )

            return TemporalDecision(
                temporal_state="Fatigue",
                alert_level="critical",
                trigger_alert=trigger_alert,
                fatigue_ratio=float(
                    fatigue_ratio
                ),
                average_fatigue_probability=(
                    average_probability
                ),
                consecutive_fatigue=(
                    self.consecutive_fatigue
                ),
                history_size=history_size,
                reason=reason,
            )

        if warning_by_ratio:

            return TemporalDecision(
                temporal_state="Possible Fatigue",
                alert_level="warning",
                trigger_alert=False,
                fatigue_ratio=float(
                    fatigue_ratio
                ),
                average_fatigue_probability=(
                    average_probability
                ),
                consecutive_fatigue=(
                    self.consecutive_fatigue
                ),
                history_size=history_size,
                reason=(
                    "The majority of recent predictions "
                    "indicate Fatigue."
                ),
            )

        return TemporalDecision(
            temporal_state="Alert",
            alert_level="none",
            trigger_alert=False,
            fatigue_ratio=float(
                fatigue_ratio
            ),
            average_fatigue_probability=(
                average_probability
            ),
            consecutive_fatigue=(
                self.consecutive_fatigue
            ),
            history_size=history_size,
            reason=(
                "Recent predictions are predominantly Alert."
            ),
        )

    def information(
        self,
    ) -> Dict[str, Any]:
        """
        Return temporal settings.
        """

        return {
            "window_size": (
                self.window_size
            ),
            "minimum_history": (
                self.minimum_history
            ),
            "warning_ratio_threshold": (
                self.warning_ratio_threshold
            ),
            "critical_ratio_threshold": (
                self.critical_ratio_threshold
            ),
            "consecutive_fatigue_required": (
                self.consecutive_fatigue_required
            ),
            "alert_cooldown_seconds": (
                self.alert_cooldown_seconds
            ),
        }


# ============================================================
# CSV LOGGER
# ============================================================

class RuntimeLoggerV2:
    """
    Save raw V2 features, predictions, and temporal decisions.
    """

    FIELDNAMES = [
        "timestamp",
        "elapsed_seconds",
        "ear",
        "yawn_score",
        "head_tilt",
        "hands_detected",
        "condition",
        "low_light",
        "face_confidence",
        "blink_count",
        "predicted_class",
        "predicted_label",
        "alert_probability",
        "fatigue_probability",
        "decision_threshold",
        "confidence",
        "threshold_margin",
        "temporal_state",
        "alert_level",
        "trigger_alert",
        "fatigue_ratio",
        "average_fatigue_probability",
        "consecutive_fatigue",
        "history_size",
        "reason",
    ]

    def __init__(
        self,
        directory: Path,
    ) -> None:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.filepath = (
            directory
            / (
                "driver_guardian_v2_session_"
                f"{timestamp}.csv"
            )
        )

        self.session_start = monotonic()

        self.enabled = True

        self._file = self.filepath.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.FIELDNAMES,
        )

        self._writer.writeheader()

        self._file.flush()

    def set_enabled(
        self,
        enabled: bool,
    ) -> None:
        """
        Enable or disable row writing.
        """

        self.enabled = bool(
            enabled
        )

    def log(
        self,
        features: Dict[str, Any],
        prediction: PredictionResult,
        decision: TemporalDecision,
    ) -> None:
        """
        Write one inference event.
        """

        if not self.enabled:
            return

        row = {
            "timestamp": (
                datetime.now().isoformat(
                    timespec="milliseconds"
                )
            ),
            "elapsed_seconds": (
                monotonic()
                - self.session_start
            ),
            **features,
            "predicted_class": (
                prediction.predicted_class
            ),
            "predicted_label": (
                prediction.predicted_label
            ),
            "alert_probability": (
                prediction.alert_probability
            ),
            "fatigue_probability": (
                prediction.fatigue_probability
            ),
            "decision_threshold": (
                prediction.threshold
            ),
            "confidence": (
                prediction.confidence
            ),
            "threshold_margin": (
                prediction.threshold_margin
            ),
            "temporal_state": (
                decision.temporal_state
            ),
            "alert_level": (
                decision.alert_level
            ),
            "trigger_alert": (
                decision.trigger_alert
            ),
            "fatigue_ratio": (
                decision.fatigue_ratio
            ),
            "average_fatigue_probability": (
                decision
                .average_fatigue_probability
            ),
            "consecutive_fatigue": (
                decision.consecutive_fatigue
            ),
            "history_size": (
                decision.history_size
            ),
            "reason": (
                decision.reason
            ),
        }

        self._writer.writerow(
            row
        )

        self._file.flush()

    def close(self) -> None:
        """
        Close the active log file.
        """

        if not self._file.closed:
            self._file.flush()
            self._file.close()


# ============================================================
# DISPLAY HELPERS
# ============================================================

def draw_text_box(
    frame: np.ndarray,
    text: str,
    origin: Tuple[
        int,
        int,
    ],
    text_colour: Tuple[
        int,
        int,
        int,
    ] = WHITE,
    background_colour: Tuple[
        int,
        int,
        int,
    ] = BLACK,
    font_scale: float = 0.65,
    thickness: int = 2,
    padding: int = 6,
) -> Tuple[int, int]:
    """
    Draw one readable text line with a solid background.
    """

    x, y = origin

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        frame,
        (
            x - padding,
            y - text_height - padding,
        ),
        (
            x + text_width + padding,
            y + baseline + padding,
        ),
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
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_colour,
        thickness,
        cv2.LINE_AA,
    )

    return (
        text_width,
        text_height,
    )


def probability_colour(
    fatigue_probability: float,
    threshold: float,
) -> Tuple[
    int,
    int,
    int,
]:
    """
    Choose a display colour for current probability.
    """

    if fatigue_probability >= threshold:
        return RED

    if fatigue_probability >= max(
        0.40,
        threshold - 0.20,
    ):
        return YELLOW

    return GREEN


def decision_colour(
    decision: TemporalDecision,
) -> Tuple[
    int,
    int,
    int,
]:
    """
    Choose a display colour for the temporal decision.
    """

    if decision.alert_level == "critical":
        return RED

    if decision.alert_level == "warning":
        return YELLOW

    if decision.temporal_state == "Monitoring":
        return CYAN

    return GREEN


def draw_probability_bar(
    frame: np.ndarray,
    fatigue_probability: float,
    threshold: float,
    x: int,
    y: int,
    width: int = 500,
    height: int = 28,
) -> None:
    """
    Draw Fatigue probability and threshold.
    """

    probability = float(
        np.clip(
            fatigue_probability,
            0.0,
            1.0,
        )
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
        width
        * probability
    )

    cv2.rectangle(
        frame,
        (
            x + 2,
            y + 2,
        ),
        (
            x
            + max(
                2,
                filled_width - 2,
            ),
            y + height - 2,
        ),
        probability_colour(
            probability,
            threshold,
        ),
        -1,
    )

    threshold_x = int(
        x
        + width
        * threshold
    )

    cv2.line(
        frame,
        (
            threshold_x,
            y - 4,
        ),
        (
            threshold_x,
            y + height + 4,
        ),
        YELLOW,
        3,
    )


def annotate_runtime(
    frame: np.ndarray,
    prediction: Optional[
        PredictionResult
    ],
    decision: Optional[
        TemporalDecision
    ],
    inference_hz: float,
    frames_per_second: float,
    logging_enabled: bool,
) -> np.ndarray:
    """
    Add model and temporal results to the V2 feature frame.
    """

    annotated = frame.copy()

    frame_height, frame_width = (
        annotated.shape[
            :2
        ]
    )

    # Performance panel.
    right_x = max(
        10,
        frame_width - 300,
    )

    draw_text_box(
        annotated,
        (
            f"FPS: "
            f"{frames_per_second:.1f}"
        ),
        (
            right_x,
            30,
        ),
        WHITE,
        BLACK,
        0.65,
        2,
    )

    draw_text_box(
        annotated,
        (
            f"Inference: "
            f"{inference_hz:.1f} Hz"
        ),
        (
            right_x,
            62,
        ),
        WHITE,
        BLACK,
        0.65,
        2,
    )

    draw_text_box(
        annotated,
        (
            "Logging: "
            + (
                "ON"
                if logging_enabled
                else "OFF"
            )
        ),
        (
            right_x,
            94,
        ),
        (
            GREEN
            if logging_enabled
            else GREY
        ),
        BLACK,
        0.60,
        2,
    )

    if prediction is None:

        draw_text_box(
            annotated,
            "Waiting for model inference...",
            (
                20,
                300,
            ),
            CYAN,
            BLACK,
            0.70,
            2,
        )

        return annotated

    model_colour = probability_colour(
        prediction.fatigue_probability,
        prediction.threshold,
    )

    draw_text_box(
        annotated,
        (
            f"Model: "
            f"{prediction.predicted_label} | "
            f"Fatigue "
            f"{prediction.fatigue_probability:.1%} | "
            f"Alert "
            f"{prediction.alert_probability:.1%}"
        ),
        (
            20,
            300,
        ),
        model_colour,
        BLACK,
        0.72,
        2,
    )

    draw_text_box(
        annotated,
        (
            "Fatigue probability: "
            f"{prediction.fatigue_probability:.1%} | "
            "Threshold: "
            f"{prediction.threshold:.0%}"
        ),
        (
            20,
            338,
        ),
        WHITE,
        BLACK,
        0.60,
        1,
    )

    draw_probability_bar(
        annotated,
        prediction.fatigue_probability,
        prediction.threshold,
        25,
        355,
        width=min(
            500,
            frame_width - 60,
        ),
    )

    if decision is None:
        return annotated

    temporal_colour = decision_colour(
        decision
    )

    draw_text_box(
        annotated,
        (
            "Temporal state: "
            f"{decision.temporal_state}"
        ),
        (
            20,
            430,
        ),
        temporal_colour,
        BLACK,
        0.72,
        2,
    )

    draw_text_box(
        annotated,
        (
            "Alert level: "
            f"{decision.alert_level}"
        ),
        (
            20,
            467,
        ),
        temporal_colour,
        BLACK,
        0.68,
        2,
    )

    draw_text_box(
        annotated,
        (
            "Recent Fatigue ratio: "
            f"{decision.fatigue_ratio:.0%}"
        ),
        (
            20,
            505,
        ),
        WHITE,
        BLACK,
        0.58,
        1,
    )

    draw_text_box(
        annotated,
        (
            "Average Fatigue probability: "
            f"{decision.average_fatigue_probability:.1%}"
        ),
        (
            20,
            538,
        ),
        WHITE,
        BLACK,
        0.58,
        1,
    )

    draw_text_box(
        annotated,
        (
            "Consecutive Fatigue: "
            f"{decision.consecutive_fatigue} | "
            "History: "
            f"{decision.history_size}"
        ),
        (
            20,
            571,
        ),
        WHITE,
        BLACK,
        0.58,
        1,
    )

    if decision.alert_level == "critical":

        alert_text = (
            "NEW FATIGUE ALERT"
            if decision.trigger_alert
            else "SUSTAINED FATIGUE"
        )

        draw_text_box(
            annotated,
            alert_text,
            (
                20,
                614,
            ),
            RED,
            BLACK,
            0.82,
            3,
        )

    elif decision.alert_level == "warning":

        draw_text_box(
            annotated,
            "POSSIBLE FATIGUE",
            (
                20,
                614,
            ),
            YELLOW,
            BLACK,
            0.78,
            2,
        )

    else:

        draw_text_box(
            annotated,
            "No active fatigue alert",
            (
                20,
                614,
            ),
            GREEN,
            BLACK,
            0.68,
            2,
        )

    draw_text_box(
        annotated,
        (
            "Reason: "
            f"{decision.reason}"
        ),
        (
            20,
            min(
                frame_height - 20,
                655,
            ),
        ),
        WHITE,
        BLACK,
        0.52,
        1,
    )

    return annotated


# ============================================================
# INFORMATION
# ============================================================

def load_feature_contract() -> Optional[
    Dict[str, Any]
]:
    """
    Load the runtime contract when present.
    """

    if not FEATURE_CONTRACT_PATH.exists():
        return None

    with FEATURE_CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def print_runtime_information(
    predictor: V2Predictor,
    decision_agent: TemporalDecisionAgentV2,
    vision_agent: VisionAgentV2,
    logger: RuntimeLoggerV2,
) -> None:
    """
    Print current runtime settings.
    """

    print("\n" + "-" * 72)
    print("DriverGuardianAI V2 Runtime Information")
    print("-" * 72)

    print("\nModel:")

    for key, value in predictor.information().items():
        print(
            f"{key}: {value}"
        )

    print("\nTemporal decision configuration:")

    for key, value in decision_agent.information().items():
        print(
            f"{key}: {value}"
        )

    print("\nVision configuration:")

    print(
        "condition: "
        f"{vision_agent.condition}"
    )

    print(
        "low_light_threshold: "
        f"{vision_agent.low_light_threshold}"
    )

    print(
        "raw_blink_ear_threshold: "
        f"{vision_agent.raw_blink_ear_threshold}"
    )

    print(
        "minimum_closed_frames: "
        f"{vision_agent.minimum_closed_frames}"
    )

    print(
        "hand_lower_frame_ratio: "
        f"{vision_agent.hand_lower_frame_ratio}"
    )

    print("\nLogging:")

    print(
        f"enabled: {logger.enabled}"
    )

    print(
        f"path: {logger.filepath}"
    )

    print("-" * 72)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:
    """
    Run DriverGuardianAI V2 in real time.
    """

    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Real-Time Raw-Feature Application")
    print("=" * 72)

    print("Q or Esc: quit")
    print("R: reset blink count and temporal history")
    print("I: print model/runtime information")
    print("0: condition none")
    print("1: condition glasses")
    print("2: condition hat")
    print("3: condition dark")
    print("L: toggle logging")
    print(
        "Inference interval: "
        f"{INFERENCE_INTERVAL_SECONDS} seconds"
    )

    predictor = V2Predictor(
        MODEL_PATH
    )

    decision_agent = (
        TemporalDecisionAgentV2()
    )

    vision_agent = VisionAgentV2(
        low_light_threshold=(
            LOW_LIGHT_THRESHOLD
        ),
        raw_blink_ear_threshold=(
            RAW_BLINK_EAR_THRESHOLD
        ),
        minimum_closed_frames=(
            MINIMUM_CLOSED_FRAMES
        ),
        hand_lower_frame_ratio=(
            HAND_LOWER_FRAME_RATIO
        ),
        condition=(
            DEFAULT_CONDITION
        ),
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

    application_start_time = monotonic()

    frame_counter = 0

    last_fps_time = monotonic()

    current_fps = 0.0

    print(
        "\nModel loaded successfully."
    )

    print(
        "Fatigue threshold: "
        f"{predictor.fatigue_threshold:.2f}"
    )

    print(
        "Expected raw features: "
        f"{predictor.feature_columns}"
    )

    print(
        "Logging to: "
        f"{logger.filepath}"
    )

    contract = load_feature_contract()

    if contract is not None:
        print(
            "Feature contract loaded: "
            f"{FEATURE_CONTRACT_PATH}"
        )

    print_runtime_information(
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

            elapsed_fps_time = (
                current_time
                - last_fps_time
            )

            if elapsed_fps_time >= 1.0:

                current_fps = (
                    frame_counter
                    / elapsed_fps_time
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

            inference_hz = (
                1.0
                / INFERENCE_INTERVAL_SECONDS
            )

            displayed_frame = (
                annotate_runtime(
                    vision_result.annotated_frame,
                    latest_prediction,
                    latest_decision,
                    inference_hz,
                    current_fps,
                    logger.enabled,
                )
            )

            cv2.imshow(
                WINDOW_NAME,
                displayed_frame,
            )

            key = (
                cv2.waitKey(
                    1
                )
                & 0xFF
            )

            if key in {
                ord(
                    "q"
                ),
                ord(
                    "Q"
                ),
                27,
            }:
                break

            if key in {
                ord(
                    "r"
                ),
                ord(
                    "R"
                ),
            }:

                decision_agent.reset()

                vision_agent.reset_temporal_state()

                latest_prediction = None
                latest_decision = None
                last_inference_time = 0.0

                print(
                    "Blink count and temporal "
                    "history reset."
                )

            elif key in {
                ord(
                    "i"
                ),
                ord(
                    "I"
                ),
            }:

                print_runtime_information(
                    predictor,
                    decision_agent,
                    vision_agent,
                    logger,
                )

            elif key == ord(
                "0"
            ):

                vision_agent.set_condition(
                    "none"
                )

                print(
                    "Condition set to none."
                )

            elif key == ord(
                "1"
            ):

                vision_agent.set_condition(
                    "glasses"
                )

                print(
                    "Condition set to glasses."
                )

            elif key == ord(
                "2"
            ):

                vision_agent.set_condition(
                    "hat"
                )

                print(
                    "Condition set to hat."
                )

            elif key == ord(
                "3"
            ):

                vision_agent.set_condition(
                    "dark"
                )

                print(
                    "Condition set to dark."
                )

            elif key in {
                ord(
                    "l"
                ),
                ord(
                    "L"
                ),
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

        runtime_seconds = (
            monotonic()
            - application_start_time
        )

        print(
            "\nDriverGuardianAI V2 stopped safely."
        )

        print(
            "Runtime: "
            f"{runtime_seconds:.1f} seconds"
        )

        print(
            "Log file: "
            f"{logger.filepath}"
        )


if __name__ == "__main__":
    main()