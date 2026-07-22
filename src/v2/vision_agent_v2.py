"""
DriverGuardianAI V2 Vision Agent.

This module extracts the raw feature representation expected by the
V2 Histogram Gradient Boosting model.

V2 feature contract
-------------------
1. ear
   Raw mean eye-aspect ratio. No division by 0.30 and no clipping.

2. yawn_score
   Raw mouth opening divided by interocular distance. No division by
   0.60 and no clipping.

3. head_tilt
   Absolute vertical displacement, in pixels, between the left-eye and
   right-eye midpoints. Do not convert to degrees.

4. hands_detected
   1.0 only when at least one detected wrist is below the configured
   lower-frame threshold. Simple hand presence is not sufficient.

5. condition
   One of: none, glasses, hat, dark.

6. low_light
   True when mean grayscale brightness is below 50 by default.

7. face_confidence
   1.0 when Face Mesh returns landmarks. Legacy Face Mesh does not
   expose a direct per-frame confidence value.

8. blink_count
   Raw cumulative blink count since this VisionAgentV2 instance was
   created or last reset.

Important
---------
This module deliberately does not scale, normalise, standardise, or
clip model features. The saved V2 model bundle performs its own fitted
preprocessing internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class VisionResultV2:
    """
    Result returned after processing one video frame.
    """

    features: Optional[Dict[str, Any]]
    annotated_frame: np.ndarray
    face_detected: bool
    status: str
    diagnostics: Dict[str, Any]


# ============================================================
# VISION AGENT
# ============================================================

class VisionAgentV2:
    """
    Extract raw DriverGuardianAI V2 features from webcam frames.
    """

    # --------------------------------------------------------
    # MediaPipe Face Mesh landmark indexes
    # --------------------------------------------------------

    LEFT_EYE = (
        33,
        160,
        158,
        133,
        153,
        144,
    )

    RIGHT_EYE = (
        362,
        385,
        387,
        263,
        373,
        380,
    )

    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133

    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362

    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    UPPER_LIP = 13
    LOWER_LIP = 14

    LEFT_WRIST = 0
    RIGHT_WRIST = 0

    ALLOWED_CONDITIONS = {
        "none",
        "glasses",
        "hat",
        "dark",
    }

    def __init__(
        self,
        max_num_faces: int = 1,
        max_num_hands: int = 2,
        face_detection_confidence: float = 0.50,
        face_tracking_confidence: float = 0.50,
        hand_detection_confidence: float = 0.50,
        hand_tracking_confidence: float = 0.50,
        low_light_threshold: float = 50.0,
        raw_blink_ear_threshold: float = 0.20,
        minimum_closed_frames: int = 2,
        hand_lower_frame_ratio: float = 0.65,
        condition: str = "none",
        automatically_mark_dark_condition: bool = True,
        draw_face_mesh: bool = False,
        draw_hand_landmarks: bool = False,
    ) -> None:
        """
        Initialise MediaPipe detectors and temporal blink state.

        Parameters
        ----------
        low_light_threshold:
            Mean grayscale threshold used by the original collection
            contract. The default is 50.

        raw_blink_ear_threshold:
            Raw EAR value below which the eyes are treated as closed.
            The default 0.20 approximately corresponds to the old
            normalised threshold of 0.65 after multiplying by 0.30.

        minimum_closed_frames:
            Number of consecutive closed-eye frames required before an
            eye closure may count as a blink.

        hand_lower_frame_ratio:
            A detected wrist must have normalised y greater than this
            value to produce hands_detected = 1.0.

        condition:
            Manual visible condition: none, glasses, hat, or dark.
            When automatically_mark_dark_condition is True, a low-light
            frame overrides this value with dark.
        """

        if minimum_closed_frames < 1:
            raise ValueError(
                "minimum_closed_frames must be at least 1."
            )

        if not 0.0 < hand_lower_frame_ratio < 1.0:
            raise ValueError(
                "hand_lower_frame_ratio must be between 0 and 1."
            )

        condition = self._normalise_condition(
            condition
        )

        self.low_light_threshold = float(
            low_light_threshold
        )

        self.raw_blink_ear_threshold = float(
            raw_blink_ear_threshold
        )

        self.minimum_closed_frames = int(
            minimum_closed_frames
        )

        self.hand_lower_frame_ratio = float(
            hand_lower_frame_ratio
        )

        self.condition = condition

        self.automatically_mark_dark_condition = bool(
            automatically_mark_dark_condition
        )

        self.draw_face_mesh = bool(
            draw_face_mesh
        )

        self.draw_hand_landmarks = bool(
            draw_hand_landmarks
        )

        # Raw cumulative blink state.
        self.closed_eye_frames = 0
        self.eye_was_closed = False
        self.blink_count = 0

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=(
                face_detection_confidence
            ),
            min_tracking_confidence=(
                face_tracking_confidence
            ),
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=(
                hand_detection_confidence
            ),
            min_tracking_confidence=(
                hand_tracking_confidence
            ),
        )

    # ========================================================
    # RESOURCE MANAGEMENT
    # ========================================================

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """

        self.face_mesh.close()
        self.hands.close()

    def reset_temporal_state(self) -> None:
        """
        Reset cumulative blink tracking.
        """

        self.closed_eye_frames = 0
        self.eye_was_closed = False
        self.blink_count = 0

    def set_condition(
        self,
        condition: str,
    ) -> None:
        """
        Change the manually selected visible condition.
        """

        self.condition = self._normalise_condition(
            condition
        )

    # ========================================================
    # VALIDATION HELPERS
    # ========================================================

    @classmethod
    def _normalise_condition(
        cls,
        condition: str,
    ) -> str:
        """
        Validate and normalise a condition value.
        """

        normalised = str(
            condition
        ).strip().lower()

        if normalised not in cls.ALLOWED_CONDITIONS:
            raise ValueError(
                "condition must be one of: "
                f"{sorted(cls.ALLOWED_CONDITIONS)}"
            )

        return normalised

    @staticmethod
    def _validate_frame(
        frame: np.ndarray,
    ) -> None:
        """
        Validate a supplied OpenCV frame.
        """

        if frame is None:
            raise ValueError(
                "The supplied frame is None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "The supplied frame must be a NumPy array."
            )

        if frame.size == 0:
            raise ValueError(
                "The supplied frame is empty."
            )

        if frame.ndim != 3:
            raise ValueError(
                "The supplied frame must be a BGR image."
            )

    # ========================================================
    # GEOMETRY HELPERS
    # ========================================================

    @staticmethod
    def _distance(
        point_a: np.ndarray,
        point_b: np.ndarray,
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        """

        return float(
            np.linalg.norm(
                point_a - point_b
            )
        )

    @staticmethod
    def _landmark_point(
        landmarks,
        index: int,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Convert a normalised landmark to pixel coordinates.
        """

        landmark = landmarks[
            index
        ]

        return np.array(
            [
                landmark.x
                * frame_width,
                landmark.y
                * frame_height,
            ],
            dtype=np.float64,
        )

    def _eye_midpoint(
        self,
        landmarks,
        first_index: int,
        second_index: int,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Return the pixel midpoint between two eye-corner landmarks.
        """

        first = self._landmark_point(
            landmarks,
            first_index,
            frame_width,
            frame_height,
        )

        second = self._landmark_point(
            landmarks,
            second_index,
            frame_width,
            frame_height,
        )

        return (
            first + second
        ) / 2.0

    # ========================================================
    # EAR
    # ========================================================

    def _calculate_eye_ear(
        self,
        landmarks,
        eye_indexes: Sequence[int],
        frame_width: int,
        frame_height: int,
    ) -> float:
        """
        Calculate raw eye-aspect ratio for one eye.
        """

        if len(
            eye_indexes
        ) != 6:
            raise ValueError(
                "Each eye definition must contain six landmarks."
            )

        points = [
            self._landmark_point(
                landmarks,
                index,
                frame_width,
                frame_height,
            )
            for index in eye_indexes
        ]

        horizontal = self._distance(
            points[0],
            points[3],
        )

        if horizontal <= 1e-9:
            return 0.0

        vertical_1 = self._distance(
            points[1],
            points[5],
        )

        vertical_2 = self._distance(
            points[2],
            points[4],
        )

        return float(
            (
                vertical_1
                + vertical_2
            )
            / (
                2.0
                * horizontal
            )
        )

    def _calculate_raw_ear(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[
        float,
        float,
        float,
    ]:
        """
        Calculate left, right, and mean raw EAR.
        """

        left_ear = self._calculate_eye_ear(
            landmarks,
            self.LEFT_EYE,
            frame_width,
            frame_height,
        )

        right_ear = self._calculate_eye_ear(
            landmarks,
            self.RIGHT_EYE,
            frame_width,
            frame_height,
        )

        mean_ear = float(
            (
                left_ear
                + right_ear
            )
            / 2.0
        )

        return (
            left_ear,
            right_ear,
            mean_ear,
        )

    # ========================================================
    # YAWN SCORE
    # ========================================================

    def _calculate_raw_yawn_score(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[
        float,
        float,
        float,
    ]:
        """
        Calculate raw mouth opening divided by interocular distance.

        Returns
        -------
        yawn_score
        mouth_opening_pixels
        interocular_distance_pixels
        """

        upper_lip = self._landmark_point(
            landmarks,
            self.UPPER_LIP,
            frame_width,
            frame_height,
        )

        lower_lip = self._landmark_point(
            landmarks,
            self.LOWER_LIP,
            frame_width,
            frame_height,
        )

        left_eye_reference = (
            self._landmark_point(
                landmarks,
                self.LEFT_EYE_OUTER,
                frame_width,
                frame_height,
            )
        )

        right_eye_reference = (
            self._landmark_point(
                landmarks,
                self.RIGHT_EYE_OUTER,
                frame_width,
                frame_height,
            )
        )

        mouth_opening = self._distance(
            upper_lip,
            lower_lip,
        )

        interocular_distance = self._distance(
            left_eye_reference,
            right_eye_reference,
        )

        if interocular_distance <= 1e-9:
            return (
                0.0,
                mouth_opening,
                interocular_distance,
            )

        yawn_score = float(
            mouth_opening
            / interocular_distance
        )

        return (
            yawn_score,
            mouth_opening,
            interocular_distance,
        )

    # ========================================================
    # HEAD TILT
    # ========================================================

    def _calculate_raw_head_tilt(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[
        float,
        float,
        float,
    ]:
        """
        Calculate absolute eye-midpoint vertical displacement in pixels.

        Returns
        -------
        head_tilt_pixels
        left_eye_midpoint_y
        right_eye_midpoint_y
        """

        left_midpoint = self._eye_midpoint(
            landmarks,
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            frame_width,
            frame_height,
        )

        right_midpoint = self._eye_midpoint(
            landmarks,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_INNER,
            frame_width,
            frame_height,
        )

        left_y = float(
            left_midpoint[1]
        )

        right_y = float(
            right_midpoint[1]
        )

        head_tilt_pixels = float(
            abs(
                left_y
                - right_y
            )
        )

        return (
            head_tilt_pixels,
            left_y,
            right_y,
        )

    # ========================================================
    # HAND FEATURE
    # ========================================================

    def _calculate_hands_detected(
        self,
        hand_results,
    ) -> Tuple[
        float,
        int,
        Tuple[float, ...],
    ]:
        """
        Apply the original lower-frame wrist rule.

        hands_detected is 1.0 only when at least one detected wrist has
        normalised y greater than hand_lower_frame_ratio.
        """

        if not hand_results.multi_hand_landmarks:
            return (
                0.0,
                0,
                tuple(),
            )

        wrist_y_values = []

        for hand_landmarks in (
            hand_results.multi_hand_landmarks
        ):
            wrist = hand_landmarks.landmark[
                self.LEFT_WRIST
            ]

            wrist_y_values.append(
                float(
                    wrist.y
                )
            )

        lower_frame_wrist_found = any(
            wrist_y
            > self.hand_lower_frame_ratio
            for wrist_y in wrist_y_values
        )

        return (
            1.0
            if lower_frame_wrist_found
            else 0.0,
            len(
                wrist_y_values
            ),
            tuple(
                wrist_y_values
            ),
        )

    # ========================================================
    # BLINK FEATURE
    # ========================================================

    def _update_blink_count(
        self,
        raw_ear: float,
    ) -> Tuple[
        float,
        bool,
    ]:
        """
        Update and return raw cumulative blink count.
        """

        eyes_closed = (
            raw_ear
            < self.raw_blink_ear_threshold
        )

        if eyes_closed:
            self.closed_eye_frames += 1

            if (
                self.closed_eye_frames
                >= self.minimum_closed_frames
            ):
                self.eye_was_closed = True

        else:
            if self.eye_was_closed:
                self.blink_count += 1

            self.closed_eye_frames = 0
            self.eye_was_closed = False

        return (
            float(
                self.blink_count
            ),
            bool(
                eyes_closed
            ),
        )

    # ========================================================
    # LIGHT AND CONDITION
    # ========================================================

    @staticmethod
    def _calculate_brightness(
        frame: np.ndarray,
    ) -> float:
        """
        Calculate mean grayscale brightness.
        """

        grayscale = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        return float(
            np.mean(
                grayscale
            )
        )

    def _resolve_condition(
        self,
        low_light: bool,
    ) -> str:
        """
        Resolve the runtime condition category.
        """

        if (
            low_light
            and self.automatically_mark_dark_condition
        ):
            return "dark"

        return self.condition

    # ========================================================
    # DRAWING
    # ========================================================

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        face_landmarks,
        hand_results,
    ) -> None:
        """
        Optionally draw MediaPipe landmarks in place.
        """

        if (
            self.draw_face_mesh
            and face_landmarks is not None
        ):
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=(
                    face_landmarks
                ),
                connections=(
                    self.mp_face_mesh
                    .FACEMESH_CONTOURS
                ),
            )

        if (
            self.draw_hand_landmarks
            and hand_results
            .multi_hand_landmarks
        ):
            for hand_landmarks in (
                hand_results
                .multi_hand_landmarks
            ):
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=(
                        hand_landmarks
                    ),
                    connections=(
                        self.mp_hands
                        .HAND_CONNECTIONS
                    ),
                )

    @staticmethod
    def _draw_text_line(
        frame: np.ndarray,
        text: str,
        line_number: int,
        colour: Tuple[
            int,
            int,
            int,
        ] = (
            255,
            255,
            255,
        ),
    ) -> None:
        """
        Draw one diagnostics line.
        """

        y = (
            25
            + line_number
            * 23
        )

        cv2.putText(
            frame,
            text,
            (
                10,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            1,
            cv2.LINE_AA,
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        features: Dict[str, Any],
        diagnostics: Dict[str, Any],
    ) -> np.ndarray:
        """
        Add raw feature and diagnostics values to a frame.
        """

        annotated = frame.copy()

        lines = [
            (
                "V2 RAW FEATURES"
            ),
            (
                f"EAR raw: "
                f"{features['ear']:.3f}"
            ),
            (
                f"Yawn raw: "
                f"{features['yawn_score']:.3f}"
            ),
            (
                f"Head tilt px: "
                f"{features['head_tilt']:.2f}"
            ),
            (
                f"Hands lower-frame: "
                f"{int(features['hands_detected'])}"
            ),
            (
                f"Brightness: "
                f"{diagnostics['brightness']:.1f}"
            ),
            (
                f"Condition: "
                f"{features['condition']}"
            ),
            (
                f"Low light: "
                f"{features['low_light']}"
            ),
            (
                f"Blink count raw: "
                f"{features['blink_count']:.0f}"
            ),
            (
                f"Eyes closed: "
                f"{diagnostics['eyes_closed']}"
            ),
        ]

        for line_number, text in enumerate(
            lines
        ):
            colour = (
                (
                    0,
                    255,
                    255,
                )
                if line_number == 0
                else (
                    255,
                    255,
                    255,
                )
            )

            self._draw_text_line(
                annotated,
                text,
                line_number,
                colour,
            )

        return annotated

    def _annotate_no_face(
        self,
        frame: np.ndarray,
        brightness: float,
        hands_detected: float,
    ) -> np.ndarray:
        """
        Add a no-face message.
        """

        annotated = frame.copy()

        cv2.putText(
            annotated,
            "No face detected",
            (
                10,
                30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            (
                f"Brightness: "
                f"{brightness:.1f} | "
                f"Hands lower-frame: "
                f"{int(hands_detected)}"
            ),
            (
                10,
                60,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        return annotated

    # ========================================================
    # FRAME PROCESSING
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> VisionResultV2:
        """
        Process one BGR webcam frame.

        Returns raw V2 model features and an annotated frame.
        """

        self._validate_frame(
            frame
        )

        frame_height, frame_width = (
            frame.shape[
                :2
            ]
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        rgb_frame.flags.writeable = False

        face_results = self.face_mesh.process(
            rgb_frame
        )

        hand_results = self.hands.process(
            rgb_frame
        )

        rgb_frame.flags.writeable = True

        brightness = (
            self._calculate_brightness(
                frame
            )
        )

        low_light = bool(
            brightness
            < self.low_light_threshold
        )

        condition = self._resolve_condition(
            low_light
        )

        (
            hands_detected,
            detected_hand_count,
            wrist_y_values,
        ) = self._calculate_hands_detected(
            hand_results
        )

        if not face_results.multi_face_landmarks:

            annotated = self._annotate_no_face(
                frame,
                brightness,
                hands_detected,
            )

            diagnostics = {
                "brightness": (
                    brightness
                ),
                "low_light": (
                    low_light
                ),
                "condition": (
                    condition
                ),
                "detected_hand_count": (
                    detected_hand_count
                ),
                "wrist_y_values": (
                    wrist_y_values
                ),
            }

            return VisionResultV2(
                features=None,
                annotated_frame=annotated,
                face_detected=False,
                status="No face detected",
                diagnostics=diagnostics,
            )

        face_landmarks = (
            face_results
            .multi_face_landmarks[
                0
            ]
        )

        landmarks = (
            face_landmarks
            .landmark
        )

        (
            left_ear,
            right_ear,
            raw_ear,
        ) = self._calculate_raw_ear(
            landmarks,
            frame_width,
            frame_height,
        )

        (
            yawn_score,
            mouth_opening_pixels,
            interocular_distance_pixels,
        ) = self._calculate_raw_yawn_score(
            landmarks,
            frame_width,
            frame_height,
        )

        (
            head_tilt_pixels,
            left_eye_midpoint_y,
            right_eye_midpoint_y,
        ) = self._calculate_raw_head_tilt(
            landmarks,
            frame_width,
            frame_height,
        )

        (
            blink_count,
            eyes_closed,
        ) = self._update_blink_count(
            raw_ear
        )

        face_confidence = 1.0

        features: Dict[str, Any] = {
            "ear": float(
                raw_ear
            ),
            "yawn_score": float(
                yawn_score
            ),
            "head_tilt": float(
                head_tilt_pixels
            ),
            "hands_detected": float(
                hands_detected
            ),
            "condition": str(
                condition
            ),
            "low_light": bool(
                low_light
            ),
            "face_confidence": float(
                face_confidence
            ),
            "blink_count": float(
                blink_count
            ),
        }

        diagnostics: Dict[str, Any] = {
            "frame_width": int(
                frame_width
            ),
            "frame_height": int(
                frame_height
            ),
            "left_ear": float(
                left_ear
            ),
            "right_ear": float(
                right_ear
            ),
            "raw_ear": float(
                raw_ear
            ),
            "raw_blink_ear_threshold": float(
                self.raw_blink_ear_threshold
            ),
            "eyes_closed": bool(
                eyes_closed
            ),
            "closed_eye_frames": int(
                self.closed_eye_frames
            ),
            "mouth_opening_pixels": float(
                mouth_opening_pixels
            ),
            "interocular_distance_pixels": float(
                interocular_distance_pixels
            ),
            "left_eye_midpoint_y": float(
                left_eye_midpoint_y
            ),
            "right_eye_midpoint_y": float(
                right_eye_midpoint_y
            ),
            "head_tilt_pixels": float(
                head_tilt_pixels
            ),
            "brightness": float(
                brightness
            ),
            "low_light_threshold": float(
                self.low_light_threshold
            ),
            "detected_hand_count": int(
                detected_hand_count
            ),
            "wrist_y_values": tuple(
                wrist_y_values
            ),
            "hand_lower_frame_ratio": float(
                self.hand_lower_frame_ratio
            ),
        }

        annotated = self._annotate_frame(
            frame,
            features,
            diagnostics,
        )

        self._draw_landmarks(
            annotated,
            face_landmarks,
            hand_results,
        )

        return VisionResultV2(
            features=features,
            annotated_frame=annotated,
            face_detected=True,
            status="Face detected",
            diagnostics=diagnostics,
        )


# ============================================================
# MANUAL CAMERA TEST
# ============================================================

def run_manual_camera_test(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    condition: str = "none",
) -> None:
    """
    Run VisionAgentV2 without the classifier.

    Controls
    --------
    Q or Esc:
        Quit.

    R:
        Reset cumulative blink count.

    0:
        Set condition to none.

    1:
        Set condition to glasses.

    2:
        Set condition to hat.

    3:
        Set condition to dark.
    """

    capture = cv2.VideoCapture(
        camera_index
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        width,
    )

    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        height,
    )

    agent = VisionAgentV2(
        condition=condition
    )

    window_name = (
        "DriverGuardianAI V2 "
        "Vision Test"
    )

    print("=" * 72)
    print("DriverGuardianAI V2 Vision Test")
    print("=" * 72)
    print("Q or Esc: quit")
    print("R: reset cumulative blink count")
    print("0: none")
    print("1: glasses")
    print("2: hat")
    print("3: dark")

    try:
        while True:

            success, frame = capture.read()

            if not success:
                print(
                    "Could not read webcam frame."
                )
                break

            result = agent.process_frame(
                frame
            )

            cv2.imshow(
                window_name,
                result.annotated_frame,
            )

            key = (
                cv2.waitKey(
                    1
                )
                & 0xFF
            )

            if key in (
                ord(
                    "q"
                ),
                27,
            ):
                break

            if key == ord(
                "r"
            ):
                agent.reset_temporal_state()

                print(
                    "Blink count reset."
                )

            elif key == ord(
                "0"
            ):
                agent.set_condition(
                    "none"
                )

                print(
                    "Condition set to none."
                )

            elif key == ord(
                "1"
            ):
                agent.set_condition(
                    "glasses"
                )

                print(
                    "Condition set to glasses."
                )

            elif key == ord(
                "2"
            ):
                agent.set_condition(
                    "hat"
                )

                print(
                    "Condition set to hat."
                )

            elif key == ord(
                "3"
            ):
                agent.set_condition(
                    "dark"
                )

                print(
                    "Condition set to dark."
                )

    finally:
        capture.release()
        agent.close()
        cv2.destroyAllWindows()

        print(
            "VisionAgentV2 stopped safely."
        )


if __name__ == "__main__":
    run_manual_camera_test()