"""
Vision Agent for DriverGuardianAI.

Responsibilities
----------------
- Process webcam frames.
- Detect facial and hand landmarks with MediaPipe.
- Calculate the eight features required by the trained model.
- Track blinking over time.
- Return annotated frames for the real-time application.

Model feature contract
----------------------
1. ear
2. yawn_score
3. head_tilt
4. hands_detected
5. condition
6. low_light
7. face_confidence
8. blink_count

Important
---------
The normalisation constants in this module must eventually be verified
against the feature-generation code used to build dataset_exp3.csv.
"""

from collections import deque
from dataclasses import dataclass
from math import atan2, degrees
from time import monotonic
from typing import Deque, Dict, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class VisionResult:
    """
    Result returned after processing one video frame.
    """

    features: Optional[Dict]
    annotated_frame: np.ndarray
    face_detected: bool
    status: str


class VisionAgent:
    """
    Extract DriverGuardianAI features from webcam frames.
    """

    # MediaPipe Face Mesh landmark indexes.
    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)

    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    UPPER_LIP = 13
    LOWER_LIP = 14

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    def __init__(
        self,
        max_num_faces: int = 1,
        max_num_hands: int = 2,
        face_detection_confidence: float = 0.50,
        face_tracking_confidence: float = 0.50,
        hand_detection_confidence: float = 0.50,
        hand_tracking_confidence: float = 0.50,
        low_light_threshold: float = 70.0,
        blink_ear_threshold: float = 0.65,
        minimum_closed_frames: int = 2,
        blink_window_seconds: float = 60.0
    ):
        """
        Initialise MediaPipe detectors and temporal blink state.
        """

        if minimum_closed_frames < 1:
            raise ValueError(
                "minimum_closed_frames must be at least 1."
            )

        self.low_light_threshold = float(
            low_light_threshold
        )

        self.blink_ear_threshold = float(
            blink_ear_threshold
        )

        self.minimum_closed_frames = int(
            minimum_closed_frames
        )

        self.blink_window_seconds = float(
            blink_window_seconds
        )

        self.closed_eye_frames = 0
        self.eye_was_closed = False

        self.blink_timestamps: Deque[float] = deque()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=face_detection_confidence,
            min_tracking_confidence=face_tracking_confidence
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=hand_detection_confidence,
            min_tracking_confidence=hand_tracking_confidence
        )

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """

        self.face_mesh.close()
        self.hands.close()

    @staticmethod
    def _distance(
        point_a: np.ndarray,
        point_b: np.ndarray
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        """

        return float(
            np.linalg.norm(point_a - point_b)
        )

    @staticmethod
    def _landmark_point(
        landmarks,
        index: int,
        frame_width: int,
        frame_height: int
    ) -> np.ndarray:
        """
        Convert a normalised landmark into pixel coordinates.
        """

        landmark = landmarks[index]

        return np.array(
            [
                landmark.x * frame_width,
                landmark.y * frame_height
            ],
            dtype=np.float32
        )

    def _calculate_raw_ear(
        self,
        landmarks,
        eye_indexes: Sequence[int],
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Calculate the standard eye-aspect ratio for one eye.
        """

        p1 = self._landmark_point(
            landmarks,
            eye_indexes[0],
            frame_width,
            frame_height
        )

        p2 = self._landmark_point(
            landmarks,
            eye_indexes[1],
            frame_width,
            frame_height
        )

        p3 = self._landmark_point(
            landmarks,
            eye_indexes[2],
            frame_width,
            frame_height
        )

        p4 = self._landmark_point(
            landmarks,
            eye_indexes[3],
            frame_width,
            frame_height
        )

        p5 = self._landmark_point(
            landmarks,
            eye_indexes[4],
            frame_width,
            frame_height
        )

        p6 = self._landmark_point(
            landmarks,
            eye_indexes[5],
            frame_width,
            frame_height
        )

        horizontal = self._distance(
            p1,
            p4
        )

        if horizontal <= 1e-6:
            return 0.0

        vertical_1 = self._distance(
            p2,
            p6
        )

        vertical_2 = self._distance(
            p3,
            p5
        )

        return (
            vertical_1 + vertical_2
        ) / (
            2.0 * horizontal
        )

    def _calculate_ear(
        self,
        landmarks,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float]:
        """
        Calculate raw and normalised eye openness.

        The model feature is clipped to 0-1 because the dataset
        appears to use a normalised eye feature.
        """

        left_ear = self._calculate_raw_ear(
            landmarks,
            self.LEFT_EYE,
            frame_width,
            frame_height
        )

        right_ear = self._calculate_raw_ear(
            landmarks,
            self.RIGHT_EYE,
            frame_width,
            frame_height
        )

        raw_ear = (
            left_ear + right_ear
        ) / 2.0

        # Approximate normalisation:
        # a raw EAR near 0.30 is treated as fully open.
        normalised_ear = float(
            np.clip(
                raw_ear / 0.30,
                0.0,
                1.0
            )
        )

        return raw_ear, normalised_ear

    def _calculate_yawn_score(
        self,
        landmarks,
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Calculate a normalised mouth-opening score.
        """

        mouth_left = self._landmark_point(
            landmarks,
            self.MOUTH_LEFT,
            frame_width,
            frame_height
        )

        mouth_right = self._landmark_point(
            landmarks,
            self.MOUTH_RIGHT,
            frame_width,
            frame_height
        )

        upper_lip = self._landmark_point(
            landmarks,
            self.UPPER_LIP,
            frame_width,
            frame_height
        )

        lower_lip = self._landmark_point(
            landmarks,
            self.LOWER_LIP,
            frame_width,
            frame_height
        )

        mouth_width = self._distance(
            mouth_left,
            mouth_right
        )

        if mouth_width <= 1e-6:
            return 0.0

        mouth_opening = self._distance(
            upper_lip,
            lower_lip
        )

        mouth_ratio = (
            mouth_opening / mouth_width
        )

        # Approximate normalisation:
        # a mouth ratio near 0.60 is treated as fully open.
        return float(
            np.clip(
                mouth_ratio / 0.60,
                0.0,
                1.0
            )
        )

    def _calculate_head_tilt(
        self,
        landmarks,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float]:
        """
        Calculate face roll from the eye line.

        Returns raw degrees and a 0-1 normalised feature.
        """

        left_eye = self._landmark_point(
            landmarks,
            self.LEFT_EYE_OUTER,
            frame_width,
            frame_height
        )

        right_eye = self._landmark_point(
            landmarks,
            self.RIGHT_EYE_OUTER,
            frame_width,
            frame_height
        )

        delta_y = float(
            right_eye[1] - left_eye[1]
        )

        delta_x = float(
            right_eye[0] - left_eye[0]
        )

        raw_degrees = abs(
            degrees(
                atan2(
                    delta_y,
                    delta_x
                )
            )
        )

        normalised_tilt = float(
            np.clip(
                raw_degrees / 45.0,
                0.0,
                1.0
            )
        )

        return raw_degrees, normalised_tilt

    def _update_blink_counter(
        self,
        normalised_ear: float
    ) -> float:
        """
        Track blink events and return a normalised blink-rate feature.
        """

        current_time = monotonic()

        while (
            self.blink_timestamps
            and current_time - self.blink_timestamps[0]
            > self.blink_window_seconds
        ):
            self.blink_timestamps.popleft()

        eyes_closed = (
            normalised_ear
            < self.blink_ear_threshold
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
                self.blink_timestamps.append(
                    current_time
                )

            self.closed_eye_frames = 0
            self.eye_was_closed = False

        blink_count = len(
            self.blink_timestamps
        )

        # The dataset contains fractional values, which suggests
        # blink_count was normalised. Thirty blinks per minute
        # is mapped to 1.0 for this initial integration.
        normalised_blink_count = float(
            np.clip(
                blink_count / 30.0,
                0.0,
                1.0
            )
        )

        return normalised_blink_count

    @staticmethod
    def _calculate_brightness(
        frame: np.ndarray
    ) -> float:
        """
        Calculate average grayscale brightness.
        """

        grayscale = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        return float(
            np.mean(grayscale)
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        features: Dict,
        raw_ear: float,
        raw_head_tilt: float,
        brightness: float
    ) -> np.ndarray:
        """
        Add extracted feature values to the frame.
        """

        annotated = frame.copy()

        lines = [
            f"EAR feature: {features['ear']:.3f}",
            f"Raw EAR: {raw_ear:.3f}",
            f"Yawn score: {features['yawn_score']:.3f}",
            f"Head tilt: {features['head_tilt']:.3f}",
            f"Raw tilt: {raw_head_tilt:.1f} deg",
            f"Hands: {features['hands_detected']}",
            f"Brightness: {brightness:.1f}",
            f"Condition: {features['condition']}",
            f"Blink feature: {features['blink_count']:.3f}"
        ]

        for index, line in enumerate(
            lines
        ):
            cv2.putText(
                annotated,
                line,
                (
                    10,
                    25 + index * 23
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated

    def process_frame(
        self,
        frame: np.ndarray
    ) -> VisionResult:
        """
        Process one BGR webcam frame.

        Returns
        -------
        VisionResult
            Eight model features and an annotated frame.
        """

        if frame is None:
            raise ValueError(
                "The supplied frame is None."
            )

        if frame.size == 0:
            raise ValueError(
                "The supplied frame is empty."
            )

        frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame.flags.writeable = False

        face_results = self.face_mesh.process(
            rgb_frame
        )

        hand_results = self.hands.process(
            rgb_frame
        )

        rgb_frame.flags.writeable = True

        brightness = self._calculate_brightness(
            frame
        )

        low_light = (
            brightness
            < self.low_light_threshold
        )

        condition = (
            "dark"
            if low_light
            else "none"
        )

        hands_detected = 0

        if hand_results.multi_hand_landmarks:
            hands_detected = 1

        if not face_results.multi_face_landmarks:
            annotated = frame.copy()

            cv2.putText(
                annotated,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            return VisionResult(
                features=None,
                annotated_frame=annotated,
                face_detected=False,
                status="No face detected"
            )

        face_landmarks = (
            face_results.multi_face_landmarks[0]
        )

        landmarks = face_landmarks.landmark

        raw_ear, ear = self._calculate_ear(
            landmarks,
            frame_width,
            frame_height
        )

        yawn_score = self._calculate_yawn_score(
            landmarks,
            frame_width,
            frame_height
        )

        raw_head_tilt, head_tilt = (
            self._calculate_head_tilt(
                landmarks,
                frame_width,
                frame_height
            )
        )

        blink_count = self._update_blink_counter(
            ear
        )

        # Legacy Face Mesh does not expose a direct per-frame
        # face score. Because landmarks were returned after the
        # configured detection threshold, use 1.0 initially.
        face_confidence = 1.0

        features = {
            "ear": ear,
            "yawn_score": yawn_score,
            "head_tilt": head_tilt,
            "hands_detected": float(
                hands_detected
            ),
            "condition": condition,
            "low_light": bool(
                low_light
            ),
            "face_confidence": face_confidence,
            "blink_count": blink_count
        }

        annotated = self._annotate_frame(
            frame=frame,
            features=features,
            raw_ear=raw_ear,
            raw_head_tilt=raw_head_tilt,
            brightness=brightness
        )

        return VisionResult(
            features=features,
            annotated_frame=annotated,
            face_detected=True,
            status="Face detected"
        )