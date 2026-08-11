from __future__ import annotations

from collections import deque
from typing import Any


class EnvironmentalPerceptionService:
    """Read-only camera quality and conservative occlusion perception.

    The service never changes fatigue predictions, calibration, temporal state,
    alerts or camera behaviour. Automatic occlusion is deliberately conservative:
    it can identify strong eye-region obstruction patterns (especially dark,
    symmetric sunglasses-like coverage) but it does not pretend to be a general
    glasses/hat classifier without dedicated training images.
    """

    LEFT_EYE = (33, 133, 159, 145, 160, 144)
    RIGHT_EYE = (362, 263, 386, 374, 385, 380)

    def __init__(self) -> None:
        self._occlusion_history: deque[tuple[str, float]] = deque(maxlen=5)

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _landmark_xy(face_landmarks: Any, index: int, width: int, height: int) -> tuple[int, int]:
        point = face_landmarks.landmark[index]
        return (
            int(max(0, min(width - 1, point.x * width))),
            int(max(0, min(height - 1, point.y * height))),
        )

    @classmethod
    def _eye_roi(cls, gray: Any, face_landmarks: Any, indices: tuple[int, ...]) -> Any | None:
        height, width = gray.shape[:2]
        points = [cls._landmark_xy(face_landmarks, idx, width, height) for idx in indices]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        eye_width = max(8, max(xs) - min(xs))
        eye_height = max(5, max(ys) - min(ys))
        pad_x = max(4, int(eye_width * 0.32))
        pad_y = max(4, int(max(eye_height * 0.9, eye_width * 0.18)))
        x1 = max(0, min(xs) - pad_x)
        x2 = min(width, max(xs) + pad_x + 1)
        y1 = max(0, min(ys) - pad_y)
        y2 = min(height, max(ys) + pad_y + 1)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        return gray[y1:y2, x1:x2]

    @staticmethod
    def _patch(gray: Any, x1: float, y1: float, x2: float, y2: float, bbox: tuple[int, int, int, int]) -> Any | None:
        bx1, by1, bx2, by2 = bbox
        fw, fh = max(1, bx2 - bx1), max(1, by2 - by1)
        xx1, xx2 = int(bx1 + fw * x1), int(bx1 + fw * x2)
        yy1, yy2 = int(by1 + fh * y1), int(by1 + fh * y2)
        xx1, yy1 = max(0, xx1), max(0, yy1)
        xx2, yy2 = min(gray.shape[1], xx2), min(gray.shape[0], yy2)
        if xx2 - xx1 < 5 or yy2 - yy1 < 5:
            return None
        return gray[yy1:yy2, xx1:xx2]

    def _analyse_occlusion(
        self,
        gray: Any,
        face_landmarks: Any | None,
        lighting: str,
        cv2: Any,
    ) -> dict[str, Any]:
        if face_landmarks is None:
            return {
                "automatic_occlusion": "unknown",
                "automatic_occlusion_confidence": 0.0,
                "automatic_occlusion_summary": "No face landmarks are available for occlusion analysis.",
                "eye_visibility_score": 0.0,
                "eye_region_brightness_ratio": 0.0,
                "eye_dark_ratio": 0.0,
            }

        if lighting in {"too_dark", "too_bright", "glare"}:
            # Do not confuse exposure problems with physical occlusion.
            label = "uncertain"
            confidence = 0.45
            summary = "Eye occlusion cannot be classified reliably under the current exposure conditions."
            self._occlusion_history.append((label, confidence))
            return {
                "automatic_occlusion": label,
                "automatic_occlusion_confidence": confidence,
                "automatic_occlusion_summary": summary,
                "eye_visibility_score": 0.45,
                "eye_region_brightness_ratio": 0.0,
                "eye_dark_ratio": 0.0,
            }

        height, width = gray.shape[:2]
        points = [
            self._landmark_xy(face_landmarks, i, width, height)
            for i in range(min(468, len(face_landmarks.landmark)))
        ]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        bbox = (max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys)))
        bx1, by1, bx2, by2 = bbox
        face_w, face_h = max(1, bx2 - bx1), max(1, by2 - by1)

        # If the landmarked face touches the image border, part of the face may
        # simply be outside the camera frame rather than covered by an object.
        margin_x, margin_y = width * 0.018, height * 0.018
        clipped = bx1 <= margin_x or by1 <= margin_y or bx2 >= width - margin_x or by2 >= height - margin_y

        left = self._eye_roi(gray, face_landmarks, self.LEFT_EYE)
        right = self._eye_roi(gray, face_landmarks, self.RIGHT_EYE)
        cheek_l = self._patch(gray, 0.12, 0.48, 0.34, 0.68, bbox)
        cheek_r = self._patch(gray, 0.66, 0.48, 0.88, 0.68, bbox)

        if left is None or right is None or cheek_l is None or cheek_r is None:
            label, confidence = "uncertain", 0.35
            summary = "The eye/face regions are too small for reliable occlusion analysis."
        else:
            import numpy as np
            eye_pixels = np.concatenate([left.reshape(-1), right.reshape(-1)])
            cheek_pixels = np.concatenate([cheek_l.reshape(-1), cheek_r.reshape(-1)])
            eye_median = float(np.median(eye_pixels))
            cheek_median = max(1.0, float(np.median(cheek_pixels)))
            ratio = eye_median / cheek_median
            dark_threshold = max(38.0, min(90.0, cheek_median * 0.55))
            dark_ratio = float((eye_pixels < dark_threshold).mean())
            left_median = float(np.median(left))
            right_median = float(np.median(right))
            symmetry = 1.0 - min(1.0, abs(left_median - right_median) / 55.0)

            # Edge density helps distinguish a genuinely covered/dark eye region
            # from a uniformly dim frame, without claiming clear-glasses detection.
            left_edges = cv2.Canny(left, 45, 110)
            right_edges = cv2.Canny(right, 45, 110)
            edge_density = float(((left_edges > 0).mean() + (right_edges > 0).mean()) / 2.0)

            sunglasses_strength = self._clip(
                ((0.70 - ratio) / 0.28) * 0.48
                + ((dark_ratio - 0.28) / 0.45) * 0.36
                + symmetry * 0.16
            )

            if clipped:
                label = "partial_face"
                confidence = 0.72
                summary = "Part of the landmarked face is close to the camera boundary."
            elif ratio < 0.58 and dark_ratio > 0.48 and symmetry > 0.55 and cheek_median > 55:
                label = "sunglasses"
                confidence = max(0.72, min(0.95, 0.70 + sunglasses_strength * 0.25))
                summary = "Both eye regions are substantially darker than the visible face, consistent with sunglasses or dark eye covering."
            elif ratio < 0.72 and dark_ratio > 0.34 and symmetry > 0.40:
                label = "eye_occlusion"
                confidence = max(0.58, min(0.82, 0.52 + sunglasses_strength * 0.28))
                summary = "Eye visibility is reduced, but the image is not specific enough to label the covering as sunglasses."
            else:
                label = "none"
                confidence = max(0.62, min(0.90, 0.72 + (ratio - 0.72) * 0.30 - max(0.0, dark_ratio - 0.25) * 0.15))
                summary = "No strong automatic eye-region occlusion pattern is detected."

            visibility = self._clip(1.0 - max(sunglasses_strength * 0.75, dark_ratio * 0.45))
            result_metrics = {
                "eye_visibility_score": round(visibility, 4),
                "eye_region_brightness_ratio": round(ratio, 4),
                "eye_dark_ratio": round(dark_ratio, 4),
                "eye_edge_density": round(edge_density, 4),
            }

            self._occlusion_history.append((label, confidence))
            # Stabilise labels across a short 2–3 second window. Strong
            # sunglasses/partial-face detections can still promote immediately.
            if label not in {"sunglasses", "partial_face"} and len(self._occlusion_history) >= 3:
                weighted: dict[str, float] = {}
                for history_label, history_confidence in self._occlusion_history:
                    weighted[history_label] = weighted.get(history_label, 0.0) + history_confidence
                stable = max(weighted, key=weighted.get)
                if stable != label and weighted[stable] >= weighted.get(label, 0.0) * 1.25:
                    label = stable
                    confidence = min(confidence, 0.78)
                    if stable == "none":
                        summary = "Recent frames do not support a persistent eye-region occlusion."
                    elif stable == "eye_occlusion":
                        summary = "Recent frames indicate persistent reduced eye visibility."

            return {
                "automatic_occlusion": label,
                "automatic_occlusion_confidence": round(confidence, 4),
                "automatic_occlusion_summary": summary,
                **result_metrics,
            }

        self._occlusion_history.append((label, confidence))
        return {
            "automatic_occlusion": label,
            "automatic_occlusion_confidence": round(confidence, 4),
            "automatic_occlusion_summary": summary,
            "eye_visibility_score": 0.0,
            "eye_region_brightness_ratio": 0.0,
            "eye_dark_ratio": 0.0,
            "eye_edge_density": 0.0,
        }

    def analyse(self, frame: Any, cv2: Any, face_landmarks: Any | None = None) -> dict[str, Any]:
        if frame is None:
            return self.empty()

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness, contrast = cv2.meanStdDev(gray)
            brightness = float(mean_brightness[0][0])
            contrast_value = float(contrast[0][0])

            total = max(1, int(gray.size))
            underexposed = float((gray < 35).sum()) / total
            overexposed = float((gray > 225).sum()) / total
            blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Highlights in the upper central region are a glare proxy only.
            height, width = gray.shape[:2]
            y1, y2 = int(height * 0.08), int(height * 0.58)
            x1, x2 = int(width * 0.18), int(width * 0.82)
            upper_region = gray[y1:y2, x1:x2]
            region_total = max(1, int(upper_region.size))
            glare_ratio = float((upper_region > 238).sum()) / region_total

            if brightness < 45 or underexposed > 0.55:
                lighting = "too_dark"
            elif brightness > 205 or overexposed > 0.42:
                lighting = "too_bright"
            elif glare_ratio > 0.16:
                lighting = "glare"
            else:
                lighting = "normal"

            if blur_variance < 38:
                sharpness = "blurred"
            elif blur_variance < 85:
                sharpness = "moderate"
            else:
                sharpness = "sharp"

            penalties = 0.0
            penalties += min(0.35, underexposed * 0.55)
            penalties += min(0.30, overexposed * 0.50)
            penalties += min(0.20, glare_ratio * 0.90)
            if blur_variance < 38:
                penalties += 0.28
            elif blur_variance < 85:
                penalties += 0.12
            if contrast_value < 22:
                penalties += 0.15

            quality_score = max(0.0, min(1.0, 1.0 - penalties))
            if quality_score >= 0.80:
                quality = "good"
            elif quality_score >= 0.58:
                quality = "moderate"
            else:
                quality = "limited"

            reasons: list[str] = []
            if lighting == "too_dark":
                reasons.append("cabin image is underexposed")
            elif lighting == "too_bright":
                reasons.append("cabin image is overexposed")
            elif lighting == "glare":
                reasons.append("strong highlights may reduce visibility")
            if sharpness == "blurred":
                reasons.append("image is blurred")
            if contrast_value < 22:
                reasons.append("image contrast is low")

            occlusion = self._analyse_occlusion(gray, face_landmarks, lighting, cv2)
            if occlusion["automatic_occlusion"] in {"sunglasses", "eye_occlusion", "partial_face"}:
                penalties += min(0.18, (1.0 - occlusion.get("eye_visibility_score", 1.0)) * 0.18)
                quality_score = max(0.0, min(1.0, 1.0 - penalties))
                if quality_score < 0.58:
                    quality = "limited"
                elif quality_score < 0.80:
                    quality = "moderate"
                reasons.append(occlusion["automatic_occlusion_summary"].rstrip("."))

            summary = (
                "Camera image quality is suitable for contextual analysis."
                if not reasons
                else "; ".join(reasons).capitalize() + "."
            )

            return {
                "environment_available": True,
                "frame_brightness": round(brightness, 2),
                "frame_contrast": round(contrast_value, 2),
                "frame_blur_variance": round(blur_variance, 2),
                "underexposed_ratio": round(underexposed, 4),
                "overexposed_ratio": round(overexposed, 4),
                "glare_ratio": round(glare_ratio, 4),
                "automatic_cabin_light": lighting,
                "automatic_sharpness": sharpness,
                "automatic_perception_quality": quality,
                "automatic_perception_score": round(quality_score, 4),
                "automatic_perception_summary": summary,
                **occlusion,
            }
        except Exception as error:
            result = self.empty()
            result["automatic_perception_summary"] = (
                f"Environmental analysis unavailable: {type(error).__name__}."
            )
            return result

    @staticmethod
    def empty() -> dict[str, Any]:
        return {
            "environment_available": False,
            "frame_brightness": 0.0,
            "frame_contrast": 0.0,
            "frame_blur_variance": 0.0,
            "underexposed_ratio": 0.0,
            "overexposed_ratio": 0.0,
            "glare_ratio": 0.0,
            "automatic_cabin_light": "unknown",
            "automatic_sharpness": "unknown",
            "automatic_perception_quality": "standby",
            "automatic_perception_score": 0.0,
            "automatic_perception_summary": "Start Monitoring to analyse camera image quality.",
            "automatic_occlusion": "unknown",
            "automatic_occlusion_confidence": 0.0,
            "automatic_occlusion_summary": "Start Monitoring to analyse eye visibility.",
            "eye_visibility_score": 0.0,
            "eye_region_brightness_ratio": 0.0,
            "eye_dark_ratio": 0.0,
            "eye_edge_density": 0.0,
        }
