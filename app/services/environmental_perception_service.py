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
        # Raw frame-level evidence is retained separately from the temporally
        # stabilised label. Environmental analysis runs roughly every 0.5 s,
        # so 8 samples represent about four seconds of recent evidence.
        self._occlusion_history: deque[dict[str, Any]] = deque(maxlen=8)
        self._stable_occlusion = "none"
        self._stable_occlusion_confidence = 0.72
        self._candidate_occlusion: str | None = None
        self._candidate_count = 0

    def reset_temporal_state(self) -> None:
        """Reset occlusion persistence at the start of a Monitoring session."""
        self._occlusion_history.clear()
        self._stable_occlusion = "none"
        self._stable_occlusion_confidence = 0.72
        self._candidate_occlusion = None
        self._candidate_count = 0

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

    def _stabilise_occlusion(
        self,
        raw_label: str,
        raw_confidence: float,
        lighting: str,
        metrics: dict[str, float],
    ) -> tuple[str, float, str]:
        """Apply persistence and hysteresis without hiding raw evidence."""

        raw_label = str(raw_label or "unknown")
        raw_confidence = self._clip(raw_confidence)
        item = {
            "label": raw_label,
            "confidence": raw_confidence,
            "lighting": lighting,
            "brightness_ratio": float(metrics.get("eye_region_brightness_ratio", 0.0) or 0.0),
            "dark_ratio": float(metrics.get("eye_dark_ratio", 0.0) or 0.0),
            "visibility": float(metrics.get("eye_visibility_score", 0.0) or 0.0),
        }
        self._occlusion_history.append(item)

        # "uncertain" is meaningful information about the current frame. It
        # must not instantly erase an already-established physical occlusion.
        if raw_label in {"unknown", "uncertain"}:
            if self._stable_occlusion not in {"none", "unknown", "uncertain"}:
                held_confidence = max(0.50, self._stable_occlusion_confidence * 0.92)
                self._stable_occlusion_confidence = held_confidence
                return (
                    self._stable_occlusion,
                    held_confidence,
                    "Temporal state held while the current frame is visually uncertain.",
                )
            return (
                "uncertain",
                max(0.35, raw_confidence),
                "Current lighting or visibility is insufficient for a stable physical-occlusion label.",
            )

        # Recent weighted support. Eye-occlusion evidence contributes partial
        # support to sunglasses because both indicate reduced eye visibility,
        # but only raw sunglasses frames can promote the specific sunglasses label.
        recent = list(self._occlusion_history)
        sunglasses_votes = sum(
            x["confidence"] for x in recent if x["label"] == "sunglasses"
        )
        eye_votes = sum(
            x["confidence"] for x in recent if x["label"] == "eye_occlusion"
        )
        none_votes = sum(
            x["confidence"] for x in recent if x["label"] == "none"
        )
        partial_votes = sum(
            x["confidence"] for x in recent if x["label"] == "partial_face"
        )

        # Candidate streak is deliberately separate from weighted history.
        if raw_label == self._candidate_occlusion:
            self._candidate_count += 1
        else:
            self._candidate_occlusion = raw_label
            self._candidate_count = 1

        previous = self._stable_occlusion

        # Partial face is visually distinct but still requires persistence.
        if raw_label == "partial_face":
            if self._candidate_count >= 2 or partial_votes >= 1.35:
                self._stable_occlusion = "partial_face"
                self._stable_occlusion_confidence = min(0.90, max(0.65, partial_votes / 2.0))
        elif raw_label == "sunglasses":
            # Specific sunglasses promotion requires persistent direct evidence,
            # not one lucky dark/glare frame.
            if (
                self._candidate_count >= 2
                and sunglasses_votes >= 1.35
            ) or sunglasses_votes >= 2.05:
                self._stable_occlusion = "sunglasses"
                support = sunglasses_votes + eye_votes * 0.20
                self._stable_occlusion_confidence = min(0.94, max(0.68, support / 3.0))
            elif previous in {"none", "uncertain", "unknown"}:
                # While waiting for sunglasses persistence, use the safer
                # generic eye-occlusion state if recent evidence supports it.
                if sunglasses_votes + eye_votes >= 1.40:
                    self._stable_occlusion = "eye_occlusion"
                    self._stable_occlusion_confidence = min(
                        0.84, max(0.58, (sunglasses_votes + eye_votes) / 3.0)
                    )
        elif raw_label == "eye_occlusion":
            # Do not immediately downgrade an established sunglasses state.
            if previous == "sunglasses":
                if eye_votes >= 2.20 and sunglasses_votes < 0.80:
                    self._stable_occlusion = "eye_occlusion"
                    self._stable_occlusion_confidence = min(0.82, max(0.58, eye_votes / 3.0))
            elif self._candidate_count >= 2 or eye_votes + sunglasses_votes >= 1.35:
                self._stable_occlusion = "eye_occlusion"
                self._stable_occlusion_confidence = min(
                    0.84, max(0.58, (eye_votes + sunglasses_votes * 0.65) / 3.0)
                )
        elif raw_label == "none":
            # Clearing an established physical occlusion requires stronger
            # persistence than entering it. This is the hysteresis that stops
            # one bright/reflection frame from snapping back to "none".
            required_none_streak = (
                4
                if previous in {"sunglasses", "partial_face"}
                else 3
            )
            required_none_votes = (
                2.60
                if previous in {"sunglasses", "partial_face"}
                else 1.85
            )
            if self._candidate_count >= required_none_streak and none_votes >= required_none_votes:
                self._stable_occlusion = "none"
                self._stable_occlusion_confidence = min(0.90, max(0.65, none_votes / 4.0))
            elif previous in {"none", "unknown", "uncertain"}:
                self._stable_occlusion = "none"
                self._stable_occlusion_confidence = max(0.62, raw_confidence)

        stable = self._stable_occlusion
        confidence = self._clip(self._stable_occlusion_confidence)

        if stable == "sunglasses":
            summary = "Persistent recent eye-region evidence is consistent with sunglasses or dark bilateral eye covering."
        elif stable == "eye_occlusion":
            summary = "Persistent recent evidence indicates reduced eye visibility without enough specificity for a sunglasses label."
        elif stable == "partial_face":
            summary = "Persistent recent landmarks indicate that part of the face is close to or outside the camera boundary."
        elif stable == "none":
            summary = "Recent frames do not support a persistent eye-region occlusion."
        else:
            summary = "Temporal occlusion evidence remains uncertain."

        return stable, confidence, summary

    def _analyse_occlusion(
        self,
        gray: Any,
        face_landmarks: Any | None,
        lighting: str,
        cv2: Any,
    ) -> dict[str, Any]:
        if face_landmarks is None:
            return {
                "raw_automatic_occlusion": "unknown",
                "raw_automatic_occlusion_confidence": 0.0,
                "raw_automatic_occlusion_summary": "No face landmarks are available for occlusion analysis.",
                "automatic_occlusion": self._stable_occlusion,
                "automatic_occlusion_confidence": round(
                    self._stable_occlusion_confidence * 0.90, 4
                ),
                "automatic_occlusion_summary": (
                    "No face landmarks are available; the last stable occlusion state is being held temporarily."
                ),
                "occlusion_temporal_window": len(self._occlusion_history),
                "eye_visibility_score": 0.0,
                "eye_region_brightness_ratio": 0.0,
                "eye_dark_ratio": 0.0,
                "eye_edge_density": 0.0,
            }

        height, width = gray.shape[:2]
        points = [
            self._landmark_xy(face_landmarks, i, width, height)
            for i in range(min(468, len(face_landmarks.landmark)))
        ]
        xs = sorted(p[0] for p in points)
        ys = sorted(p[1] for p in points)

        # Use a robust face box for reference patches so one outlying landmark
        # cannot stretch the face box to the image boundary.
        trim_x = max(0, int(len(xs) * 0.02))
        trim_y = max(0, int(len(ys) * 0.02))
        x_low = xs[min(trim_x, len(xs) - 1)]
        x_high = xs[max(0, len(xs) - 1 - trim_x)]
        y_low = ys[min(trim_y, len(ys) - 1)]
        y_high = ys[max(0, len(ys) - 1 - trim_y)]
        bbox = (
            max(0, x_low),
            max(0, y_low),
            min(width, x_high),
            min(height, y_high),
        )
        bx1, by1, bx2, by2 = bbox

        # Partial-face evidence must come from stable anatomical boundary
        # anchors, not the minimum/maximum of every FaceMesh point.
        boundary_indices = (10, 152, 234, 454)  # forehead, chin, left/right cheek
        boundary_points = [
            self._landmark_xy(face_landmarks, idx, width, height)
            for idx in boundary_indices
            if idx < len(face_landmarks.landmark)
        ]
        margin_x, margin_y = width * 0.012, height * 0.012
        clipped = any(
            x <= margin_x
            or y <= margin_y
            or x >= width - 1 - margin_x
            or y >= height - 1 - margin_y
            for x, y in boundary_points
        )

        left = self._eye_roi(gray, face_landmarks, self.LEFT_EYE)
        right = self._eye_roi(gray, face_landmarks, self.RIGHT_EYE)
        cheek_l = self._patch(gray, 0.12, 0.48, 0.34, 0.68, bbox)
        cheek_r = self._patch(gray, 0.66, 0.48, 0.88, 0.68, bbox)

        if left is None or right is None or cheek_l is None or cheek_r is None:
            label, confidence = "uncertain", 0.35
            summary = (
                "Eye-region extraction produced an invalid or undersized ROI; "
                "occlusion cannot be classified from this frame."
            )
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

            visibility = self._clip(
                1.0 - max(sunglasses_strength * 0.75, dark_ratio * 0.45)
            )
            exposure_unreliable = lighting in {"too_dark", "too_bright", "glare"}

            # Strong clear-eye evidence is allowed to override a global glare
            # flag. V8.3.3 incorrectly returned "uncertain" for frames with
            # visibility ~1.0, darkness ~0 and a normal eye/cheek ratio.
            clear_eye_evidence = (
                visibility >= 0.82
                and 0.76 <= ratio <= 1.35
                and dark_ratio <= 0.22
                and symmetry >= 0.35
            )

            sunglasses_evidence = (
                ratio < 0.58
                and dark_ratio > 0.48
                and symmetry > 0.55
                and cheek_median > 55
            )

            generic_occlusion_evidence = (
                ratio < 0.72
                and dark_ratio > 0.34
                and symmetry > 0.40
            )

            if clipped:
                label = "partial_face"
                confidence = 0.72
                summary = (
                    "Stable facial boundary landmarks indicate that part of "
                    "the face is close to the camera boundary."
                )
            elif clear_eye_evidence:
                label = "none"
                confidence = max(
                    0.70,
                    min(
                        0.94,
                        0.76
                        + (visibility - 0.82) * 0.45
                        + max(0.0, 0.22 - dark_ratio) * 0.18,
                    ),
                )
                summary = (
                    "Eye visibility is strong and the eye-region measurements "
                    "support a clear, unobstructed face."
                )
            elif sunglasses_evidence:
                # Strong bilateral dark-eye evidence is still measurable under
                # moderate glare. Global exposure lowers confidence rather than
                # erasing the physical evidence.
                label = "sunglasses"
                confidence = max(
                    0.68,
                    min(
                        0.93,
                        0.70
                        + sunglasses_strength * 0.25
                        - (0.08 if exposure_unreliable else 0.0),
                    ),
                )
                summary = (
                    "Both eye regions are persistently much darker than the "
                    "visible face, consistent with sunglasses or dark eye covering."
                )
            elif generic_occlusion_evidence:
                label = "eye_occlusion"
                confidence = max(
                    0.56,
                    min(
                        0.80,
                        0.52
                        + sunglasses_strength * 0.28
                        - (0.06 if exposure_unreliable else 0.0),
                    ),
                )
                summary = (
                    "Eye visibility is reduced, but the image is not specific "
                    "enough to label the covering as sunglasses."
                )
            elif exposure_unreliable:
                label = "uncertain"
                confidence = 0.45
                summary = (
                    "Eye-region evidence is ambiguous while lighting is "
                    f"{lighting.replace('_', ' ')}."
                )
            else:
                label = "none"
                confidence = max(
                    0.62,
                    min(
                        0.90,
                        0.72
                        + (ratio - 0.72) * 0.30
                        - max(0.0, dark_ratio - 0.25) * 0.15,
                    ),
                )
                summary = "No strong automatic eye-region occlusion pattern is detected."
            result_metrics = {
                "eye_visibility_score": round(visibility, 4),
                "eye_region_brightness_ratio": round(ratio, 4),
                "eye_dark_ratio": round(dark_ratio, 4),
                "eye_edge_density": round(edge_density, 4),
            }

            raw_label = label
            raw_confidence = confidence
            raw_summary = summary
            stable_label, stable_confidence, stable_summary = self._stabilise_occlusion(
                raw_label,
                raw_confidence,
                lighting,
                result_metrics,
            )

            return {
                "raw_automatic_occlusion": raw_label,
                "raw_automatic_occlusion_confidence": round(raw_confidence, 4),
                "raw_automatic_occlusion_summary": raw_summary,
                "automatic_occlusion": stable_label,
                "automatic_occlusion_confidence": round(stable_confidence, 4),
                "automatic_occlusion_summary": stable_summary,
                "occlusion_temporal_window": len(self._occlusion_history),
                **result_metrics,
            }

        result_metrics = {
            "eye_visibility_score": 0.0,
            "eye_region_brightness_ratio": 0.0,
            "eye_dark_ratio": 0.0,
            "eye_edge_density": 0.0,
        }
        stable_label, stable_confidence, stable_summary = self._stabilise_occlusion(
            label,
            confidence,
            lighting,
            result_metrics,
        )
        return {
            "raw_automatic_occlusion": label,
            "raw_automatic_occlusion_confidence": round(confidence, 4),
            "raw_automatic_occlusion_summary": summary,
            "automatic_occlusion": stable_label,
            "automatic_occlusion_confidence": round(stable_confidence, 4),
            "automatic_occlusion_summary": stable_summary,
            "occlusion_temporal_window": len(self._occlusion_history),
            **result_metrics,
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
            "raw_automatic_occlusion": "unknown",
            "raw_automatic_occlusion_confidence": 0.0,
            "raw_automatic_occlusion_summary": "Start Monitoring to analyse eye visibility.",
            "automatic_occlusion": "unknown",
            "automatic_occlusion_confidence": 0.0,
            "automatic_occlusion_summary": "Start Monitoring to analyse eye visibility.",
            "occlusion_temporal_window": 0,
            "eye_visibility_score": 0.0,
            "eye_region_brightness_ratio": 0.0,
            "eye_dark_ratio": 0.0,
            "eye_edge_density": 0.0,
        }
