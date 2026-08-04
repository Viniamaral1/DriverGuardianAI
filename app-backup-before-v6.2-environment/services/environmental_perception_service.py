from __future__ import annotations

from typing import Any


class EnvironmentalPerceptionService:
    """Read-only image-quality analysis for the existing camera frame.

    The output is contextual metadata only. It never changes fatigue
    predictions, calibration, temporal state, alerts or camera behaviour.
    """

    def analyse(self, frame: Any, cv2: Any) -> dict[str, Any]:
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

            # Highlights in the upper central region are a conservative glare
            # proxy. This is not eye or sunglasses recognition.
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
            "automatic_perception_summary": (
                "Start Monitoring to analyse camera image quality."
            ),
        }
