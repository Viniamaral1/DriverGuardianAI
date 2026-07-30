"""DriverGuardianAI V3 calibrated real-time dashboard."""
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

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "v2" / "ablation" / "driver_guardian_core_behaviour.joblib"
LOG_DIR = ROOT / "logs" / "v3"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP, LOWER_LIP, LEFT_MOUTH, RIGHT_MOUTH = 13, 14, 78, 308
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263

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


def draw_text(img, text, pos, scale=0.55, colour=WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def draw_panel(img, tl, br, colour=PANEL):
    cv2.rectangle(img, tl, br, colour, -1)
    cv2.rectangle(img, tl, br, BORDER, 1)


def state_colour(state: str):
    return {"CRITICAL": RED, "WARNING": AMBER, "MONITORING": GREEN, "CALIBRATING": CYAN}.get(state, MUTED)


def draw_metric_card(img, x, y, width, title, value, subtitle):
    draw_panel(img, (x, y), (x + width, y + 88), PANEL_ALT)
    draw_text(img, title.upper(), (x + 14, y + 24), 0.40, MUTED)
    draw_text(img, value, (x + 14, y + 55), 0.78, WHITE, 2)
    draw_text(img, subtitle, (x + 14, y + 77), 0.34, MUTED)


def draw_probability_gauge(img, centre, radius, probability):
    p = float(np.clip(probability, 0.0, 1.0))
    start, end = 145, 395
    cv2.ellipse(img, centre, (radius, radius), 0, start, end, BORDER, 14, cv2.LINE_AA)
    colour = GREEN if p < 0.55 else AMBER if p < 0.80 else RED
    angle = start + int(p * (end - start))
    cv2.ellipse(img, centre, (radius, radius), 0, start, angle, colour, 14, cv2.LINE_AA)
    text = f"{p * 100:.0f}%"
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
    draw_text(img, text, (centre[0] - size[0] // 2, centre[1] + 8), 1.1, WHITE, 2)
    draw_text(img, "DECISION RISK", (centre[0] - 62, centre[1] + 38), 0.40, MUTED)


def draw_timeline(img, values, x, y, width, height):
    draw_panel(img, (x, y), (x + width, y + height), PANEL)
    draw_text(img, "CALIBRATED FATIGUE RISK - RECENT HISTORY", (x + 18, y + 27), 0.43, WHITE)
    px, py, pw, ph = x + 18, y + 42, width - 36, height - 60
    cv2.rectangle(img, (px, py), (px + pw, py + ph), PANEL_ALT, -1)
    for threshold, colour in ((0.65, AMBER), (0.82, RED)):
        ty = py + ph - int(ph * threshold)
        cv2.line(img, (px, ty), (px + pw, ty), colour, 1)
    history = list(values)
    if len(history) < 2:
        return
    points = []
    for i, value in enumerate(history):
        x_i = px + int(i * pw / max(1, len(history) - 1))
        y_i = py + ph - int(np.clip(value, 0.0, 1.0) * ph)
        points.append((x_i, y_i))
    cv2.polylines(img, [np.asarray(points, dtype=np.int32)], False, CYAN, 2, cv2.LINE_AA)


def landmark_point(landmarks, index, width, height):
    p = landmarks[index]
    return np.array([p.x * width, p.y * height], dtype=np.float32)


def calculate_ear(landmarks, indices, width, height):
    p1, p2, p3, p4, p5, p6 = [landmark_point(landmarks, i, width, height) for i in indices]
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal <= 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


def extract_features(frame, face_mesh):
    h, w = frame.shape[:2]
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return {"ear": 0.0, "yawn_score": 0.0, "head_tilt": 0.0, "face_detected": 0.0}, None

    face_landmarks = result.multi_face_landmarks[0]
    landmarks = face_landmarks.landmark
    ear = (calculate_ear(landmarks, LEFT_EYE, w, h) + calculate_ear(landmarks, RIGHT_EYE, w, h)) / 2.0
    upper = landmark_point(landmarks, UPPER_LIP, w, h)
    lower = landmark_point(landmarks, LOWER_LIP, w, h)
    left = landmark_point(landmarks, LEFT_MOUTH, w, h)
    right = landmark_point(landmarks, RIGHT_MOUTH, w, h)
    horizontal = np.linalg.norm(left - right)
    yawn = float(np.linalg.norm(upper - lower) / horizontal) if horizontal > 1e-6 else 0.0
    le = landmark_point(landmarks, LEFT_EYE_OUTER, w, h)
    re = landmark_point(landmarks, RIGHT_EYE_OUTER, w, h)
    delta = re - le
    tilt = abs(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))

    return {
        "ear": float(np.clip(ear, 0.0, 1.0)),
        "yawn_score": float(np.clip(yawn, 0.0, 2.0)),
        "head_tilt": float(np.clip(tilt, 0.0, 90.0)),
        "face_detected": 1.0,
    }, face_landmarks


def load_model(path: Path):
    if not path.exists():
        return None, 0.64, f"Model not found: {path}"
    try:
        bundle = joblib.load(path)
        threshold = 0.64
        estimator = bundle
        if isinstance(bundle, dict):
            estimator = bundle.get("model") or bundle.get("pipeline") or bundle.get("estimator") or bundle.get("classifier")
            threshold = float(bundle.get("threshold", bundle.get("selected_threshold", bundle.get("fatigue_threshold", threshold))))
        if estimator is None:
            return None, threshold, "No model or pipeline found in the bundle."
        if not hasattr(estimator, "predict_proba"):
            return None, threshold, "Loaded estimator does not implement predict_proba()."
        return estimator, threshold, None
    except Exception as exc:
        return None, 0.64, f"{type(exc).__name__}: {exc}"


def heuristic_probability(features):
    ear_risk = np.clip((0.27 - features["ear"]) / 0.12, 0.0, 1.0)
    yawn_risk = np.clip(features["yawn_score"] / 0.45, 0.0, 1.0)
    tilt_risk = np.clip(features["head_tilt"] / 25.0, 0.0, 1.0)
    return float(np.clip(0.68 * ear_risk + 0.22 * yawn_risk + 0.10 * tilt_risk, 0.0, 1.0))


def predict_probability(estimator, features):
    if estimator is None:
        return heuristic_probability(features)
    X = pd.DataFrame([{
        "ear": features["ear"],
        "yawn_score": features["yawn_score"],
        "head_tilt": features["head_tilt"],
    }])
    return float(np.clip(estimator.predict_proba(X)[0, 1], 0.0, 1.0))


class PersonalCalibration:
    def __init__(self, required_seconds=10.0, minimum_samples=80):
        self.required_seconds = required_seconds
        self.minimum_samples = minimum_samples
        self.reset()

    def reset(self):
        self.started_at = None
        self.ear_samples = []
        self.yawn_samples = []
        self.tilt_samples = []
        self.complete = False
        self.baseline_ear = 0.25
        self.baseline_yawn = 0.0
        self.baseline_tilt = 0.0

    def update(self, features, now):
        if self.complete or not features["face_detected"]:
            return
        if self.started_at is None:
            self.started_at = now
        ear = features["ear"]
        if 0.08 <= ear <= 0.45:
            self.ear_samples.append(ear)
            self.yawn_samples.append(features["yawn_score"])
            self.tilt_samples.append(features["head_tilt"])
        if now - self.started_at >= self.required_seconds and len(self.ear_samples) >= self.minimum_samples:
            self.baseline_ear = float(np.median(self.ear_samples))
            self.baseline_yawn = float(np.median(self.yawn_samples))
            self.baseline_tilt = float(np.median(self.tilt_samples))
            self.complete = True
            print("\nPersonal calibration completed.")
            print(f"Baseline EAR:       {self.baseline_ear:.4f}")
            print(f"Baseline yawn:      {self.baseline_yawn:.4f}")
            print(f"Baseline head tilt: {self.baseline_tilt:.2f}")

    def elapsed(self, now):
        return 0.0 if self.started_at is None else now - self.started_at

    def remaining(self, now):
        return max(0.0, self.required_seconds - self.elapsed(now))

    def calculate_fused_probability(self, model_probability, features):
        if not self.complete:
            return 0.0, {"model_risk": 0.0, "eye_risk": 0.0, "yawn_risk": 0.0, "tilt_risk": 0.0}

        ear_drop_ratio = (self.baseline_ear - features["ear"]) / max(self.baseline_ear, 1e-6)
        eye_risk = float(np.clip((ear_drop_ratio - 0.08) / 0.32, 0.0, 1.0))
        yawn_change = max(0.0, features["yawn_score"] - self.baseline_yawn)
        yawn_risk = float(np.clip(yawn_change / 0.18, 0.0, 1.0))
        tilt_change = abs(features["head_tilt"] - self.baseline_tilt)
        tilt_risk = float(np.clip((tilt_change - 4.0) / 18.0, 0.0, 1.0))
        model_risk = float(np.clip(model_probability, 0.0, 1.0))
        support = max(eye_risk, yawn_risk, tilt_risk)
        model_weight = 0.35 if support >= 0.35 else 0.12
        fused = float(np.clip(model_weight * model_risk + 0.58 * eye_risk + 0.22 * yawn_risk + 0.08 * tilt_risk, 0.0, 1.0))
        return fused, {"model_risk": model_risk, "eye_risk": eye_risk, "yawn_risk": yawn_risk, "tilt_risk": tilt_risk}


class TemporalStateEngine:
    def __init__(self):
        self.window = deque(maxlen=12)
        self.reset()

    def reset(self):
        self.window.clear()
        self.state = "MONITORING"
        self.warning_counter = 0
        self.critical_counter = 0
        self.release_counter = 0
        self.alert_count = 0

    def update(self, probability, face_detected, calibration_complete):
        if not calibration_complete:
            self.state = "CALIBRATING"
            self.warning_counter = self.critical_counter = self.release_counter = 0
            return self.state, 0.0
        if not face_detected:
            self.state = "NO FACE"
            self.warning_counter = self.critical_counter = self.release_counter = 0
            return self.state, probability

        self.window.append(probability)
        smoothed = float(np.mean(self.window))
        previous = self.state

        if smoothed >= 0.82:
            self.critical_counter += 1
            self.warning_counter += 1
            self.release_counter = 0
        elif smoothed >= 0.65:
            self.warning_counter += 1
            self.critical_counter = 0
            self.release_counter = 0
        elif smoothed <= 0.45:
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
        elif self.state in {"NO FACE", "CALIBRATING"}:
            self.state = "MONITORING"

        if previous != "CRITICAL" and self.state == "CRITICAL":
            self.alert_count += 1
        return self.state, smoothed


class SessionLogger:
    def __init__(self):
        self.enabled = False
        self.file = None
        self.writer = None
        self.path = None

    def toggle(self):
        if self.enabled:
            self.stop()
            return False
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"driver_guardian_v3_{stamp}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        fields = [
            "timestamp", "elapsed_seconds", "ear", "yawn_score", "head_tilt",
            "baseline_ear", "baseline_yawn", "baseline_tilt",
            "raw_model_probability", "decision_probability", "smoothed_probability",
            "model_risk", "eye_risk", "yawn_risk", "tilt_risk",
            "temporal_state", "calibration_complete"
        ]
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.enabled = True
        return True

    def stop(self):
        if self.file is not None:
            self.file.close()
        self.enabled = False
        self.file = None
        self.writer = None

    def write(self, elapsed, features, calibration, raw_p, decision_p, smoothed_p, evidence, state):
        if not self.enabled or self.writer is None:
            return
        self.writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_seconds": round(elapsed, 3),
            "ear": round(features["ear"], 6),
            "yawn_score": round(features["yawn_score"], 6),
            "head_tilt": round(features["head_tilt"], 6),
            "baseline_ear": round(calibration.baseline_ear, 6),
            "baseline_yawn": round(calibration.baseline_yawn, 6),
            "baseline_tilt": round(calibration.baseline_tilt, 6),
            "raw_model_probability": round(raw_p, 6),
            "decision_probability": round(decision_p, 6),
            "smoothed_probability": round(smoothed_p, 6),
            "model_risk": round(evidence["model_risk"], 6),
            "eye_risk": round(evidence["eye_risk"], 6),
            "yawn_risk": round(evidence["yawn_risk"], 6),
            "tilt_risk": round(evidence["tilt_risk"], 6),
            "temporal_state": state,
            "calibration_complete": int(calibration.complete),
        })
        self.file.flush()


def build_dashboard(frame, features, raw_p, decision_p, smoothed_p, state, history, fps, elapsed, alerts, logging_enabled, model_loaded, calibration, evidence):
    canvas = np.full((900, 1440, 3), BACKGROUND, dtype=np.uint8)
    draw_text(canvas, "DRIVERGUARDIAN AI", (36, 48), 1.0, WHITE, 2)
    draw_text(canvas, "V3 CALIBRATED REAL-TIME MONITORING", (37, 73), 0.42, CYAN)
    draw_text(canvas, f"{fps:.1f} FPS  |  {elapsed:.1f} seconds", (1110, 49), 0.46, MUTED)

    cx, cy, cw, ch = 34, 96, 910, 590
    draw_panel(canvas, (cx, cy), (cx + cw, cy + ch), PANEL)
    camera_copy = frame.copy()
    if not calibration.complete:
        camera_copy = cv2.addWeighted(camera_copy, 0.58, np.zeros_like(camera_copy), 0.42, 0)
        draw_text(camera_copy, "CALIBRATING", (48, 92), 1.10, CYAN, 3)
        draw_text(camera_copy, "Keep your eyes naturally open and look forward", (48, 140), 0.60, WHITE)
        draw_text(camera_copy, f"{calibration.remaining(time.perf_counter()):.1f} seconds remaining", (48, 184), 0.60, MUTED)
    display = cv2.resize(camera_copy, (cw - 24, ch - 24))
    canvas[cy + 12:cy + 12 + display.shape[0], cx + 12:cx + 12 + display.shape[1]] = display

    colour = state_colour(state)
    cv2.rectangle(canvas, (cx + 24, cy + 24), (cx + 265, cy + 76), colour, -1)
    draw_text(canvas, state, (cx + 45, cy + 60), 0.72, WHITE, 2)
    draw_text(canvas, "TRAINED MODEL" if model_loaded else "HEURISTIC FALLBACK", (cx + cw - 205, cy + 48), 0.40, GREEN if model_loaded else AMBER)

    rx, rw = 968, 438
    draw_panel(canvas, (rx, 96), (rx + rw, 686), PANEL)
    draw_probability_gauge(canvas, (rx + rw // 2, 257), 112, smoothed_p)
    cv2.rectangle(canvas, (rx + 28, 382), (rx + rw - 28, 438), colour, -1)
    draw_text(canvas, f"STATUS: {state}", (rx + 50, 418), 0.70, WHITE, 2)

    card_w = 181
    draw_metric_card(canvas, rx + 28, 466, card_w, "EAR", f"{features['ear']:.3f}", f"baseline {calibration.baseline_ear:.3f}")
    draw_metric_card(canvas, rx + 229, 466, card_w, "Yawn Score", f"{features['yawn_score']:.3f}", f"risk {evidence['yawn_risk']:.2f}")
    draw_metric_card(canvas, rx + 28, 568, card_w, "Eye Risk", f"{evidence['eye_risk']:.2f}", "baseline deviation")
    draw_metric_card(canvas, rx + 229, 568, card_w, "Alerts", str(alerts), "critical triggers")

    draw_timeline(canvas, history, 34, 712, 1020, 158)
    draw_panel(canvas, (1078, 712), (1406, 870), PANEL)
    draw_text(canvas, "SESSION", (1100, 744), 0.46, WHITE)
    draw_text(canvas, f"Face: {'DETECTED' if features['face_detected'] else 'NOT DETECTED'}", (1100, 776), 0.42, GREEN if features["face_detected"] else RED)
    draw_text(canvas, f"Raw model: {raw_p:.3f}", (1100, 803), 0.40, MUTED)
    draw_text(canvas, f"Decision risk: {decision_p:.3f}", (1100, 830), 0.40, CYAN)
    draw_text(canvas, f"Logging: {'ON' if logging_enabled else 'OFF'}", (1100, 855), 0.38, GREEN if logging_enabled else MUTED)
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="DriverGuardianAI V3 calibrated dashboard")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--no-landmarks", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    estimator, threshold, model_error = load_model(args.model)
    if model_error:
        print("\nModel warning:")
        print(model_error)
        print("Using heuristic fallback.\n")
    model_loaded = estimator is not None

    if args.video is not None:
        capture = cv2.VideoCapture(str(args.video))
        source = str(args.video)
    else:
        capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        source = f"camera {args.camera}"
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    temporal = TemporalStateEngine()
    calibration = PersonalCalibration(required_seconds=10.0, minimum_samples=80)
    logger = SessionLogger()
    history = deque(maxlen=180)

    start = previous = time.perf_counter()
    displayed_fps = 0.0

    print("\n" + "=" * 72)
    print("DriverGuardianAI V3 - Calibrated Decision Layer")
    print("=" * 72)
    print(f"Input source: {source}")
    print(f"Model path:   {args.model}")
    print(f"Model loaded: {model_loaded}")
    print(f"Threshold:    {threshold:.2f}")
    print("Controls: Q/Esc quit | R reset/recalibrate | L logging | I info")
    print("=" * 72)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.video is not None:
                    break
                continue

            now = time.perf_counter()
            elapsed = now - start
            dt = max(now - previous, 1e-6)
            previous = now
            instant_fps = 1.0 / dt
            displayed_fps = instant_fps if displayed_fps == 0.0 else 0.90 * displayed_fps + 0.10 * instant_fps

            if args.video is None:
                frame = cv2.flip(frame, 1)

            features, face_landmarks = extract_features(frame, face_mesh)
            if face_landmarks is not None and not args.no_landmarks:
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
                )

            if features["face_detected"]:
                try:
                    raw_p = predict_probability(estimator, features)
                except Exception as exc:
                    print("Prediction error:", type(exc).__name__, exc)
                    raw_p = heuristic_probability(features)
            else:
                raw_p = 0.0

            calibration.update(features, now)
            if calibration.complete:
                decision_p, evidence = calibration.calculate_fused_probability(raw_p, features)
            else:
                decision_p = 0.0
                evidence = {"model_risk": 0.0, "eye_risk": 0.0, "yawn_risk": 0.0, "tilt_risk": 0.0}

            state, smoothed_p = temporal.update(decision_p, bool(features["face_detected"]), calibration.complete)
            history.append(smoothed_p)
            logger.write(elapsed, features, calibration, raw_p, decision_p, smoothed_p, evidence, state)

            dashboard = build_dashboard(
                frame, features, raw_p, decision_p, smoothed_p, state, history,
                displayed_fps, elapsed, temporal.alert_count, logger.enabled,
                model_loaded, calibration, evidence,
            )
            cv2.imshow("DriverGuardianAI V3 Calibrated", dashboard)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                temporal.reset()
                calibration.reset()
                history.clear()
                print("Temporal state and personal calibration reset.")
            if key == ord("l"):
                enabled = logger.toggle()
                print(f"Logging started: {logger.path}" if enabled else "Logging stopped.")
            if key == ord("i"):
                print("\n" + "-" * 68)
                print(f"FPS: {displayed_fps:.2f}")
                print("Calibration complete:", calibration.complete)
                print(f"Personal EAR baseline: {calibration.baseline_ear:.4f}")
                print(f"EAR: {features['ear']:.4f}")
                print(f"Yawn score: {features['yawn_score']:.4f}")
                print(f"Head tilt: {features['head_tilt']:.2f}")
                print(f"Raw model probability: {raw_p:.4f}")
                print(f"Decision probability: {decision_p:.4f}")
                print(f"Smoothed probability: {smoothed_p:.4f}")
                print(f"Eye risk: {evidence['eye_risk']:.4f}")
                print(f"Yawn risk: {evidence['yawn_risk']:.4f}")
                print(f"Tilt risk: {evidence['tilt_risk']:.4f}")
                print("Temporal state:", state)
                print("Critical alerts:", temporal.alert_count)
                print("-" * 68)
    finally:
        logger.stop()
        capture.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()