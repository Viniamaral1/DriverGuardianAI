from __future__ import annotations

import importlib
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator


class LiveMonitoringService:
    """Run the existing DriverGuardian V3 pipeline behind Guardian OS.

    The service owns the webcam and V3 inference objects in a background thread.
    FastAPI reads thread-safe metric snapshots and streams the latest JPEG frame.
    """

    def __init__(
        self,
        root: Path,
        settings_provider: Callable[[], dict[str, Any]],
        event_callback: Callable[[str, str, str], None],
    ) -> None:
        self.root = root
        self.settings_provider = settings_provider
        self.event_callback = event_callback

        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._startup_event = threading.Event()

        self._jpeg: bytes | None = None
        self._error: str | None = None
        self._running = False
        self._starting = False
        self._model_loaded = False
        self._model_warning: str | None = None
        self._log_path: str | None = None

        self._metrics: dict[str, Any] = self._empty_metrics()
        self._previous_temporal_state = "READY"
        self._previous_alert_count = 0

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "monitoring": False,
            "state": "READY",
            "fatigue_probability": 0.0,
            "raw_probability": 0.0,
            "decision_probability": 0.0,
            "ear": 0.0,
            "blink_rate": 0.0,
            "yawn_score": 0.0,
            "head_tilt": 0.0,
            "face_detected": False,
            "session_seconds": 0,
            "alert_count": 0,
            "camera_status": "STANDBY",
            "model_status": "READY",
            "calibration_complete": False,
            "calibration_remaining": 0.0,
            "baseline_ear": 0.0,
            "fps": 0.0,
            "critical_duration": 0.0,
            "cooldown_remaining": 0.0,
            "model_risk": 0.0,
            "eye_risk": 0.0,
            "yawn_risk": 0.0,
            "tilt_risk": 0.0,
            "error": None,
            "log_path": None,
        }

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, wait_seconds: float = 4.0) -> tuple[bool, str]:
        with self._lock:
            if self._running or self._starting:
                return True, "Monitoring is already active."
            self._starting = True
            self._error = None
            self._startup_event.clear()
            self._stop_event.clear()
            self._metrics = self._empty_metrics()
            self._metrics.update(
                {
                    "state": "STARTING",
                    "camera_status": "CONNECTING",
                    "model_status": "LOADING",
                }
            )

        self._thread = threading.Thread(
            target=self._run,
            name="guardian-live-monitoring",
            daemon=True,
        )
        self._thread.start()
        self._startup_event.wait(timeout=wait_seconds)

        with self._lock:
            if self._error:
                return False, self._error
            if self._running:
                return True, "Live V3 monitoring started."
            return True, "Monitoring service is still starting."

    def stop(self, join_seconds: float = 4.0) -> tuple[bool, str]:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=join_seconds)

        with self._lock:
            self._running = False
            self._starting = False
            self._metrics.update(
                {
                    "monitoring": False,
                    "state": "READY",
                    "camera_status": "STANDBY",
                    "model_status": "READY",
                    "face_detected": False,
                    "fatigue_probability": 0.0,
                    "raw_probability": 0.0,
                    "decision_probability": 0.0,
                    "session_seconds": 0,
                }
            )
            self._frame_condition.notify_all()

        return True, "Monitoring stopped."

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._metrics)
            result["monitoring"] = self._running
            result["error"] = self._error
            result["log_path"] = self._log_path
            return result

    def mjpeg_frames(self) -> Iterator[bytes]:
        """Yield the newest frame whenever it changes."""
        last_frame: bytes | None = None

        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: self._jpeg is not None and self._jpeg is not last_frame
                    or self._stop_event.is_set(),
                    timeout=1.0,
                )
                frame = self._jpeg
                stopped = self._stop_event.is_set() and not self._running

            if frame is not None and frame is not last_frame:
                last_frame = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            if stopped:
                break

    def _publish_metrics(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._metrics.update(values)

    def _publish_frame(self, jpeg: bytes) -> None:
        with self._frame_condition:
            self._jpeg = jpeg
            self._frame_condition.notify_all()

    def _publish_error(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._running = False
            self._starting = False
            self._metrics.update(
                {
                    "monitoring": False,
                    "state": "ERROR",
                    "camera_status": "ERROR",
                    "model_status": "ERROR",
                    "error": message,
                }
            )
            self._startup_event.set()
        self.event_callback("CAMERA", message, "danger")

    def _run(self) -> None:
        capture = None
        face_mesh = None
        alert_event_logger = None
        session_logger = None

        try:
            cv2 = importlib.import_module("cv2")
            mp = importlib.import_module("mediapipe")
            v3 = importlib.import_module("realtime_driver_guardian_v3_alerts")

            settings = dict(self.settings_provider())
            camera_index = int(settings.get("camera_index", 0))
            alert_volume = int(settings.get("alert_volume", 80))
            sound_enabled = alert_volume > 0

            estimator, model_threshold, model_error = v3.load_model(
                v3.DEFAULT_MODEL_PATH
            )
            self._model_loaded = estimator is not None
            self._model_warning = model_error

            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            capture = cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened() and backend != cv2.CAP_ANY:
                capture.release()
                capture = cv2.VideoCapture(camera_index)

            if not capture.isOpened():
                raise RuntimeError(
                    f"Could not open camera {camera_index}. Close other camera applications and try again."
                )

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )

            temporal_engine = v3.TemporalStateEngine()
            calibration = v3.PersonalCalibration(
                required_seconds=10.0,
                minimum_samples=80,
            )
            alert_event_logger = v3.AlertEventLogger()
            alert_manager = v3.AlertManager(
                v3.AlertConfig(
                    minimum_critical_seconds=1.5,
                    cooldown_seconds=10.0,
                    sound_enabled=sound_enabled,
                ),
                alert_event_logger,
            )
            session_logger = v3.SessionLogger()
            session_logger.start()
            self._log_path = str(session_logger.path) if session_logger.path else None

            blink_times: deque[float] = deque()
            eye_closed = False
            start_time = time.perf_counter()
            previous_time = start_time
            displayed_fps = 0.0

            with self._lock:
                self._running = True
                self._starting = False
                self._metrics.update(
                    {
                        "monitoring": True,
                        "state": "CALIBRATING",
                        "camera_status": "CONNECTED",
                        "model_status": "ACTIVE" if self._model_loaded else "HEURISTIC",
                        "error": None,
                        "log_path": self._log_path,
                    }
                )
                self._startup_event.set()

            self.event_callback(
                "MONITOR",
                f"Camera {camera_index} connected; personal calibration started",
                "success",
            )
            if model_error:
                self.event_callback(
                    "MODEL",
                    f"Model warning: {model_error}. Heuristic fallback is active.",
                    "warning",
                )

            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    time.sleep(0.02)
                    continue

                now = time.perf_counter()
                elapsed = now - start_time
                frame_duration = max(now - previous_time, 1e-6)
                previous_time = now
                instantaneous_fps = 1.0 / frame_duration
                displayed_fps = (
                    instantaneous_fps
                    if displayed_fps == 0.0
                    else 0.90 * displayed_fps + 0.10 * instantaneous_fps
                )

                frame = cv2.flip(frame, 1)
                features, face_landmarks = v3.extract_features(frame, face_mesh)

                if face_landmarks is not None:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=(
                            mp.solutions.drawing_styles
                            .get_default_face_mesh_contours_style()
                        ),
                    )

                if features["face_detected"]:
                    try:
                        raw_probability = v3.predict_probability(
                            estimator,
                            features,
                        )
                    except Exception:
                        raw_probability = v3.heuristic_probability(features)
                else:
                    raw_probability = 0.0

                calibration.update(features, now)

                if calibration.complete:
                    decision_probability, risk_evidence = (
                        calibration.calculate_fused_probability(
                            raw_probability,
                            features,
                        )
                    )
                else:
                    decision_probability = 0.0
                    risk_evidence = {
                        "model_risk": 0.0,
                        "eye_risk": 0.0,
                        "yawn_risk": 0.0,
                        "tilt_risk": 0.0,
                    }

                temporal_state, smoothed_probability = temporal_engine.update(
                    decision_probability,
                    bool(features["face_detected"]),
                    calibration.complete,
                )

                alert_triggered = alert_manager.update(
                    state=temporal_state,
                    now=now,
                    elapsed_seconds=elapsed,
                    smoothed_probability=smoothed_probability,
                    raw_probability=raw_probability,
                    decision_probability=decision_probability,
                    features=features,
                    calibration=calibration,
                    risk_evidence=risk_evidence,
                )

                session_logger.write(
                    elapsed_seconds=elapsed,
                    features=features,
                    calibration=calibration,
                    raw_probability=raw_probability,
                    decision_probability=decision_probability,
                    smoothed_probability=smoothed_probability,
                    risk_evidence=risk_evidence,
                    temporal_state=temporal_state,
                    alert_manager=alert_manager,
                    now=now,
                )

                # Derive a rolling blink rate from calibrated eye closure transitions.
                if features["face_detected"]:
                    threshold = (
                        calibration.baseline_ear * 0.72
                        if calibration.complete
                        else 0.18
                    )
                    currently_closed = features["ear"] < threshold
                    if eye_closed and not currently_closed:
                        blink_times.append(now)
                    eye_closed = currently_closed
                else:
                    eye_closed = False

                while blink_times and now - blink_times[0] > 60.0:
                    blink_times.popleft()
                observation_seconds = max(1.0, min(60.0, elapsed))
                blink_rate = len(blink_times) * 60.0 / observation_seconds

                if temporal_state != self._previous_temporal_state:
                    level = (
                        "danger"
                        if temporal_state == "CRITICAL"
                        else "warning"
                        if temporal_state in {"WARNING", "NO FACE"}
                        else "success"
                    )
                    self.event_callback(
                        "AI",
                        f"Driver state changed to {temporal_state}",
                        level,
                    )
                    self._previous_temporal_state = temporal_state

                if alert_triggered or alert_manager.alert_count > self._previous_alert_count:
                    self.event_callback(
                        "ALERT",
                        f"Controlled alert #{alert_manager.alert_count} triggered at {smoothed_probability * 100:.1f}% risk",
                        "danger",
                    )
                    self._previous_alert_count = alert_manager.alert_count

                calibration_remaining = calibration.remaining(now)
                self._publish_metrics(
                    {
                        "monitoring": True,
                        "state": temporal_state,
                        "fatigue_probability": round(smoothed_probability, 4),
                        "raw_probability": round(raw_probability, 4),
                        "decision_probability": round(decision_probability, 4),
                        "ear": round(float(features["ear"]), 4),
                        "blink_rate": round(blink_rate, 1),
                        "yawn_score": round(float(features["yawn_score"]), 4),
                        "head_tilt": round(float(features["head_tilt"]), 2),
                        "face_detected": bool(features["face_detected"]),
                        "session_seconds": round(elapsed),
                        "alert_count": alert_manager.alert_count,
                        "camera_status": "CONNECTED",
                        "model_status": "ACTIVE" if self._model_loaded else "HEURISTIC",
                        "calibration_complete": calibration.complete,
                        "calibration_remaining": round(calibration_remaining, 1),
                        "baseline_ear": round(calibration.baseline_ear, 4),
                        "fps": round(displayed_fps, 1),
                        "critical_duration": round(
                            alert_manager.current_critical_duration,
                            2,
                        ),
                        "cooldown_remaining": round(
                            alert_manager.cooldown_remaining(now),
                            2,
                        ),
                        "model_risk": round(risk_evidence["model_risk"], 4),
                        "eye_risk": round(risk_evidence["eye_risk"], 4),
                        "yawn_risk": round(risk_evidence["yawn_risk"], 4),
                        "tilt_risk": round(risk_evidence["tilt_risk"], 4),
                        "error": None,
                        "log_path": self._log_path,
                    }
                )

                state_colour = (
                    (45, 55, 245)
                    if temporal_state == "CRITICAL"
                    else (50, 175, 245)
                    if temporal_state == "WARNING"
                    else (70, 210, 125)
                )
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (10, 16, 24), -1)
                cv2.putText(
                    frame,
                    f"Guardian OS | {temporal_state} | Risk {smoothed_probability * 100:.1f}% | {displayed_fps:.1f} FPS",
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    state_colour,
                    2,
                    cv2.LINE_AA,
                )

                encode_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                if encode_ok:
                    self._publish_frame(encoded.tobytes())

        except Exception as error:
            self._publish_error(f"{type(error).__name__}: {error}")
        finally:
            if session_logger is not None:
                session_logger.stop()
            if alert_event_logger is not None:
                alert_event_logger.close()
            if capture is not None:
                capture.release()
            if face_mesh is not None:
                face_mesh.close()

            with self._lock:
                self._running = False
                self._starting = False
                if self._error is None:
                    self._metrics.update(
                        {
                            "monitoring": False,
                            "state": "READY",
                            "camera_status": "STANDBY",
                            "model_status": "READY",
                            "face_detected": False,
                        }
                    )
                self._startup_event.set()
                self._frame_condition.notify_all()
