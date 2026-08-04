from __future__ import annotations

import importlib
import os
import threading
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator

from app.services.persistent_calibration import PersistentCalibrationAdapter


class LiveMonitoringService:
    """Thread-safe adapter around the existing V3 monitoring components.

    The original V3 feature extraction, calibration, temporal logic, model
    prediction, alert manager, and loggers remain unchanged. This adapter only
    manages their lifecycle for FastAPI and publishes snapshots/JPEG frames.
    """

    def __init__(
        self,
        root: Path,
        settings_provider: Callable[[], dict[str, Any]],
        event_callback: Callable[[str, str, str], None],
        profile_provider: Callable[[], dict[str, Any] | None] | None = None,
        profile_service: Any | None = None,
    ) -> None:
        self.root = root
        self.settings_provider = settings_provider
        self.event_callback = event_callback
        self.profile_provider = profile_provider or (lambda: None)
        self.profile_service = profile_service

        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._startup_event = threading.Event()

        self._capture: Any | None = None
        self._jpeg: bytes | None = None
        self._error: str | None = None
        self._running = False
        self._starting = False
        self._stopping = False
        self._model_loaded = False
        self._model_warning: str | None = None
        self._log_path: str | None = None
        self._camera_backend: str | None = None

        self._metrics: dict[str, Any] = self.empty_metrics()
        self._previous_temporal_state = "READY"
        self._previous_alert_count = 0

    @staticmethod
    def empty_metrics() -> dict[str, Any]:
        return {
            "monitoring": False,
            "starting": False,
            "stopping": False,
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
            "camera_backend": None,
            "model_status": "READY",
            "model_warning": None,
            "calibration_complete": False,
            "calibration_remaining": 0.0,
            "baseline_ear": 0.0,
            "calibration_mode": "full",
            "calibration_status": "FULL CALIBRATION",
            "calibration_required_seconds": 10.0,
            "calibration_fallback_reason": None,
            "driver_profile_id": None,
            "driver_profile_name": "Guest",
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
    def active(self) -> bool:
        with self._lock:
            return self._running or self._starting or self._stopping

    def start(self, wait_seconds: float = 8.0) -> tuple[bool, str]:
        with self._lock:
            if self._running:
                return True, "Monitoring is already active."
            if self._starting:
                return True, "Monitoring is already starting."
            if self._stopping:
                return False, "Monitoring is still stopping. Try again in a moment."

            self._starting = True
            self._stopping = False
            self._error = None
            self._jpeg = None
            self._model_loaded = False
            self._model_warning = None
            self._log_path = None
            self._camera_backend = None
            self._previous_temporal_state = "READY"
            self._previous_alert_count = 0
            self._startup_event.clear()
            self._stop_event.clear()
            self._metrics = self.empty_metrics()
            self._metrics.update(
                {
                    "starting": True,
                    "state": "STARTING",
                    "camera_status": "CONNECTING",
                    "model_status": "WAITING",
                }
            )

            self._thread = threading.Thread(
                target=self._run,
                name="guardian-v3-adapter",
                daemon=True,
            )
            self._thread.start()

        # Wait long enough to receive a real camera-open result, but never block
        # FastAPI forever if a Windows camera backend hangs.
        self._startup_event.wait(timeout=wait_seconds)

        with self._lock:
            if self._error:
                return False, self._error
            if self._running:
                return True, "Camera connected. V3 monitoring is active."
            if self._starting:
                return True, "Camera connection is still starting."
            return False, "Monitoring stopped before startup completed."

    def stop(self, join_seconds: float = 6.0) -> tuple[bool, str]:
        with self._lock:
            if not self.active:
                self._reset_to_ready()
                return True, "Monitoring is already stopped."

            self._stopping = True
            self._starting = False
            self._metrics.update(
                {
                    "starting": False,
                    "stopping": True,
                    "state": "STOPPING",
                    "camera_status": "DISCONNECTING",
                }
            )
            capture = self._capture
            thread = self._thread
            self._stop_event.set()

        # Releasing the capture here is intentional: on Windows it unblocks a
        # pending capture.read(), allowing the worker to reach its finally block.
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=join_seconds)

        with self._lock:
            still_alive = bool(thread and thread.is_alive())
            if still_alive:
                self._metrics.update(
                    {
                        "stopping": True,
                        "state": "STOPPING",
                        "camera_status": "DISCONNECTING",
                    }
                )
                return False, "Stop was requested; the camera backend is still releasing."

            self._reset_to_ready()
            return True, "Monitoring stopped and the camera was released."

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._metrics)
            result.update(
                {
                    "monitoring": self._running,
                    "starting": self._starting,
                    "stopping": self._stopping,
                    "error": self._error,
                    "log_path": self._log_path,
                    "camera_backend": self._camera_backend,
                    "model_warning": self._model_warning,
                }
            )
            return result

    def diagnostics(self) -> dict[str, Any]:
        result = self.snapshot()
        result.update(
            {
                "thread_alive": bool(self._thread and self._thread.is_alive()),
                "stop_requested": self._stop_event.is_set(),
                "frame_available": self._jpeg is not None,
            }
        )
        return result

    def mjpeg_frames(self) -> Iterator[bytes]:
        last_frame: bytes | None = None

        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: (
                        self._jpeg is not None
                        and self._jpeg is not last_frame
                    )
                    or (
                        self._stop_event.is_set()
                        and not self._running
                        and not self._starting
                    ),
                    timeout=1.0,
                )
                frame = self._jpeg
                finished = (
                    self._stop_event.is_set()
                    and not self._running
                    and not self._starting
                )

            if frame is not None and frame is not last_frame:
                last_frame = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store, no-cache, must-revalidate\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            if finished:
                break

    def _reset_to_ready(self) -> None:
        self._running = False
        self._starting = False
        self._stopping = False
        self._capture = None
        self._jpeg = None
        self._thread = None
        self._stop_event.clear()
        self._metrics = self.empty_metrics()
        self._frame_condition.notify_all()

    def _publish_metrics(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._metrics.update(values)

    def _publish_frame(self, jpeg: bytes) -> None:
        with self._frame_condition:
            self._jpeg = jpeg
            self._frame_condition.notify_all()

    def _publish_error(self, message: str, source: str = "MONITOR") -> None:
        with self._lock:
            self._error = message
            self._running = False
            self._starting = False
            self._stopping = False
            self._metrics.update(
                {
                    "monitoring": False,
                    "starting": False,
                    "stopping": False,
                    "state": "ERROR",
                    "camera_status": "ERROR",
                    "model_status": "ERROR",
                    "error": message,
                }
            )
            self._startup_event.set()
            self._frame_condition.notify_all()

        self.event_callback(source, message, "danger")

    @staticmethod
    def _camera_candidates(cv2: Any, camera_index: int) -> list[tuple[str, Any]]:
        if os.name != "nt":
            return [("DEFAULT", cv2.VideoCapture(camera_index))]

        # V3 originally uses DirectShow. Keep it first, then fall back to MSMF
        # and the default backend because Windows camera drivers vary.
        return [
            ("DSHOW", cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)),
            ("MSMF", cv2.VideoCapture(camera_index, cv2.CAP_MSMF)),
            ("DEFAULT", cv2.VideoCapture(camera_index)),
        ]

    def _open_camera(self, cv2: Any, camera_index: int) -> tuple[Any, str]:
        failures: list[str] = []

        for backend_name, capture in self._camera_candidates(cv2, camera_index):
            if self._stop_event.is_set():
                try:
                    capture.release()
                except Exception:
                    pass
                raise RuntimeError("Camera startup was cancelled.")

            try:
                opened = capture.isOpened()
            except Exception as error:
                opened = False
                failures.append(f"{backend_name}: {type(error).__name__}: {error}")

            if opened:
                # These settings mirror the standalone V3 script.
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                try:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                # A successful isOpened() is not always enough on Windows.
                # Require one actual frame before declaring the camera ready.
                frame_ok = False
                for _ in range(20):
                    if self._stop_event.is_set():
                        break
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        frame_ok = True
                        break
                    time.sleep(0.05)

                if frame_ok:
                    return capture, backend_name

                failures.append(f"{backend_name}: opened but returned no frames")

            try:
                capture.release()
            except Exception:
                pass

        detail = "; ".join(failures) if failures else "no backend could open the device"
        raise RuntimeError(
            f"Could not open camera {camera_index}. {detail}. "
            "Close Jupyter/OpenCV/Teams/Zoom and check the camera index in Settings."
        )

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
            persistent_enabled = bool(
                settings.get("persistent_calibration_enabled", True)
            )
            active_profile = (
                self.profile_provider()
                if persistent_enabled
                else None
            )

            # Open the real webcam before loading the model so the user receives
            # a definite camera result instead of waiting behind joblib loading.
            capture, backend_name = self._open_camera(cv2, camera_index)
            with self._lock:
                self._capture = capture
                self._camera_backend = backend_name

            if self._stop_event.is_set():
                return

            try:
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.55,
                )
            except AttributeError as error:
                raise RuntimeError(
                    "This MediaPipe installation does not expose mp.solutions. "
                    "Run Guardian OS from the same Python environment where the standalone V3 works."
                ) from error

            with self._lock:
                self._running = True
                self._starting = False
                self._metrics.update(
                    {
                        "monitoring": True,
                        "starting": False,
                        "state": "CALIBRATING",
                        "camera_status": "CONNECTED",
                        "camera_backend": backend_name,
                        "model_status": "LOADING",
                        "driver_profile_id": (
                            active_profile.get("id") if active_profile else None
                        ),
                        "driver_profile_name": (
                            active_profile.get("name") if active_profile else "Guest"
                        ),
                        "calibration_mode": (
                            "quick"
                            if active_profile and active_profile.get("calibration")
                            else "full"
                        ),
                        "calibration_status": (
                            "VERIFYING SAVED PROFILE"
                            if active_profile and active_profile.get("calibration")
                            else "FULL CALIBRATION"
                        ),
                        "calibration_required_seconds": (
                            3.0
                            if active_profile and active_profile.get("calibration")
                            else 10.0
                        ),
                        "error": None,
                    }
                )
                self._startup_event.set()

            self.event_callback(
                "CAMERA",
                f"Camera {camera_index} connected using {backend_name}",
                "success",
            )

            # Load the existing V3 model after the camera is confirmed. Any
            # incompatible persisted model falls back to the existing heuristic.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator, _, model_error = v3.load_model(v3.DEFAULT_MODEL_PATH)

            version_messages = [
                str(item.message)
                for item in caught
                if "InconsistentVersionWarning" in type(item.message).__name__
                or "version" in str(item.message).lower()
            ]
            if version_messages:
                self._model_warning = (
                    "The model was trained with a different scikit-learn version. "
                    "Use scikit-learn 1.7.1 for reproducible predictions."
                )

            self._model_loaded = estimator is not None
            if model_error:
                self._model_warning = model_error

            self._publish_metrics(
                {
                    "model_status": "ACTIVE" if self._model_loaded else "HEURISTIC",
                    "model_warning": self._model_warning,
                }
            )
            if self._model_warning:
                self.event_callback("MODEL", self._model_warning, "warning")

            temporal_engine = v3.TemporalStateEngine()
            calibration = PersistentCalibrationAdapter(
                v3.PersonalCalibration,
                profile=active_profile,
                profile_service=self.profile_service,
                event_callback=self.event_callback,
                camera_index=camera_index,
                quick_seconds=3.0,
                quick_minimum_samples=20,
                match_tolerance=0.12,
                full_seconds=10.0,
                full_minimum_samples=80,
            )
            self.event_callback(
                "PROFILE",
                (
                    f"Active driver: {calibration.profile_name}. "
                    f"{'Quick verification ready.' if calibration.mode == 'quick' else 'Full calibration required.'}"
                ),
                "info",
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

            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success or frame is None:
                    if self._stop_event.is_set():
                        break
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
                        raw_probability = v3.predict_probability(estimator, features)
                    except Exception as error:
                        if self._model_loaded:
                            self._model_loaded = False
                            self._model_warning = (
                                f"Model prediction failed ({type(error).__name__}); "
                                "the existing V3 heuristic fallback is now active."
                            )
                            self.event_callback("MODEL", self._model_warning, "warning")
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

                if features["face_detected"]:
                    eye_threshold = (
                        calibration.baseline_ear * 0.72
                        if calibration.complete
                        else 0.18
                    )
                    currently_closed = features["ear"] < eye_threshold
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
                        f"Controlled alert #{alert_manager.alert_count} triggered at "
                        f"{smoothed_probability * 100:.1f}% risk",
                        "danger",
                    )
                    self._previous_alert_count = alert_manager.alert_count

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
                        "camera_backend": backend_name,
                        "model_status": "ACTIVE" if self._model_loaded else "HEURISTIC",
                        "model_warning": self._model_warning,
                        "calibration_complete": calibration.complete,
                        "calibration_remaining": round(calibration.remaining(now), 1),
                        "baseline_ear": round(calibration.baseline_ear, 4),
                        "calibration_mode": calibration.mode,
                        "calibration_status": calibration.status,
                        "calibration_required_seconds": (
                            3.0 if calibration.mode == "quick" else 10.0
                        ),
                        "calibration_fallback_reason": calibration.fallback_reason,
                        "driver_profile_id": calibration.profile_id,
                        "driver_profile_name": calibration.profile_name,
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
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame.shape[1], 46),
                    (10, 16, 24),
                    -1,
                )
                cv2.putText(
                    frame,
                    (
                        f"Guardian OS | {temporal_state} | "
                        f"Risk {smoothed_probability * 100:.1f}% | "
                        f"{displayed_fps:.1f} FPS"
                    ),
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
            self._publish_error(
                f"{type(error).__name__}: {error}",
                source="CAMERA",
            )
        finally:
            if session_logger is not None:
                try:
                    session_logger.stop()
                except Exception:
                    pass
            if alert_event_logger is not None:
                try:
                    alert_event_logger.close()
                except Exception:
                    pass
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            if face_mesh is not None:
                try:
                    face_mesh.close()
                except Exception:
                    pass

            with self._lock:
                self._capture = None
                self._running = False
                self._starting = False
                self._stopping = False
                if self._error is None:
                    self._metrics.update(
                        {
                            "monitoring": False,
                            "starting": False,
                            "stopping": False,
                            "state": "READY",
                            "camera_status": "STANDBY",
                            "model_status": "READY",
                            "face_detected": False,
                        }
                    )
                self._startup_event.set()
                self._frame_condition.notify_all()
