from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the exact Python environment and webcam used by Guardian OS."
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    print("=" * 72)
    print("Guardian OS camera and environment diagnostic")
    print("=" * 72)
    print("Python:", sys.executable)
    print("Version:", sys.version.replace("\n", " "))
    print("Project:", Path.cwd())
    print("Camera index:", args.camera)
    print()

    try:
        cv2 = importlib.import_module("cv2")
        print("OpenCV:", cv2.__version__)
    except Exception as error:
        print("FAILED: OpenCV import:", type(error).__name__, error)
        return 1

    try:
        mp = importlib.import_module("mediapipe")
        print("MediaPipe:", getattr(mp, "__version__", "unknown"))
        print("MediaPipe solutions:", hasattr(mp, "solutions"))
    except Exception as error:
        print("FAILED: MediaPipe import:", type(error).__name__, error)
        return 1

    try:
        sklearn = importlib.import_module("sklearn")
        print("scikit-learn:", sklearn.__version__)
    except Exception as error:
        print("scikit-learn import warning:", type(error).__name__, error)

    print()
    print("Testing V3 import...")
    try:
        v3 = importlib.import_module("realtime_driver_guardian_v3_alerts")
        print("V3 module: OK")
        print("Model path:", v3.DEFAULT_MODEL_PATH)
        print("Model exists:", v3.DEFAULT_MODEL_PATH.exists())
    except Exception as error:
        print("FAILED: V3 import:", type(error).__name__, error)
        return 1

    backends = []
    if os.name == "nt":
        backends.extend(
            [
                ("DSHOW", cv2.CAP_DSHOW),
                ("MSMF", cv2.CAP_MSMF),
            ]
        )
    backends.append(("DEFAULT", None))

    print()
    print("Testing camera backends...")

    for name, backend in backends:
        print(f"- {name}: opening...")
        capture = (
            cv2.VideoCapture(args.camera, backend)
            if backend is not None
            else cv2.VideoCapture(args.camera)
        )

        try:
            if not capture.isOpened():
                print(f"  {name}: could not open")
                continue

            ok = False
            frame = None
            for _ in range(25):
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)

            if not ok or frame is None:
                print(f"  {name}: opened but returned no frame")
                continue

            print(f"  {name}: SUCCESS — frame shape {frame.shape}")
            print()
            print("Diagnostic passed.")
            print(
                "Start Guardian OS with this same Python executable and without --reload:"
            )
            print(
                f'  & "{sys.executable}" -m uvicorn app.main:app --host 127.0.0.1 --port 8010'
            )
            return 0
        finally:
            capture.release()

    print()
    print("FAILED: No camera backend returned a frame.")
    print("Close Jupyter, OpenCV windows, Teams, Zoom, and browser camera tabs.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
