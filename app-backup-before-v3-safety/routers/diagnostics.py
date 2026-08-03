from __future__ import annotations

import platform
import sys
from typing import Any

from fastapi import APIRouter

from app.services.app_state import guardian_state

router = APIRouter()


@router.get("")
def diagnostics() -> dict[str, Any]:
    service = guardian_state.monitoring_service
    monitoring = (
        service.diagnostics()
        if service is not None
        else {"available": False, "error": "Monitoring service unavailable."}
    )

    versions: dict[str, str | None] = {}
    for name in ("cv2", "mediapipe", "sklearn", "fastapi"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception:
            versions[name] = None

    return {
        "system": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "packages": versions,
        "monitoring": monitoring,
        "voice": guardian_state.voice_status(),
        "report": (
            guardian_state.report_service.snapshot()
            if guardian_state.report_service is not None
            else {"state": "IDLE"}
        ),
        "settings": {
            "camera_index": guardian_state.settings.get("camera_index", 0),
            "automatic_reports": guardian_state.settings.get("automatic_reports", True),
            "driver_name": guardian_state.settings.get("driver_name", ""),
        },
    }
