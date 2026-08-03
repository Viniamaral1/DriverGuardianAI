from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services.app_state import guardian_state

router = APIRouter()


@router.get("/stream")
def stream():
    service = guardian_state.monitoring_service
    if service is None:
        raise HTTPException(status_code=503, detail="Monitoring service is unavailable.")

    return StreamingResponse(
        service.mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )
