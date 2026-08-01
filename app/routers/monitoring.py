from fastapi import APIRouter, HTTPException

from app.services.app_state import guardian_state

router = APIRouter()


@router.get("/status")
def status():
    return guardian_state.snapshot()


@router.post("/start")
def start():
    success, message = guardian_state.start_monitoring()
    payload = guardian_state.snapshot()
    payload["success"] = success
    payload["message"] = message
    if not success:
        raise HTTPException(status_code=503, detail=message)
    return payload


@router.post("/stop")
def stop():
    success, message = guardian_state.stop_monitoring()
    payload = guardian_state.snapshot()
    payload["success"] = success
    payload["message"] = message
    return payload
