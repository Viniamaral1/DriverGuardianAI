from fastapi import APIRouter, HTTPException

from app.services.app_state import guardian_state

router = APIRouter()


@router.get("/status")
def status():
    return guardian_state.snapshot()


@router.get("/diagnostics")
def diagnostics():
    service = guardian_state.monitoring_service
    if service is None:
        return {
            "available": False,
            "detail": "Monitoring service has not been initialised.",
        }
    return {
        "available": True,
        **service.diagnostics(),
    }


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

    if not success:
        # A delayed Windows camera release is not a fatal server error.
        # Return 202 so the frontend can keep showing STOPPING and retry.
        payload["pending"] = True

    return payload
