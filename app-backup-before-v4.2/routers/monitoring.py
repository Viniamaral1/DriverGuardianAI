from fastapi import APIRouter

from app.services.app_state import guardian_state

router = APIRouter()


@router.get("/status")
def status():
    return guardian_state.snapshot()


@router.post("/start")
def start():
    guardian_state.start_monitoring()
    return guardian_state.snapshot()


@router.post("/stop")
def stop():
    guardian_state.stop_monitoring()
    return guardian_state.snapshot()
