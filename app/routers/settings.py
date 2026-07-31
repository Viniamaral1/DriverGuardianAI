from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.app_state import guardian_state

router = APIRouter()


class SettingsUpdate(BaseModel):
    theme: str | None = None
    accent: str | None = None
    voice_output: bool | None = None
    camera_index: int | None = None
    alert_volume: int | None = None
    sensitivity: int | None = None


@router.get("")
def get_settings():
    return guardian_state.settings


@router.put("")
def update_settings(payload: SettingsUpdate):
    updates = payload.model_dump(exclude_none=True)
    guardian_state.settings.update(updates)
    guardian_state.save_settings()
    guardian_state.add_event("SETTINGS", "Application preferences updated", "success")
    return guardian_state.settings
