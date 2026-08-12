from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.app_state import guardian_state

router = APIRouter()


class SettingsUpdate(BaseModel):
    theme: str | None = None
    accent: str | None = None
    voice_output: bool | None = None
    camera_index: int | None = Field(default=None, ge=0, le=9)
    alert_volume: int | None = Field(default=None, ge=0, le=100)
    sensitivity: int | None = Field(default=None, ge=1, le=100)
    driver_name: str | None = Field(default=None, max_length=40)
    automatic_reports: bool | None = None
    persistent_calibration_enabled: bool | None = None
    visual_evidence_enabled: bool | None = None


@router.get("")
def get_settings():
    return guardian_state.settings


@router.put("")
def update_settings(payload: SettingsUpdate):
    updates = payload.model_dump(exclude_none=True)
    if "driver_name" in updates:
        updates["driver_name"] = updates["driver_name"].strip()

    guardian_state.settings.update(updates)
    guardian_state.save_settings()
    guardian_state.add_event(
        "SETTINGS",
        "Application preferences updated",
        "success",
    )
    return guardian_state.settings
