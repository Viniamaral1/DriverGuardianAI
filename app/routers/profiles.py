from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.app_state import guardian_state

router = APIRouter()


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class ActiveProfileUpdate(BaseModel):
    profile_id: str | None = Field(default=None, max_length=64)


def service():
    if guardian_state.driver_profile_service is None:
        raise HTTPException(status_code=503, detail="Driver profiles are unavailable.")
    return guardian_state.driver_profile_service


@router.get("")
def snapshot():
    return service().snapshot()


@router.post("")
def create_profile(payload: ProfileCreate):
    try:
        profile = service().create(payload.name)
        service().set_active(profile["id"])
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    guardian_state.add_event(
        "PROFILE",
        f"Driver profile created: {profile['name']}",
        "success",
    )
    return service().snapshot()


@router.put("/active")
def set_active(payload: ActiveProfileUpdate):
    if guardian_state.monitoring_service and guardian_state.monitoring_service.active:
        raise HTTPException(
            status_code=409,
            detail="Stop Monitoring before changing the active driver profile.",
        )
    try:
        profile = service().set_active(payload.profile_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    label = profile["name"] if profile else "Guest"
    guardian_state.add_event("PROFILE", f"Active driver: {label}", "success")
    return service().snapshot()


@router.post("/{profile_id}/reset-calibration")
def reset_calibration(profile_id: str):
    if guardian_state.monitoring_service and guardian_state.monitoring_service.active:
        raise HTTPException(
            status_code=409,
            detail="Stop Monitoring before resetting a calibration profile.",
        )
    try:
        service().reset_calibration(profile_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    guardian_state.add_event(
        "PROFILE",
        "Saved personal calibration reset",
        "warning",
    )
    return service().snapshot()


@router.delete("/{profile_id}")
def delete_profile(profile_id: str):
    if guardian_state.monitoring_service and guardian_state.monitoring_service.active:
        raise HTTPException(
            status_code=409,
            detail="Stop Monitoring before deleting a driver profile.",
        )
    if not service().delete(profile_id):
        raise HTTPException(status_code=404, detail="Driver profile was not found.")

    guardian_state.add_event("PROFILE", "Driver profile deleted", "warning")
    return service().snapshot()
