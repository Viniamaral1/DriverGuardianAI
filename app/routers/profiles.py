from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Any
import json
import re

from app.services.app_state import guardian_state

router = APIRouter()


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class ActiveProfileUpdate(BaseModel):
    profile_id: str | None = Field(default=None, max_length=64)


class PassportPrivacyUpdate(BaseModel):
    allow_export: bool = True
    include_perception_history: bool = True


class PassportImport(BaseModel):
    passport: dict[str, Any]


def service():
    if guardian_state.driver_profile_service is None:
        raise HTTPException(status_code=503, detail="Driver profiles are unavailable.")
    return guardian_state.driver_profile_service


def passport_service():
    if guardian_state.calibration_passport_service is None:
        raise HTTPException(
            status_code=503,
            detail="Calibration Passport is unavailable.",
        )
    return guardian_state.calibration_passport_service


def validation_service():
    if guardian_state.passport_validation_service is None:
        raise HTTPException(
            status_code=503,
            detail="Passport Validation is unavailable.",
        )
    return guardian_state.passport_validation_service


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


@router.get("/{profile_id}/passport")
def get_passport(profile_id: str):
    try:
        return passport_service().build(profile_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/{profile_id}/passport/validation")
def passport_validation(profile_id: str):
    try:
        return validation_service().evaluate(profile_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.put("/{profile_id}/passport/privacy")
def update_passport_privacy(
    profile_id: str,
    payload: PassportPrivacyUpdate,
):
    try:
        return passport_service().update_privacy(
            profile_id,
            allow_export=payload.allow_export,
            include_perception_history=payload.include_perception_history,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/{profile_id}/passport/export")
def export_passport(profile_id: str):
    try:
        payload = passport_service().export_payload(profile_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        str(payload.get("identity", {}).get("profile_name") or "driver"),
    ).strip("-") or "driver"
    filename = f"Guardian-Calibration-Passport-{safe_name}.json"
    return Response(
        json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, no-store",
        },
    )


@router.post("/{profile_id}/passport/import")
def import_passport(profile_id: str, payload: PassportImport):
    if guardian_state.monitoring_service and guardian_state.monitoring_service.active:
        raise HTTPException(
            status_code=409,
            detail="Stop Monitoring before importing a Calibration Passport.",
        )
    try:
        passport = passport_service().import_into(
            profile_id,
            payload.passport,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    guardian_state.add_event(
        "PASSPORT",
        "Personal AI Calibration Passport imported",
        "success",
    )
    return passport


@router.post("/{profile_id}/passport/reset")
def reset_passport(profile_id: str):
    if guardian_state.monitoring_service and guardian_state.monitoring_service.active:
        raise HTTPException(
            status_code=409,
            detail="Stop Monitoring before resetting Passport metadata.",
        )
    try:
        passport = passport_service().reset(profile_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    guardian_state.add_event(
        "PASSPORT",
        "Calibration Passport metadata reset",
        "warning",
    )
    return passport


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
