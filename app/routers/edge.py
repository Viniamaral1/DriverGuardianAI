from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.app_state import guardian_state

router = APIRouter()


class ContextUpdate(BaseModel):
    automatic_enabled: bool | None = None
    location: str | None = Field(default=None, max_length=120)
    manual_override: bool | None = None
    weather: str | None = Field(default=None, max_length=60)
    external_light: str | None = Field(default=None, max_length=60)
    cabin_light: str | None = Field(default=None, max_length=60)
    occlusion: str | None = Field(default=None, max_length=60)
    road_condition: str | None = Field(default=None, max_length=60)
    notes: str | None = Field(default=None, max_length=240)


class SyncUpdate(BaseModel):
    session_ids: list[str] | None = None


def service():
    if guardian_state.edge_memory_service is None:
        raise HTTPException(status_code=503, detail="Edge memory is unavailable.")
    return guardian_state.edge_memory_service


@router.get("")
def snapshot():
    service().refresh_from_reports()
    return service().snapshot()


@router.post("/refresh")
def refresh():
    result = service().refresh_from_reports()
    guardian_state.add_event(
        "EDGE",
        f"Local memory refreshed: {result['session_count']} sessions",
        "success",
    )
    return {**result, "snapshot": service().snapshot()}


@router.put("/context")
def update_context(payload: ContextUpdate):
    context = service().set_context(payload.model_dump(exclude_none=True))
    guardian_state.add_event(
        "CONTEXT",
        "Journey context updated locally",
        "success",
    )
    return {"context": context, "snapshot": service().snapshot()}


@router.post("/context/refresh-weather")
def refresh_weather():
    context = service().resolved_context(force_weather=True)
    guardian_state.add_event("CONTEXT", "Automatic weather context refreshed", "success" if context.get("automatic_weather", {}).get("available") else "warning")
    return {"context": context, "snapshot": service().snapshot()}


@router.post("/sync/mark-complete")
def mark_sync_complete(payload: SyncUpdate):
    count = service().mark_synced(payload.session_ids)
    guardian_state.add_event(
        "EDGE",
        f"{count} local session records marked as synced",
        "info",
    )
    return {"updated": count, "snapshot": service().snapshot()}


@router.get("/export")
def export_bundle():
    payload = service().export_bundle()
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": (
                'attachment; filename="guardian_edge_memory_export.json"'
            )
        },
    )
