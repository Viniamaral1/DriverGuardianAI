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
    payload = service().snapshot()
    metrics = guardian_state.metrics()
    payload["live_perception"] = {
        "monitoring": bool(metrics.get("monitoring")),
        "automatic_occlusion": str(metrics.get("automatic_occlusion") or "unknown"),
        "automatic_occlusion_confidence": float(
            metrics.get("automatic_occlusion_confidence") or 0.0
        ),
        "automatic_occlusion_summary": str(
            metrics.get("automatic_occlusion_summary") or ""
        ),
        "eye_visibility_score": float(metrics.get("eye_visibility_score") or 0.0),
        "eye_region_brightness_ratio": float(
            metrics.get("eye_region_brightness_ratio") or 0.0
        ),
        "eye_dark_ratio": float(metrics.get("eye_dark_ratio") or 0.0),
        "eye_edge_density": float(metrics.get("eye_edge_density") or 0.0),
        "perception_quality": str(
            metrics.get("automatic_perception_quality") or "standby"
        ),
        "perception_score": float(
            metrics.get("automatic_perception_score") or 0.0
        ),
    }
    return payload


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
    updates = payload.model_dump(exclude_none=True)
    context = service().set_context(updates)

    resolved = service().resolved_context(
        force_weather=bool(
            context.get("automatic_enabled", True)
            and str(context.get("location", "") or "").strip()
        )
    )

    guardian_state.add_event(
        "CONTEXT",
        "Journey context updated and automatic values refreshed",
        "success",
    )
    return {
        "context": resolved,
        "manual_context": context,
        "snapshot": service().snapshot(),
    }


@router.post("/context/clear-manual")
def clear_manual_context():
    manual = service().clear_manual_context()
    resolved = service().resolved_context(force_weather=False)
    guardian_state.add_event(
        "CONTEXT",
        "Manual journey context cleared",
        "info",
    )
    return {
        "manual_context": manual,
        "context": resolved,
        "snapshot": service().snapshot(),
    }


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
