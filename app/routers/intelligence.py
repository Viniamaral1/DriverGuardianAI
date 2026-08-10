from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services.app_state import guardian_state
from app.services.intelligence_service import IntelligenceService

router = APIRouter()


def service() -> IntelligenceService:
    if guardian_state.intelligence_service is None:
        guardian_state.intelligence_service = IntelligenceService(guardian_state)
    return guardian_state.intelligence_service


class DecisionMemoryMetadataRequest(BaseModel):
    label: str = Field(default="", max_length=120)
    condition: str = Field(default="", max_length=80)
    notes: str = Field(default="", max_length=1000)


@router.get("")
def snapshot():
    return service().snapshot()



@router.get("/memory")
def decision_memory_list():
    return {"sessions": service().decision_memory.list_sessions()}


@router.get("/memory/{session_id}")
def decision_memory_session(session_id: str):
    try:
        return service().decision_memory.get_session(session_id)
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Decision Memory session not found")


@router.get("/memory/{session_id}/json")
def decision_memory_json(session_id: str):
    try:
        path = service().decision_memory.resolve(session_id, ".json")
        return FileResponse(
            path,
            media_type="application/json",
            filename=path.name,
        )
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Decision Memory session not found")


@router.get("/memory/{session_id}/csv")
def decision_memory_csv(session_id: str):
    try:
        content = service().decision_memory.csv_bytes(session_id)
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Decision Memory session not found")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{session_id}.csv"'
        },
    )


@router.get("/memory/compare/{first_id}/{second_id}")
def decision_memory_compare(first_id: str, second_id: str):
    try:
        return service().decision_memory.comparison(first_id, second_id)
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Decision Memory session not found")


@router.get("/memory-summary")
def decision_memory_summary():
    return service().decision_memory.aggregate()


@router.post("/memory/{session_id}/metadata")
def decision_memory_metadata(session_id: str, payload: DecisionMemoryMetadataRequest):
    try:
        return service().decision_memory.update_metadata(
            session_id, label=payload.label, condition=payload.condition, notes=payload.notes
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Decision Memory session not found")
