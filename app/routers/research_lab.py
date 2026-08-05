from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.main_paths import ROOT
from app.services.research_lab_service import ResearchLabService

router = APIRouter()
_lab = ResearchLabService(ROOT)

class AuditRequest(BaseModel):
    dataset_path: str = Field(min_length=1, max_length=1000)

@router.get("")
def status():
    return _lab.snapshot()

@router.post("/audit")
def audit(payload: AuditRequest):
    if not _lab.deployment_mode()["research_enabled"]:
        raise HTTPException(
            status_code=403,
            detail="Research Lab is disabled in this deployment.",
        )
    try:
        return _lab.analyse(payload.dataset_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Research audit failed: {type(error).__name__}: {error}",
        ) from error

@router.get("/export")
def export_latest():
    if not _lab.deployment_mode()["research_enabled"]:
        raise HTTPException(
            status_code=403,
            detail="Research Lab export is disabled in this deployment.",
        )
    path = ROOT / "guardian_data" / "research_lab" / "latest_research_audit.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run an audit before exporting.")
    return FileResponse(path, media_type="application/json", filename=path.name)
