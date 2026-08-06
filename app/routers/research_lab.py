from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.main_paths import ROOT
from app.services.research_lab_service import ResearchLabService
from app.services.model_evaluation_service import ModelEvaluationService

router = APIRouter()
_lab = ResearchLabService(ROOT)
_evaluator = ModelEvaluationService(ROOT)

class AuditRequest(BaseModel):
    dataset_path: str = Field(min_length=1, max_length=1000)


class EvaluationRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=1000)
    test_path: str = Field(min_length=1, max_length=1000)
    calibration_path: str | None = Field(default=None, max_length=1000)

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

@router.get("/evaluation")
def evaluation_status():
    if not _lab.deployment_mode()["research_enabled"]:
        raise HTTPException(
            status_code=403,
            detail="Model Evaluation is disabled in this deployment.",
        )
    return _evaluator.snapshot()


@router.post("/evaluation")
def run_evaluation(payload: EvaluationRequest):
    if not _lab.deployment_mode()["research_enabled"]:
        raise HTTPException(
            status_code=403,
            detail="Model Evaluation is disabled in this deployment.",
        )
    try:
        return _evaluator.evaluate(
            model_path=payload.model_path,
            test_path=payload.test_path,
            calibration_path=payload.calibration_path,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model evaluation failed: {type(error).__name__}: {error}",
        ) from error


@router.get("/evaluation/export")
def export_evaluation():
    if not _lab.deployment_mode()["research_enabled"]:
        raise HTTPException(
            status_code=403,
            detail="Model Evaluation export is disabled in this deployment.",
        )
    path = ROOT / "guardian_data" / "research_lab" / "latest_model_evaluation.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Run a model evaluation before exporting.",
        )
    return FileResponse(
        path,
        media_type="application/json",
        filename=path.name,
    )
