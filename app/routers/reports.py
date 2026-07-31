from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.app_state import guardian_state
from app.services.report_service import ReportService

router = APIRouter()


def reports():
    return ReportService(guardian_state.root)


@router.get("")
def list_reports():
    return {"reports": reports().list_reports()}


@router.get("/{report_id}/data")
def report_data(report_id: str):
    try:
        path = reports().resolve(report_id, ".json")
        return FileResponse(path, media_type="application/json", filename=path.name)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Report not found")


@router.get("/{report_id}/view")
def report_view(report_id: str):
    try:
        path = reports().resolve(report_id, ".html")
        return FileResponse(path, media_type="text/html")
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="HTML report not found")
