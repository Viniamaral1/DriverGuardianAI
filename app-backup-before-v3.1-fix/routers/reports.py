from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.app_state import guardian_state
from app.services.pdf_report_service import PdfReportService
from app.services.report_service import ReportService

router = APIRouter()


def reports():
    return ReportService(guardian_state.root)


def pdf_reports():
    return PdfReportService(guardian_state.root)


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


@router.get("/{report_id}/pdf")
def report_pdf(report_id: str):
    try:
        path = pdf_reports().generate(report_id)
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=f"{report_id}.pdf",
        )
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {type(error).__name__}: {error}",
        )
