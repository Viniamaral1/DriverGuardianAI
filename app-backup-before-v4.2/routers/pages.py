from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parents[1] / "templates")


def render(request: Request, template: str, page: str, title: str):
    return templates.TemplateResponse(
        request=request,
        name=template,
        context={"page": page, "title": title},
    )


@router.get("/")
def dashboard(request: Request):
    return render(request, "dashboard.html", "dashboard", "Dashboard")


@router.get("/commander")
def commander(request: Request):
    return render(request, "commander.html", "commander", "Commander")


@router.get("/reports")
def reports(request: Request):
    return render(request, "reports.html", "reports", "Reports")


@router.get("/settings")
def settings(request: Request):
    return render(request, "settings.html", "settings", "Settings")


@router.get("/about")
def about(request: Request):
    return render(request, "about.html", "about", "About")
