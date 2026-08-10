from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
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




@router.get("/monitoring")
def monitoring_page(request: Request):
    return render(request, "monitoring.html", "monitoring", "Live Monitoring")


@router.get("/metrics", include_in_schema=False)
def metrics_page():
    return RedirectResponse(url="/reports", status_code=307)


@router.get("/commander")
def commander(request: Request):
    return render(request, "commander.html", "commander", "Commander")


@router.get("/reports")
def reports(request: Request):
    return render(request, "reports.html", "reports", "Reports & Metrics")


@router.get("/edge")
def edge_page(request: Request):
    return render(request, "edge.html", "edge", "Edge Intelligence")



@router.get("/intelligence")
def intelligence_page(request: Request):
    return render(
        request,
        "intelligence.html",
        "intelligence",
        "Guardian Intelligence",
    )


@router.get("/decision-memory")
def decision_memory_page(request: Request):
    return render(
        request,
        "decision_memory.html",
        "decision_memory",
        "Decision Memory",
    )



@router.get("/profiles")
def profiles_page(request: Request):
    return render(
        request,
        "profiles.html",
        "profiles",
        "Driver Profiles",
    )




@router.get("/research-lab")
def research_lab_page(request: Request):
    return render(request, "research_lab.html", "research_lab", "AI Research Lab")


@router.get("/settings")
def settings(request: Request):
    return render(request, "settings.html", "settings", "Settings")


@router.get("/about")
def about(request: Request):
    return render(request, "about.html", "about", "About")


@router.get("/diagnostics")
def diagnostics_page(request: Request):
    return render(request, "diagnostics.html", "diagnostics", "Diagnostics")
