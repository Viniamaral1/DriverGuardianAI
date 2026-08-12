from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import camera, commander, diagnostics, edge, intelligence, monitoring, pages, profiles, reports, research_lab, settings, websocket
from app.services.app_state import guardian_state
from app.main_paths import ROOT, APP_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    guardian_state.initialise(ROOT)
    yield
    guardian_state.shutdown()


app = FastAPI(
    title="Guardian OS V8.3.4",
    version="8.3.4",
    description="Automotive driver-monitoring dashboard and Commander interface.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

app.include_router(pages.router)
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(camera.router, prefix="/api/camera", tags=["camera"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])
app.include_router(edge.router, prefix="/api/edge", tags=["edge"])
app.include_router(intelligence.router, prefix="/api/intelligence", tags=["intelligence"])
app.include_router(profiles.router, prefix="/api/profiles", tags=["profiles"])
app.include_router(commander.router, prefix="/api/commander", tags=["commander"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(research_lab.router, prefix="/api/research-lab", tags=["research-lab"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(websocket.router, tags=["websocket"])


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "application": "Guardian OS V8.3.4",
        "version": "8.3.4",
    }
