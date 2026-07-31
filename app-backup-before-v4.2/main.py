from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import commander, monitoring, pages, reports, settings, websocket
from app.services.app_state import guardian_state

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    guardian_state.initialise(ROOT)
    yield
    guardian_state.shutdown()


app = FastAPI(
    title="DriverGuardianAI V4",
    version="4.0.0",
    description="Automotive driver-monitoring dashboard and Commander interface.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

app.include_router(pages.router)
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(commander.router, prefix="/api/commander", tags=["commander"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(websocket.router, tags=["websocket"])


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "application": "DriverGuardianAI V4",
        "version": "4.0.0",
    }
