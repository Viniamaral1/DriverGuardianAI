from __future__ import annotations

from fastapi import APIRouter

from app.services.app_state import guardian_state
from app.services.intelligence_service import IntelligenceService

router = APIRouter()


def service() -> IntelligenceService:
    return IntelligenceService(guardian_state)


@router.get("")
def snapshot():
    return service().snapshot()
