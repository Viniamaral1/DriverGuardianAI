from __future__ import annotations

from fastapi import APIRouter

from app.services.app_state import guardian_state
from app.services.intelligence_service import IntelligenceService

router = APIRouter()
_service = IntelligenceService(guardian_state)


def service() -> IntelligenceService:
    return _service


@router.get("")
def snapshot():
    return service().snapshot()
