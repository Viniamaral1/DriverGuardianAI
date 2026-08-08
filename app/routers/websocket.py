import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.services.app_state import guardian_state

router = APIRouter()


@router.websocket("/ws/live")
async def live(websocket: WebSocket):
    await websocket.accept()
    try:
        while (
            websocket.client_state == WebSocketState.CONNECTED
            and websocket.application_state == WebSocketState.CONNECTED
        ):
            try:
                await websocket.send_json(guardian_state.snapshot())
            except (WebSocketDisconnect, RuntimeError):
                break
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        pass
