import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.app_state import guardian_state

router = APIRouter()


@router.websocket("/ws/live")
async def live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(guardian_state.snapshot())
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        return
