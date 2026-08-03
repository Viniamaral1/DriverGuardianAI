from pydantic import BaseModel
from fastapi import APIRouter

from app.services.app_state import guardian_state
from app.services.commander_service import CommanderService
from app.services.voice_service import WakeWordService

router = APIRouter()


class MessageRequest(BaseModel):
    message: str


def service() -> CommanderService:
    return CommanderService(guardian_state)


@router.get("/history")
def history():
    return {"conversation": list(guardian_state.conversation)}


@router.post("/message")
def message(payload: MessageRequest):
    guardian_state.add_message("user", payload.message)
    answer = service().answer(payload.message)
    guardian_state.add_message("assistant", answer)
    guardian_state.add_event("COMMANDER", payload.message[:90], "info")
    return {"response": answer, "conversation": list(guardian_state.conversation)}


@router.post("/clear")
def clear():
    guardian_state.conversation.clear()
    guardian_state.add_message("assistant", "Conversation cleared. Commander remains online.")
    return {"conversation": list(guardian_state.conversation)}


@router.get("/voice")
def voice_status():
    return guardian_state.voice_status()


@router.post("/voice/start")
def voice_start():
    if guardian_state.voice_service is None:
        def on_command(command: str) -> str:
            guardian_state.add_message("user", f"Commander {command}")
            answer = service().answer(command)
            guardian_state.add_message("assistant", answer)
            guardian_state.add_event("VOICE", f"Wake command: {command}", "success")
            return answer
        guardian_state.voice_service = WakeWordService(on_command)
    started = guardian_state.voice_service.start()
    guardian_state.settings["wake_word_enabled"] = bool(started)
    guardian_state.save_settings()
    return guardian_state.voice_status()


@router.post("/voice/stop")
def voice_stop():
    if guardian_state.voice_service is not None:
        guardian_state.voice_service.stop()
    guardian_state.settings["wake_word_enabled"] = False
    guardian_state.save_settings()
    return guardian_state.voice_status()
