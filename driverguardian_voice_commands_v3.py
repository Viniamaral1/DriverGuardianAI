"""
DriverGuardianAI V3 — Voice Command Assistant

Purpose
-------
Provides a hands-free interface for the existing deterministic session copilot.

Supported commands
------------------
- "Driver Guardian, summarise the session"
- "Driver Guardian, why did you alert me"
- "Driver Guardian, explain calibration"
- "Driver Guardian, compare raw and calibrated risk"
- "Driver Guardian, how long was I in each state"
- "Driver Guardian, what should the driver do"
- "Driver Guardian, mute"
- "Driver Guardian, unmute"
- "Driver Guardian, help"
- "Driver Guardian, stop listening"

Safety architecture
-------------------
This assistant does not control fatigue detection, model predictions,
calibration, temporal thresholds, or alerts. It only interprets commands
and explains metrics produced by the deterministic monitoring system.

Jupyter examples
----------------
%run driverguardian_voice_commands_v3.py --keyboard
%run driverguardian_voice_commands_v3.py --microphone

One command only:
%run driverguardian_voice_commands_v3.py --question "Why did you alert me?"
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Optional

from driverguardian_copilot_v3 import DriverGuardianCopilot


WAKE_PHRASES = (
    "commander",
    "driverguard",
    "guardian",
)

STOP_PHRASES = (
    "stop listening",
    "exit",
    "quit",
    "close assistant",
)

MUTE_PHRASES = (
    "mute",
    "mute voice",
    "stop speaking",
)

UNMUTE_PHRASES = (
    "unmute",
    "unmute voice",
    "enable voice",
    "start speaking",
)


def normalise_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def remove_wake_phrase(text: str) -> tuple[str, bool]:
    normalised = normalise_text(text)

    for phrase in WAKE_PHRASES:
        if normalised.startswith(phrase):
            command = normalised[len(phrase):].strip(" ,.")
            return command, True

    return normalised, False


class MicrophoneListener:
    """
    SpeechRecognition microphone wrapper.

    Google recognition is used for this first prototype and therefore normally
    requires an internet connection. Keyboard mode remains available as a
    reliable fallback.
    """

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        phrase_time_limit: float = 8.0,
        ambient_seconds: float = 1.0,
    ) -> None:
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.ambient_seconds = ambient_seconds

        self.sr = None
        self.recognizer = None
        self.microphone = None
        self.error: Optional[str] = None

        try:
            import speech_recognition as sr

            self.sr = sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.7
            self.recognizer.non_speaking_duration = 0.4

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return (
            self.sr is not None
            and self.recognizer is not None
            and self.microphone is not None
        )

    def calibrate(self) -> bool:
        if not self.available:
            return False

        try:
            print(
                f"Calibrating microphone for {self.ambient_seconds:.1f} seconds..."
            )

            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=self.ambient_seconds,
                )

            print(
                "Microphone ready. Energy threshold:",
                round(self.recognizer.energy_threshold, 1),
            )

            return True

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def listen(self) -> Optional[str]:
        if not self.available:
            return None

        try:
            with self.microphone as source:
                print("\nListening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

            print("Recognising...")

            text = self.recognizer.recognize_google(
                audio,
                language="en-GB",
            )

            print("You said:", text)
            return text

        except self.sr.WaitTimeoutError:
            print("No speech detected.")
            return None

        except self.sr.UnknownValueError:
            print("I could not understand the command.")
            return None

        except self.sr.RequestError as exc:
            self.error = (
                "Speech-recognition service error: "
                f"{exc}"
            )
            print(self.error)
            return None

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print("Microphone error:", self.error)
            return None


class VoiceCommandAssistant:
    def __init__(
        self,
        *,
        report_path: Path | None = None,
        require_wake_phrase: bool = True,
        voice_enabled: bool = True,
    ) -> None:
        self.copilot = DriverGuardianCopilot(
            report_path=report_path
        )

        self.require_wake_phrase = require_wake_phrase
        self.voice_enabled = voice_enabled
        self.running = True

    def speak(self, text: str) -> None:
        print()
        print("Commander ")
        print("-" * 72)
        print(text)
        print("-" * 72)

        if self.voice_enabled:
            spoken = self.copilot.speaker.speak(text)

            if not spoken:
                print("Spoken output is unavailable.")

                if self.copilot.speaker.error:
                    print(
                        "Speech reason:",
                        self.copilot.speaker.error,
                    )

    def command_help(self) -> str:
        return (
            "Say Driver Guardian followed by one of these commands: "
            "summarise the session; why did you alert me; explain calibration; "
            "compare raw and calibrated risk; how long was I in each state; "
            "what should the driver do; mute; unmute; or stop listening."
        )

    def route(self, original_text: str) -> str:
        command, wake_detected = remove_wake_phrase(
            original_text
        )

        if (
            self.require_wake_phrase
            and not wake_detected
        ):
            return (
                "Wake phrase not detected. Begin the command with "
                "Commander."
            )

        if not command:
            return (
                "Yes. " + self.command_help()
            )

        if any(
            phrase in command
            for phrase in STOP_PHRASES
        ):
            self.running = False
            return "Voice assistant stopped."

        if any(
            command == phrase
            or command.startswith(phrase + " ")
            for phrase in MUTE_PHRASES
        ):
            self.voice_enabled = False
            return (
                "Voice responses are now muted. "
                "Text responses will continue."
            )

        if any(
            command == phrase
            or command.startswith(phrase + " ")
            for phrase in UNMUTE_PHRASES
        ):
            self.voice_enabled = True
            return "Voice responses are now enabled."

        if command in {
            "help",
            "commands",
            "what can i say",
        }:
            return self.command_help()

        return self.copilot.answer(command)

    def handle(self, text: str) -> str:
        response = self.route(text)
        self.speak(response)
        return response


def keyboard_loop(
    assistant: VoiceCommandAssistant,
) -> None:
    print()
    print("=" * 72)
    print("DriverGuardianAI V3 — Keyboard Voice-Command Simulator")
    print("=" * 72)
    print("This mode tests the same command router without a microphone.")
    print(assistant.command_help())
    print("=" * 72)

    while assistant.running:
        try:
            text = input("\nCommand: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue

        assistant.handle(text)


def microphone_loop(
    assistant: VoiceCommandAssistant,
    listener: MicrophoneListener,
) -> None:
    if not listener.available:
        raise RuntimeError(
            "Microphone mode is unavailable. "
            f"Reason: {listener.error}"
        )

    if not listener.calibrate():
        raise RuntimeError(
            "Microphone calibration failed. "
            f"Reason: {listener.error}"
        )

    print()
    print("=" * 72)
    print("DriverGuardianAI V3 — Voice Command Assistant")
    print("=" * 72)
    print(assistant.command_help())
    print()
    print(
        "The assistant only responds when the phrase "
        "'Driver Guardian' is detected."
    )
    print("=" * 72)

    while assistant.running:
        text = listener.listen()

        if not text:
            continue

        assistant.handle(text)

        # A short delay reduces the chance that the microphone hears the
        # assistant's own spoken response.
        time.sleep(0.8)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DriverGuardianAI fixed-command "
            "voice assistant"
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--keyboard",
        action="store_true",
        help=(
            "Test commands through keyboard input."
        ),
    )

    mode.add_argument(
        "--microphone",
        action="store_true",
        help=(
            "Listen for spoken microphone commands."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Optional session-report JSON path."
        ),
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help=(
            "Process one command and exit."
        ),
    )

    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help=(
            "Do not require 'Driver Guardian'."
        ),
    )

    parser.add_argument(
        "--mute",
        action="store_true",
        help=(
            "Start with spoken output muted."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    assistant = VoiceCommandAssistant(
        report_path=arguments.report,
        require_wake_phrase=(
            not arguments.no_wake_word
        ),
        voice_enabled=(
            not arguments.mute
        ),
    )

    if arguments.question:
        assistant.handle(
            arguments.question
        )
        return

    if arguments.microphone:
        listener = MicrophoneListener()
        microphone_loop(
            assistant,
            listener,
        )
        return

    keyboard_loop(
        assistant
    )


if __name__ == "__main__":
    main()