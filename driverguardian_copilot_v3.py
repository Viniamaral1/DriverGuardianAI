from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "v3"


def latest_report() -> Path:
    files = [p for p in REPORT_DIR.glob("session_report_*.json") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No JSON reports found in {REPORT_DIR}")
    return max(files, key=lambda p: p.stat().st_mtime)


def duration_text(seconds: float | int | None) -> str:
    if seconds is None:
        return "not available"
    total = max(0, int(round(float(seconds))))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def pct(value: float | int | None) -> str:
    if value is None:
        return "not available"
    return f"{100 * float(value):.1f}%"


class Speaker:
    def __init__(self) -> None:
        self.engine = None
        self.error = None
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 175)
            self.engine.setProperty("volume", 0.95)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.engine is not None

    def speak(self, text: str) -> bool:
        if self.engine is None:
            return False
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False


class DriverGuardianCopilot:
    def __init__(self, report_path: str | Path | None = None) -> None:
        self.report_path = Path(report_path).resolve() if report_path else latest_report()
        with self.report_path.open("r", encoding="utf-8") as file:
            self.report: dict[str, Any] = json.load(file)
        self.speaker = Speaker()

    def state_seconds(self, state: str) -> float:
        return float(self.report.get("state_seconds", {}).get(state, 0.0))

    def state_pct(self, state: str) -> float:
        return float(self.report.get("state_percentages", {}).get(state, 0.0))

    def summary(self) -> str:
        return (
            f"The session lasted {duration_text(self.report.get('duration_seconds'))}, "
            f"processed {int(self.report.get('logged_frames', 0)):,} frames at about "
            f"{float(self.report.get('estimated_fps', 0)):.1f} FPS, and generated "
            f"{int(self.report.get('alert_count', 0))} controlled alerts. "
            f"Monitoring accounted for {self.state_pct('MONITORING'):.1f}% of the session, "
            f"warning {self.state_pct('WARNING'):.1f}%, and critical "
            f"{self.state_pct('CRITICAL'):.1f}%. The maximum smoothed risk was "
            f"{pct(self.report.get('maximum_smoothed_probability'))}. "
            f"The dominant risk signal was "
            f"{str(self.report.get('dominant_risk_signal', 'unknown')).lower()}."
        )

    def explain_alerts(self) -> str:
        alerts = int(self.report.get("alert_count", 0))
        if alerts == 0:
            return (
                "No controlled alert was triggered. Brief warning or critical states "
                "may still have occurred, but they did not satisfy the sustained-duration "
                "and cooldown rules."
            )

        signal = str(self.report.get("dominant_risk_signal", "behavioural evidence")).lower()
        baseline = float(self.report.get("baseline_ear", 0))
        minimum = float(self.report.get("minimum_ear", 0))
        maximum = pct(self.report.get("maximum_smoothed_probability"))
        critical = duration_text(self.state_seconds("CRITICAL"))

        return (
            f"The system generated {alerts} controlled alerts. The strongest contributor "
            f"was {signal}. The personal EAR baseline was {baseline:.3f}, while the "
            f"minimum EAR fell to {minimum:.3f}. Maximum smoothed risk reached {maximum}, "
            f"and the session spent {critical} in the critical state. An audible alert "
            f"was only allowed after critical evidence persisted long enough, and cooldown "
            f"logic prevented repeated alerts."
        )

    def explain_model(self) -> str:
        raw = pct(self.report.get("average_raw_model_probability"))
        decision = pct(self.report.get("average_decision_probability"))
        smooth = pct(self.report.get("average_smoothed_probability"))

        return (
            f"The raw model averaged {raw}, while calibrated decision risk averaged "
            f"{decision}, and temporally smoothed risk averaged {smooth}. This difference "
            f"is intentional: the live decision layer reduces the influence of an "
            f"overconfident raw prediction unless personal behavioural evidence supports it."
        )

    def explain_calibration(self) -> str:
        ear = float(self.report.get("baseline_ear", 0))
        yawn = float(self.report.get("baseline_yawn", 0))
        tilt = float(self.report.get("baseline_tilt", 0))

        return (
            f"Calibration established a personal EAR baseline of {ear:.4f}, a yawn "
            f"baseline of {yawn:.4f}, and a head-tilt baseline of {tilt:.2f} degrees. "
            f"Live behaviour was compared with these values instead of relying on one "
            f"fixed threshold for every driver."
        )

    def state_breakdown(self) -> str:
        return (
            f"Monitoring lasted {duration_text(self.state_seconds('MONITORING'))}, "
            f"warning lasted {duration_text(self.state_seconds('WARNING'))}, "
            f"critical lasted {duration_text(self.state_seconds('CRITICAL'))}, and "
            f"no-face time was {duration_text(self.state_seconds('NO FACE'))}."
        )

    def recommendation(self) -> str:
        alerts = int(self.report.get("alert_count", 0))
        critical_pct = self.state_pct("CRITICAL")
        max_risk = float(self.report.get("maximum_smoothed_probability", 0))

        if alerts >= 2 or critical_pct >= 10 or max_risk >= 0.90:
            return (
                "This session contained substantial fatigue-like evidence. In a real "
                "driving situation, the driver should stop at a safe location and take "
                "a proper break. This portfolio system is not a certified automotive "
                "safety device."
            )

        if alerts == 1 or max_risk >= 0.82:
            return (
                "The session contained a sustained critical event. In a real driving "
                "context, that should be treated as a prompt to take a safe break."
            )

        return (
            "No sustained controlled alert was recorded. Continue monitoring and never "
            "use this demonstration as a substitute for safe driving judgment."
        )

    def help(self) -> str:
        return (
            "Try: summarise the session; why did you alert me; explain calibration; "
            "compare raw and calibrated risk; how long was I in each state; "
            "what should the driver do."
        )

    def answer(self, question: str) -> str:
        q = re.sub(r"\s+", " ", question.strip().lower())

        if any(x in q for x in ["summarise", "summarize", "overview", "session summary"]):
            return self.summary()

        if any(x in q for x in ["why did", "why alert", "what caused", "cause", "contributor"]):
            return self.explain_alerts()

        if any(x in q for x in ["raw model", "calibrated risk", "decision risk", "overconfident"]):
            return self.explain_model()

        if any(x in q for x in ["calibration", "baseline"]):
            return self.explain_calibration()

        if any(x in q for x in ["how long", "state", "warning time", "critical time"]):
            return self.state_breakdown()

        if any(x in q for x in ["what should", "recommend", "safe break", "driver do"]):
            return self.recommendation()

        if any(x in q for x in ["help", "what can i ask", "commands"]):
            return self.help()

        return "I could not map that question confidently. " + self.help()

    def ask(self, question: str, speak: bool = False) -> str:
        response = self.answer(question)
        print("\nDriverGuardian Copilot")
        print("-" * 72)
        print(response)
        print("-" * 72)

        if speak and not self.speaker.speak(response):
            print("Speech output is unavailable.")
            if self.speaker.error:
                print("Reason:", self.speaker.error)

        return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--question", type=str)
    parser.add_argument("--speak", action="store_true")
    args = parser.parse_args()

    copilot = DriverGuardianCopilot(args.report)

    if args.question:
        copilot.ask(args.question, speak=args.speak)
        return

    print("=" * 72)
    print("DriverGuardianAI V3 Session Copilot")
    print("=" * 72)
    print("Report:", copilot.report_path)
    print("Speech available:", copilot.speaker.available)
    print(copilot.help())
    print("Type 'quit' to stop.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "stop"}:
            break
        copilot.ask(question, speak=args.speak)


if __name__ == "__main__":
    main()