from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable


class AutomaticReportService:
    """Generate the existing V3 HTML/JSON report after a completed session."""

    def __init__(
        self,
        root: Path,
        event_callback: Callable[[str, str, str], None],
    ) -> None:
        self.root = root
        self.event_callback = event_callback
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {
            "state": "IDLE",
            "message": "No report is being generated.",
            "session_log": None,
            "html_report": None,
            "json_report": None,
            "error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def generate_async(self, session_log: str | Path | None) -> bool:
        if not session_log:
            self.event_callback(
                "REPORT",
                "No session log was available for automatic report generation.",
                "warning",
            )
            return False

        path = Path(session_log)
        if not path.exists():
            self.event_callback(
                "REPORT",
                f"Session log was not found: {path}",
                "warning",
            )
            return False

        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._status = {
                "state": "QUEUED",
                "message": "Report generation queued.",
                "session_log": str(path),
                "html_report": None,
                "json_report": None,
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._generate,
                args=(path,),
                name="guardian-report-generator",
                daemon=True,
            )
            self._thread.start()
        return True

    def _generate(self, session_log: Path) -> None:
        generator = self.root / "generate_session_report_v3.py"
        report_dir = self.root / "reports" / "v3"
        report_dir.mkdir(parents=True, exist_ok=True)

        if not generator.exists():
            self._fail("generate_session_report_v3.py was not found.")
            return

        before = {path.resolve() for path in report_dir.glob("session_report_*")}

        with self._lock:
            self._status["state"] = "GENERATING"
            self._status["message"] = "Building session charts and summary."

        self.event_callback(
            "REPORT",
            "Automatic session report generation started.",
            "info",
        )

        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    "--session",
                    str(session_log),
                    "--output-dir",
                    str(report_dir),
                ],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except Exception as error:
            self._fail(f"{type(error).__name__}: {error}")
            return

        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "Unknown report error").strip()
            self._fail(detail[-1200:])
            return

        after = sorted(
            (path for path in report_dir.glob("session_report_*") if path.resolve() not in before),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        html_report = next((path for path in after if path.suffix.lower() == ".html"), None)
        json_report = next((path for path in after if path.suffix.lower() == ".json"), None)

        # Some report generators overwrite an existing matching output. Fall
        # back to the most recently modified report pair.
        if html_report is None:
            html_report = self._latest(report_dir, ".html")
        if json_report is None:
            json_report = self._latest(report_dir, ".json")

        with self._lock:
            self._status = {
                "state": "COMPLETE",
                "message": "Session report generated successfully.",
                "session_log": str(session_log),
                "html_report": str(html_report) if html_report else None,
                "json_report": str(json_report) if json_report else None,
                "error": None,
            }

        self.event_callback(
            "REPORT",
            "Session report ready in the Reports page.",
            "success",
        )

    @staticmethod
    def _latest(directory: Path, suffix: str) -> Path | None:
        matches = sorted(
            directory.glob(f"session_report_*{suffix}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return matches[0] if matches else None

    def _fail(self, message: str) -> None:
        with self._lock:
            self._status.update(
                {
                    "state": "ERROR",
                    "message": "Automatic report generation failed.",
                    "error": message,
                }
            )
        self.event_callback("REPORT", f"Report generation failed: {message}", "danger")
