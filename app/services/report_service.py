from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.report_dir = root / "reports" / "v3"

    def list_reports(self) -> list[dict[str, Any]]:
        if not self.report_dir.exists():
            return []
        reports = []
        for path in sorted(self.report_dir.glob("session_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            html_path = path.with_suffix(".html")
            reports.append(
                {
                    "id": path.stem,
                    "name": path.stem.replace("session_report_driver_guardian_v3_", "Session "),
                    "json_file": path.name,
                    "html_file": html_path.name if html_path.exists() else None,
                    "modified": path.stat().st_mtime,
                    "duration_seconds": payload.get("duration_seconds", 0),
                    "alert_count": payload.get("alert_count", 0),
                    "maximum_risk": payload.get("maximum_smoothed_probability", 0),
                    "dominant_signal": payload.get("dominant_risk_signal", "unknown"),
                }
            )
        return reports

    def resolve(self, report_id: str, suffix: str = ".json") -> Path:
        candidate = (self.report_dir / f"{report_id}{suffix}").resolve()
        if self.report_dir.resolve() not in candidate.parents:
            raise ValueError("Invalid report path")
        if not candidate.exists():
            raise FileNotFoundError(report_id)
        return candidate
