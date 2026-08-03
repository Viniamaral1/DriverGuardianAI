from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PdfReportService:
    """Create a compact, shareable PDF from a generated Guardian OS JSON report."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.report_dir = root / "reports" / "v3"

    def generate(self, report_id: str) -> Path:
        json_path = self._resolve(report_id, ".json")
        pdf_path = self.report_dir / f"{report_id}.pdf"

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self._build_pdf(pdf_path, payload)
        return pdf_path

    def _resolve(self, report_id: str, suffix: str) -> Path:
        candidate = (self.report_dir / f"{report_id}{suffix}").resolve()
        if self.report_dir.resolve() not in candidate.parents:
            raise ValueError("Invalid report path")
        if not candidate.exists():
            raise FileNotFoundError(report_id)
        return candidate

    @staticmethod
    def _percentage(value: Any) -> str:
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def _number(value: Any, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    def _build_pdf(self, path: Path, payload: dict[str, Any]) -> None:
        styles = getSampleStyleSheet()
        title = ParagraphStyle(
            "GuardianTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0B6570"),
            alignment=TA_LEFT,
            spaceAfter=8,
        )
        section = ParagraphStyle(
            "GuardianSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#153844"),
            spaceBefore=10,
            spaceAfter=7,
        )
        body = ParagraphStyle(
            "GuardianBody",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#344B55"),
        )
        small = ParagraphStyle(
            "GuardianSmall",
            parent=body,
            fontSize=8,
            textColor=colors.HexColor("#6B7F87"),
        )

        document = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title="Guardian OS Session Metrics",
            author="DriverGuardianAI",
        )

        story = [
            Paragraph("Guardian OS — Session Metrics", title),
            Paragraph(
                "Locally generated driver-monitoring summary. Raw camera video is not included.",
                small,
            ),
            Spacer(1, 6),
        ]

        overview = [
            ["Generated", str(payload.get("generated_at", "—"))],
            ["Duration", f"{float(payload.get('duration_seconds', 0)):.1f} seconds"],
            ["Logged frames", str(payload.get("logged_frames", "—"))],
            ["Estimated FPS", self._number(payload.get("estimated_fps"), 1)],
            ["Alerts", str(payload.get("alert_count", 0))],
            ["Dominant signal", str(payload.get("dominant_risk_signal", "unknown"))],
        ]
        story.extend([
            Paragraph("Session overview", section),
            self._table(overview),
        ])

        risk = [
            ["Average fatigue risk", self._percentage(payload.get("average_smoothed_probability"))],
            ["Maximum fatigue risk", self._percentage(payload.get("maximum_smoothed_probability"))],
            ["Average model probability", self._percentage(payload.get("average_raw_model_probability"))],
            ["Maximum model probability", self._percentage(payload.get("maximum_raw_model_probability"))],
            ["Warning episodes", str(payload.get("warning_episodes", 0))],
            ["Critical episodes", str(payload.get("critical_episodes", 0))],
        ]
        story.extend([
            Paragraph("Risk assessment", section),
            self._table(risk),
        ])

        signals = [
            ["Baseline EAR", self._number(payload.get("baseline_ear"))],
            ["Average EAR", self._number(payload.get("average_ear"))],
            ["Minimum EAR", self._number(payload.get("minimum_ear"))],
            ["Maximum yawn score", self._number(payload.get("maximum_yawn_score"))],
            ["Baseline head tilt", self._number(payload.get("baseline_tilt"), 2)],
            ["Maximum head tilt", self._number(payload.get("maximum_head_tilt"), 2)],
        ]
        story.extend([
            Paragraph("Behavioural signals", section),
            self._table(signals),
        ])

        state_percentages = payload.get("state_percentages", {}) or {}
        state_rows = [
            [str(state).replace("_", " ").title(), f"{float(value):.1f}%"]
            for state, value in state_percentages.items()
        ]
        if state_rows:
            story.extend([
                Paragraph("Time by state", section),
                self._table(state_rows),
            ])

        max_risk = float(payload.get("maximum_smoothed_probability", 0) or 0)
        alerts = int(payload.get("alert_count", 0) or 0)
        if max_risk >= 0.82 or alerts > 0:
            summary = (
                "This session contained high-risk fatigue evidence or controlled alerts. "
                "Review the event timeline and consider whether a safe break was required."
            )
        elif max_risk >= 0.65:
            summary = (
                "This session contained warning-level fatigue evidence. Continued monitoring "
                "and an earlier break may be appropriate."
            )
        else:
            summary = (
                "The recorded session remained below the warning threshold for most of the drive."
            )

        story.extend([
            Paragraph("Guardian summary", section),
            Paragraph(summary, body),
            Spacer(1, 10),
            Paragraph(
                "Research notice: Guardian OS is not a certified medical or automotive safety device.",
                small,
            ),
        ])

        document.build(story)

    @staticmethod
    def _table(rows: list[list[str]]) -> Table:
        table = Table(rows, colWidths=[75 * mm, 85 * mm], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF4F5")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#203A43")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D8DC")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return table
