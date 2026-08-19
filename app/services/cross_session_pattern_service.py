from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any
import json
import math


class CrossSessionPatternService:
    """Deterministic longitudinal analysis over local Decision Memory.

    V9.1 identifies repeated patterns and similar historical sessions. It does
    not retrain the fatigue model, alter calibration, change alert thresholds,
    or infer medical causation from correlations.
    """

    VERSION = "9.2-historical-evidence-pattern-explainability-v1"
    ELEVATED_THRESHOLD = 0.65
    RECOVERY_THRESHOLD = 0.45

    def __init__(self, root: Path) -> None:
        self.root = root
        self.memory_dir = root / "guardian_data" / "decision_memory"

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value if value is not None else default)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _period(value: Any) -> str:
        try:
            hour = datetime.fromisoformat(str(value)).hour
        except (TypeError, ValueError):
            return "unknown"
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"

    def _sessions(self, profile_name: str) -> list[dict[str, Any]]:
        if not self.memory_dir.exists():
            return []
        wanted = str(profile_name or "").strip().casefold()
        sessions: list[dict[str, Any]] = []
        for path in self.memory_dir.glob("decision_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            current = str(payload.get("driver_profile") or "Guest").strip().casefold()
            if wanted and current != wanted:
                continue
            if len(payload.get("samples", []) or []) < 5:
                continue
            sessions.append(payload)
        sessions.sort(key=lambda item: str(item.get("started_at") or ""))
        return sessions[-40:]

    def _session_features(self, session: dict[str, Any]) -> dict[str, Any]:
        samples = session.get("samples", []) or []
        reliable = [
            s for s in samples
            if str(s.get("perception_state") or "").lower() != "insufficient"
        ]
        risks = [self._number(s.get("advisory_risk")) for s in samples]
        ears = [self._number(s.get("ear")) for s in reliable if self._number(s.get("ear")) > 0]
        yawns = [self._number(s.get("yawn_score")) for s in reliable]
        tilts = [abs(self._number(s.get("head_tilt"))) for s in reliable]
        perception = [str(s.get("perception_state") or "").lower() for s in samples]

        first_elevated = None
        peak_index = None
        if risks:
            peak_index = max(range(len(risks)), key=risks.__getitem__)
        for i, risk in enumerate(risks):
            if risk >= self.ELEVATED_THRESHOLD:
                first_elevated = self._number(samples[i].get("elapsed_seconds"))
                break

        recovery_seconds = None
        if peak_index is not None and risks[peak_index] >= self.ELEVATED_THRESHOLD:
            start_seconds = self._number(samples[peak_index].get("elapsed_seconds"))
            for later in samples[peak_index + 1:]:
                if self._number(later.get("advisory_risk")) <= self.RECOVERY_THRESHOLD:
                    recovery_seconds = max(
                        0.0,
                        self._number(later.get("elapsed_seconds")) - start_seconds,
                    )
                    break

        reason_counter: Counter[str] = Counter()
        contexts = {"weather": Counter(), "road": Counter(), "light": Counter(), "occlusion": Counter()}
        for sample in samples:
            for code in str(sample.get("perception_reason_codes") or "").split(";"):
                code = code.strip()
                if code:
                    reason_counter[code] += 1
            for field, key in (("weather", "weather"), ("road_condition", "road"), ("external_light", "light"), ("resolved_occlusion", "occlusion")):
                value = str(sample.get(field) or "").strip().lower()
                if value and value not in {"unknown", "none"}:
                    contexts[key][value] += 1

        near_miss_like = 0
        for i in range(1, len(risks)):
            if risks[i-1] >= self.ELEVATED_THRESHOLD and risks[i] <= self.RECOVERY_THRESHOLD:
                near_miss_like += 1

        return {
            "id": session.get("id"),
            "started_at": session.get("started_at"),
            "period": self._period(session.get("started_at")),
            "sample_count": len(samples),
            "duration_seconds": max([self._number(s.get("elapsed_seconds")) for s in samples] or [0.0]),
            "average_risk": fmean(risks) if risks else 0.0,
            "peak_risk": max(risks, default=0.0),
            "first_elevated_seconds": first_elevated,
            "recovery_seconds": recovery_seconds,
            "average_ear": fmean(ears) if ears else None,
            "average_yawn": fmean(yawns) if yawns else 0.0,
            "average_tilt": fmean(tilts) if tilts else 0.0,
            "insufficient_rate": perception.count("insufficient") / len(perception) if perception else 0.0,
            "degraded_rate": perception.count("degraded") / len(perception) if perception else 0.0,
            "top_reasons": reason_counter.most_common(5),
            "contexts": {k: (v.most_common(1)[0][0] if v else None) for k,v in contexts.items()},
            "near_miss_like": near_miss_like,
        }

    @staticmethod
    def _pattern(
        code: str,
        title: str,
        trend: str,
        confidence: float,
        supporting_sessions: int,
        detail: str,
        evidence: list[str],
        *,
        relevance: str = "historical",
        trust_note: str = "Correlation only; Guardian does not infer medical causation.",
        supporting_session_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "title": title,
            "trend": trend,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "supporting_sessions": int(supporting_sessions),
            "detail": detail,
            "evidence": evidence[:5],
            "relevance": relevance,
            "trust_note": trust_note,
            "supporting_session_ids": [str(value) for value in (supporting_session_ids or []) if value],
        }

    def snapshot(
        self,
        *,
        profile_name: str,
        current: dict[str, Any],
        passport_validation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sessions = self._sessions(profile_name)
        features = [self._session_features(s) for s in sessions]
        patterns: list[dict[str, Any]] = []

        passport_state = str((passport_validation or {}).get("state") or "unavailable").lower()
        history_trust = "normal" if passport_state == "valid" else "caution" if passport_state == "watch" else "restricted"

        elevated = [f for f in features if f["first_elevated_seconds"] is not None]
        if len(elevated) >= 2:
            times = [f["first_elevated_seconds"] for f in elevated]
            med = median(times)
            confidence = min(0.92, 0.45 + len(elevated) * 0.06)
            patterns.append(self._pattern(
                "RISK_ESCALATION_TIMING",
                "Recurring risk-escalation timing",
                "recurrent",
                confidence,
                len(elevated),
                f"Elevated advisory risk appeared at a median {med:.0f} seconds across {len(elevated)} sessions.",
                [f"{f['period']} session: first elevated at {f['first_elevated_seconds']:.0f}s" for f in elevated[-5:]],
                supporting_session_ids=[f["id"] for f in elevated],
            ))

        recoveries = [f for f in features if f["recovery_seconds"] is not None]
        if len(recoveries) >= 2:
            med = median([f["recovery_seconds"] for f in recoveries])
            patterns.append(self._pattern(
                "RECOVERY_BEHAVIOUR",
                "Repeated recovery after elevated risk",
                "recurrent",
                min(0.9, 0.42 + len(recoveries) * 0.07),
                len(recoveries),
                f"Risk returned to the recovery band after a median {med:.0f} seconds in {len(recoveries)} sessions.",
                [f"Recovery in {f['recovery_seconds']:.0f}s after a {f['peak_risk']:.0%} peak" for f in recoveries[-5:]],
                supporting_session_ids=[f["id"] for f in recoveries],
            ))

        ear_sessions = [f for f in features if f["average_ear"] is not None]
        if len(ear_sessions) >= 4:
            half = max(2, len(ear_sessions)//2)
            older = median([f["average_ear"] for f in ear_sessions[:half]])
            newer = median([f["average_ear"] for f in ear_sessions[-half:]])
            change = (newer - older) / max(0.0001, older)
            if abs(change) < 0.035:
                trend = "stable"
            elif change < 0:
                trend = "decreasing"
            else:
                trend = "increasing"
            patterns.append(self._pattern(
                "EAR_LONGITUDINAL_TREND",
                "Personal EAR trend across sessions",
                trend,
                min(0.88, 0.40 + len(ear_sessions) * 0.035),
                len(ear_sessions),
                f"Median session EAR changed from {older:.3f} to {newer:.3f} ({change:+.1%}).",
                [f"Older median {older:.3f}", f"Recent median {newer:.3f}", f"Relative change {change:+.1%}"],
                trust_note="Longitudinal behavioural correlation only; this is not a medical trend diagnosis.",
                supporting_session_ids=[f["id"] for f in ear_sessions],
            ))

        limitation_sessions = [f for f in features if f["insufficient_rate"] >= 0.10 or f["degraded_rate"] >= 0.25]
        reason_counts: Counter[str] = Counter()
        for f in limitation_sessions:
            reason_counts.update({code: count for code, count in f["top_reasons"]})
        if len(limitation_sessions) >= 2:
            top = reason_counts.most_common(3)
            patterns.append(self._pattern(
                "PERCEPTION_RELIABILITY_PATTERN",
                "Recurring perception limitations",
                "recurrent",
                min(0.9, 0.44 + len(limitation_sessions) * 0.05),
                len(limitation_sessions),
                f"Camera observability was degraded or insufficient in {len(limitation_sessions)} historical sessions.",
                [f"{code}: {count} recorded samples" for code,count in top] or ["Repeated degraded/insufficient perception"],
                trust_note="This describes camera observability, not fatigue or driver identity.",
                supporting_session_ids=[f["id"] for f in limitation_sessions],
            ))

        by_period: dict[str, list[float]] = {}
        for f in features:
            by_period.setdefault(f["period"], []).append(f["peak_risk"])
        eligible_periods = {k:v for k,v in by_period.items() if k != "unknown" and len(v) >= 2}
        if eligible_periods:
            highest_period, values = max(eligible_periods.items(), key=lambda item: median(item[1]))
            patterns.append(self._pattern(
                "TIME_OF_DAY_RISK_PATTERN",
                "Time-of-day risk pattern",
                "recurrent",
                min(0.84, 0.40 + len(values) * 0.06),
                len(values),
                f"{highest_period.title()} sessions have the highest median peak advisory risk at {median(values):.0%}.",
                [f"{period}: {len(vals)} sessions, median peak {median(vals):.0%}" for period,vals in sorted(eligible_periods.items())],
                trust_note="Time correlation only; external causes are not inferred.",
                supporting_session_ids=[f["id"] for f in features if f["period"] == highest_period],
            ))

        near_sessions = [f for f in features if f["near_miss_like"] > 0]
        if len(near_sessions) >= 2:
            total = sum(f["near_miss_like"] for f in near_sessions)
            patterns.append(self._pattern(
                "NEAR_MISS_RECURRENCE",
                "Repeated escalation-and-recovery sequences",
                "recurrent",
                min(0.88, 0.42 + len(near_sessions) * 0.06),
                len(near_sessions),
                f"Guardian found {total} elevated-to-recovery transitions across {len(near_sessions)} sessions.",
                [f"{f['period']} session: {f['near_miss_like']} transition(s)" for f in near_sessions[-5:]],
                supporting_session_ids=[f["id"] for f in near_sessions],
            ))

        # Current-session similarity. Uses only transparent summary distances.
        current_risk = self._number(current.get("risk"))
        current_ear = self._number(current.get("ear"))
        current_perception = str(current.get("perception_state") or "standby").lower()
        current_period = str(current.get("period") or "unknown").lower()
        similar: list[dict[str, Any]] = []
        for f in features:
            risk_similarity = max(0.0, 1.0 - abs(f["average_risk"] - current_risk) / 0.65)
            if current_ear > 0 and f["average_ear"]:
                ear_similarity = max(0.0, 1.0 - abs(f["average_ear"] - current_ear) / max(0.05, current_ear * 0.35))
            else:
                ear_similarity = 0.5
            period_similarity = 1.0 if f["period"] == current_period else 0.55
            perception_similarity = 1.0
            if current_perception == "insufficient":
                perception_similarity = min(1.0, f["insufficient_rate"] * 2.5)
            elif current_perception == "degraded":
                perception_similarity = min(1.0, f["degraded_rate"] * 2.0 + 0.35)
            else:
                perception_similarity = max(0.35, 1.0 - f["insufficient_rate"])
            score = 0.45*risk_similarity + 0.25*ear_similarity + 0.15*period_similarity + 0.15*perception_similarity
            similar.append({
                "session_id": f["id"],
                "started_at": f["started_at"],
                "period": f["period"],
                "similarity": round(score, 4),
                "average_risk": round(f["average_risk"], 4),
                "peak_risk": round(f["peak_risk"], 4),
                "first_elevated_seconds": f["first_elevated_seconds"],
                "recovery_seconds": f["recovery_seconds"],
                "evidence": [
                    f"risk similarity {risk_similarity:.0%}",
                    f"EAR similarity {ear_similarity:.0%}",
                    f"time-period match {'yes' if f['period']==current_period else 'no'}",
                ],
            })
        similar.sort(key=lambda item: item["similarity"], reverse=True)
        similar = similar[:3]

        if len(features) < 2:
            status = "insufficient_data"
            summary = "More completed Decision Memory sessions are needed before Guardian can identify cross-session patterns."
        elif history_trust == "restricted":
            status = "restricted"
            summary = "Historical patterns are available, but personalised interpretation is restricted by Passport trust."
        elif patterns:
            status = "available"
            summary = f"Guardian identified {len(patterns)} repeatable cross-session pattern(s) from {len(features)} local sessions."
        else:
            status = "developing"
            summary = f"Guardian has {len(features)} usable local sessions, but no strong repeated pattern is established yet."

        top_pattern = max(patterns, key=lambda item: item["confidence"], default=None)
        return {
            "version": self.VERSION,
            "profile_name": profile_name,
            "status": status,
            "history_trust": history_trust,
            "session_count": len(features),
            "pattern_count": len(patterns),
            "summary": summary,
            "top_pattern": top_pattern,
            "patterns": patterns,
            "similar_sessions": similar,
            "current_session": {
                "risk": round(current_risk, 4),
                "ear": round(current_ear, 6),
                "perception_state": current_perception,
                "period": current_period,
            },
            "method": "Deterministic descriptive statistics and transparent distance-based session similarity over local Decision Memory.",
            "safety_boundary": "Cross-session patterns describe repeated associations only. They do not retrain the model, alter calibration, change alerts or establish medical causation.",
            "cag_context": {
                "schema": "guardian-cross-session-context-v1",
                "status": status,
                "session_count": len(features),
                "top_patterns": [item["code"] for item in sorted(patterns, key=lambda item:item["confidence"], reverse=True)[:3]],
                "similar_session_ids": [item["session_id"] for item in similar],
            },
        }
