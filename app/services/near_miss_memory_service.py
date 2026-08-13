from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any


class NearMissMemoryService:
    """Derived historical analysis over Decision Memory samples.

    Near-Miss Memory never changes live Monitoring, model output or alerts.
    Episodes are reconstructed from the recorded advisory trace when a session
    is opened.
    """

    VERSION = "8.5-near-miss-v1"

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _timestamp(sample: dict[str, Any]) -> str | None:
        value = sample.get("timestamp")
        return str(value) if value else None

    @staticmethod
    def _elapsed(sample: dict[str, Any]) -> int:
        try:
            return int(float(sample.get("elapsed_seconds", 0) or 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _evidence_map(cls, sample: dict[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for item in sample.get("evidence", []) or []:
            key = str(item.get("key") or "").strip()
            if key:
                result[key] = cls._number(item.get("value"))
        return result

    @classmethod
    def _contributors(
        cls,
        samples: list[dict[str, Any]],
        start: int,
        end: int,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        peaks: dict[str, tuple[float, str]] = {}
        for sample in samples[max(0, start): min(len(samples), end + 1)]:
            for item in sample.get("evidence", []) or []:
                key = str(item.get("key") or "").strip()
                if not key:
                    continue
                value = cls._number(item.get("value"))
                label = str(item.get("label") or key.replace("_", " ").title())
                previous = peaks.get(key)
                if previous is None or value > previous[0]:
                    peaks[key] = (value, label)

        ordered = sorted(
            peaks.items(),
            key=lambda pair: pair[1][0],
            reverse=True,
        )
        return [
            {
                "key": key,
                "label": label,
                "peak": round(value, 4),
            }
            for key, (value, label) in ordered[:limit]
            if value >= 0.12
        ]

    @classmethod
    def _episode(
        cls,
        samples: list[dict[str, Any]],
        *,
        episode_type: str,
        title: str,
        start_index: int,
        peak_index: int,
        end_index: int,
        confidence: float,
        outcome: str,
        explanation: str,
        recovery_index: int | None = None,
    ) -> dict[str, Any]:
        start_index = max(0, min(len(samples) - 1, start_index))
        peak_index = max(start_index, min(len(samples) - 1, peak_index))
        end_index = max(peak_index, min(len(samples) - 1, end_index))
        start = samples[start_index]
        peak = samples[peak_index]
        end = samples[end_index]
        duration = max(0, cls._elapsed(end) - cls._elapsed(start))

        return {
            "type": episode_type,
            "title": title,
            "start_index": start_index,
            "peak_index": peak_index,
            "end_index": end_index,
            "recovery_index": recovery_index,
            "start_timestamp": cls._timestamp(start),
            "peak_timestamp": cls._timestamp(peak),
            "recovery_timestamp": (
                cls._timestamp(samples[recovery_index])
                if recovery_index is not None
                else None
            ),
            "duration_seconds": duration,
            "start_risk": round(cls._number(start.get("advisory_risk")), 4),
            "peak_risk": round(cls._number(peak.get("advisory_risk")), 4),
            "end_risk": round(cls._number(end.get("advisory_risk")), 4),
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "outcome": outcome,
            "explanation": explanation,
            "contributing_evidence": cls._contributors(
                samples, start_index, end_index
            ),
            "visual_evidence_available": False,
            "visual_evidence_index": None,
        }

    @classmethod
    def _near_alerts(
        cls,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        n = len(samples)
        i = 0
        while i < n:
            risk = cls._number(samples[i].get("advisory_risk"))
            if risk < 0.58:
                i += 1
                continue

            start = max(0, i - 1)
            peak = i
            j = i
            starting_alerts = int(samples[start].get("alert_count", 0) or 0)
            alert_happened = False

            while j + 1 < n:
                j += 1
                current = cls._number(samples[j].get("advisory_risk"))
                if current > cls._number(samples[peak].get("advisory_risk")):
                    peak = j
                if int(samples[j].get("alert_count", 0) or 0) > starting_alerts:
                    alert_happened = True
                if current < 0.42:
                    break
                if j - i >= 12:
                    break

            peak_risk = cls._number(samples[peak].get("advisory_risk"))
            recovered = cls._number(samples[j].get("advisory_risk")) < 0.42
            if peak_risk >= 0.68 and recovered and not alert_happened:
                confidence = (
                    cls._number(samples[peak].get("decision_confidence")) * 0.65
                    + min(1.0, peak_risk / 0.85) * 0.35
                )
                episodes.append(
                    cls._episode(
                        samples,
                        episode_type="near_alert",
                        title="Near-alert recovery",
                        start_index=start,
                        peak_index=peak,
                        end_index=j,
                        recovery_index=j,
                        confidence=confidence,
                        outcome="Recovered before a recorded alert",
                        explanation=(
                            f"Advisory risk rose to {peak_risk:.0%} and then "
                            "returned below the watch range without the recorded "
                            "alert count increasing."
                        ),
                    )
                )
                i = j + 1
            else:
                i += 1
        return cls._dedupe(episodes, radius=5)

    @classmethod
    def _recoveries(
        cls,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        n = len(samples)
        for peak in range(n):
            peak_risk = cls._number(samples[peak].get("advisory_risk"))
            if peak_risk < 0.72:
                continue
            # Local peak only.
            before = cls._number(samples[peak - 1].get("advisory_risk")) if peak else -1
            after = cls._number(samples[peak + 1].get("advisory_risk")) if peak + 1 < n else -1
            if peak_risk < before or peak_risk < after:
                continue

            for end in range(peak + 1, min(n, peak + 9)):
                end_risk = cls._number(samples[end].get("advisory_risk"))
                if peak_risk - end_risk >= 0.28 and end_risk <= 0.48:
                    start = max(0, peak - 2)
                    confidence = (
                        cls._number(samples[peak].get("decision_confidence")) * 0.7
                        + min(1.0, (peak_risk - end_risk) / 0.45) * 0.3
                    )
                    episodes.append(
                        cls._episode(
                            samples,
                            episode_type="recovery",
                            title="Driver-state recovery",
                            start_index=start,
                            peak_index=peak,
                            end_index=end,
                            recovery_index=end,
                            confidence=confidence,
                            outcome="Risk reduced after a high-risk period",
                            explanation=(
                                f"Risk fell from {peak_risk:.0%} to "
                                f"{end_risk:.0%} over the following recorded samples."
                            ),
                        )
                    )
                    break
        return cls._dedupe(episodes, radius=5)

    @classmethod
    def _escalations(
        cls,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        order = {"standby": 0, "low": 1, "elevated": 2, "high": 3}
        episodes: list[dict[str, Any]] = []
        n = len(samples)

        for start in range(n):
            start_band = str(samples[start].get("risk_band") or "standby")
            if order.get(start_band, 0) > 1:
                continue
            max_rank = order.get(start_band, 0)
            peak = start
            for end in range(start + 1, min(n, start + 9)):
                band = str(samples[end].get("risk_band") or "standby")
                rank = order.get(band, 0)
                if cls._number(samples[end].get("advisory_risk")) > cls._number(
                    samples[peak].get("advisory_risk")
                ):
                    peak = end
                max_rank = max(max_rank, rank)
                if max_rank >= 3:
                    rise = (
                        cls._number(samples[peak].get("advisory_risk"))
                        - cls._number(samples[start].get("advisory_risk"))
                    )
                    if rise >= 0.30:
                        confidence = sum(
                            cls._number(samples[k].get("decision_confidence"))
                            for k in range(start, end + 1)
                        ) / max(1, end - start + 1)
                        episodes.append(
                            cls._episode(
                                samples,
                                episode_type="escalation",
                                title="Risk escalation",
                                start_index=start,
                                peak_index=peak,
                                end_index=end,
                                confidence=confidence,
                                outcome="Risk progressed into the high band",
                                explanation=(
                                    "The advisory trace progressed from a lower "
                                    "risk band into high risk within a short temporal window."
                                ),
                            )
                        )
                    break
        return cls._dedupe(episodes, radius=6)

    @classmethod
    def _weak_signal_accumulation(
        cls,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        active_start: int | None = None
        peak: int | None = None

        for index, sample in enumerate(samples):
            evidence = cls._evidence_map(sample)
            weak = [
                value for value in evidence.values()
                if 0.22 <= value < 0.65
            ]
            risk = cls._number(sample.get("advisory_risk"))
            qualifies = len(weak) >= 2 and 0.42 <= risk < 0.78

            if qualifies:
                if active_start is None:
                    active_start = index
                    peak = index
                if peak is None or risk > cls._number(
                    samples[peak].get("advisory_risk")
                ):
                    peak = index
            elif active_start is not None and peak is not None:
                end = index - 1
                if end - active_start >= 1:
                    confidence = sum(
                        cls._number(samples[k].get("decision_confidence"))
                        for k in range(active_start, end + 1)
                    ) / max(1, end - active_start + 1)
                    episodes.append(
                        cls._episode(
                            samples,
                            episode_type="weak_signal_accumulation",
                            title="Weak signals accumulated",
                            start_index=active_start,
                            peak_index=peak,
                            end_index=end,
                            confidence=confidence,
                            outcome="Multiple sub-threshold signals aligned",
                            explanation=(
                                "At least two individually moderate evidence "
                                "signals persisted together while advisory risk "
                                "was elevated but not consistently critical."
                            ),
                        )
                    )
                active_start = None
                peak = None

        if active_start is not None and peak is not None:
            end = len(samples) - 1
            if end - active_start >= 1:
                confidence = sum(
                    cls._number(samples[k].get("decision_confidence"))
                    for k in range(active_start, end + 1)
                ) / max(1, end - active_start + 1)
                episodes.append(
                    cls._episode(
                        samples,
                        episode_type="weak_signal_accumulation",
                        title="Weak signals accumulated",
                        start_index=active_start,
                        peak_index=peak,
                        end_index=end,
                        confidence=confidence,
                        outcome="Multiple sub-threshold signals aligned",
                        explanation=(
                            "At least two individually moderate evidence signals "
                            "persisted together in the advisory trace."
                        ),
                    )
                )
        return episodes

    @classmethod
    def _uncertainty_episodes(
        cls,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        bad: list[bool] = []
        for sample in samples:
            confidence = cls._number(sample.get("decision_confidence"))
            signal = cls._number(sample.get("signal_quality"))
            image = cls._number(sample.get("image_quality"))
            perception = cls._number(sample.get("perception_confidence"))
            perception_state = str(sample.get("perception_state") or "").lower()
            occlusion = str(
                sample.get("raw_automatic_occlusion")
                or sample.get("automatic_occlusion")
                or ""
            ).lower()
            if perception_state:
                bad.append(
                    perception_state == "insufficient"
                    or (perception > 0 and perception < 0.45)
                )
            else:
                bad.append(
                    confidence < 0.58
                    or (signal > 0 and signal < 0.55)
                    or (image > 0 and image < 0.50)
                    or occlusion == "uncertain"
                )

        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, flag in enumerate(bad + [False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                end = index - 1
                if end - start + 1 >= 2:
                    runs.append((start, end))
                start = None

        # "Repeated uncertainty" means at least two runs, or one long run.
        if len(runs) < 2 and not any(end - start + 1 >= 4 for start, end in runs):
            return []

        episodes: list[dict[str, Any]] = []
        for start, end in runs:
            weakest = min(
                range(start, end + 1),
                key=lambda i: cls._number(samples[i].get("decision_confidence")),
            )
            mean_conf = sum(
                cls._number(samples[i].get("decision_confidence"))
                for i in range(start, end + 1)
            ) / max(1, end - start + 1)
            episodes.append(
                cls._episode(
                    samples,
                    episode_type="repeated_uncertainty",
                    title="Perception/decision uncertainty",
                    start_index=start,
                    peak_index=weakest,
                    end_index=end,
                    confidence=max(0.45, 1.0 - mean_conf),
                    outcome="Observation quality was repeatedly limited",
                    explanation=(
                        "Decision confidence, signal/image quality or automatic "
                        "occlusion reliability was degraded across multiple "
                        "recorded samples."
                    ),
                )
            )
        return episodes

    @classmethod
    def _baseline_drift(
        cls,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        valid = [
            (
                index,
                cls._number(sample.get("ear")),
                cls._number(sample.get("baseline_ear")),
            )
            for index, sample in enumerate(samples)
            if cls._number(sample.get("ear")) > 0
            and cls._number(sample.get("baseline_ear")) > 0
        ]
        if len(valid) < 8:
            return []

        window = max(3, min(5, len(valid) // 3))
        first = valid[:window]
        last = valid[-window:]
        first_ratio = sum(ear / base for _, ear, base in first) / len(first)
        last_ratio = sum(ear / base for _, ear, base in last) / len(last)
        drift = first_ratio - last_ratio

        if drift < 0.08:
            return []

        peak_index = min(
            (index for index, _, _ in valid),
            key=lambda idx: (
                cls._number(samples[idx].get("ear"))
                / max(0.0001, cls._number(samples[idx].get("baseline_ear")))
            ),
        )
        start_index = first[0][0]
        end_index = last[-1][0]
        confidence = min(0.90, 0.58 + drift * 1.6)

        return [
            cls._episode(
                samples,
                episode_type="baseline_drift",
                title="Personal baseline drift",
                start_index=start_index,
                peak_index=peak_index,
                end_index=end_index,
                confidence=confidence,
                outcome="EAR trend moved away from the calibrated baseline",
                explanation=(
                    f"The recent EAR-to-baseline ratio declined by approximately "
                    f"{drift:.0%} from the beginning to the end of the recorded trace."
                ),
            )
        ]

    @staticmethod
    def _dedupe(
        episodes: list[dict[str, Any]],
        *,
        radius: int,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for episode in sorted(
            episodes,
            key=lambda x: (
                x.get("peak_index", 0),
                -float(x.get("peak_risk", 0) or 0),
            ),
        ):
            if any(
                abs(
                    int(episode.get("peak_index", 0))
                    - int(existing.get("peak_index", 0))
                ) <= radius
                for existing in result
            ):
                continue
            result.append(episode)
        return result

    @classmethod
    def _attach_visual_evidence(
        cls,
        episodes: list[dict[str, Any]],
        visual_evidence: dict[str, Any] | None,
    ) -> None:
        groups = (visual_evidence or {}).get("events", []) or []
        for episode in episodes:
            peak = int(episode.get("peak_index", 0))
            best = None
            best_distance = 10**9
            for group in groups:
                try:
                    index = int(group.get("index", 0))
                except (TypeError, ValueError):
                    continue
                distance = abs(index - peak)
                if distance < best_distance:
                    best = group
                    best_distance = distance
            if best is not None and best_distance <= 2:
                episode["visual_evidence_available"] = bool(best.get("files"))
                episode["visual_evidence_index"] = int(best.get("index", 0))

    @classmethod
    def analyse(
        cls,
        samples: list[dict[str, Any]],
        *,
        visual_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(samples) < 2:
            return {
                "version": cls.VERSION,
                "episode_count": 0,
                "episodes": [],
                "by_type": {},
                "summary": "Not enough Decision Memory samples for Near-Miss analysis.",
            }

        episodes: list[dict[str, Any]] = []
        episodes.extend(cls._near_alerts(samples))
        episodes.extend(cls._recoveries(samples))
        episodes.extend(cls._escalations(samples))
        episodes.extend(cls._weak_signal_accumulation(samples))
        episodes.extend(cls._uncertainty_episodes(samples))
        episodes.extend(cls._baseline_drift(samples))

        # Remove exact same-type / same-peak duplicates and then sort by time.
        unique: dict[tuple[str, int], dict[str, Any]] = {}
        for episode in episodes:
            key = (
                str(episode.get("type")),
                int(episode.get("peak_index", 0)),
            )
            existing = unique.get(key)
            if existing is None or cls._number(episode.get("confidence")) > cls._number(
                existing.get("confidence")
            ):
                unique[key] = episode

        episodes = sorted(
            unique.values(),
            key=lambda x: (
                int(x.get("start_index", 0)),
                int(x.get("peak_index", 0)),
                str(x.get("type")),
            ),
        )

        # Consolidate nearby same-type peaks so a risk oscillation is represented
        # as one episode rather than several cards.
        consolidated: list[dict[str, Any]] = []
        for episode in episodes:
            matching = [
                existing
                for existing in consolidated
                if existing.get("type") == episode.get("type")
                and abs(
                    int(existing.get("peak_index", 0))
                    - int(episode.get("peak_index", 0))
                ) <= 4
            ]
            if not matching:
                consolidated.append(episode)
                continue
            existing = matching[-1]
            if cls._number(episode.get("peak_risk")) > cls._number(
                existing.get("peak_risk")
            ):
                consolidated[consolidated.index(existing)] = episode
            else:
                existing["end_index"] = max(
                    int(existing.get("end_index", 0)),
                    int(episode.get("end_index", 0)),
                )
                existing["duration_seconds"] = max(
                    int(existing.get("duration_seconds", 0)),
                    int(episode.get("duration_seconds", 0)),
                )

        episodes = sorted(
            consolidated,
            key=lambda x: (
                int(x.get("start_index", 0)),
                int(x.get("peak_index", 0)),
                str(x.get("type")),
            ),
        )

        # Keep the replay useful on long sessions: retain at most five of each
        # category, preferring the strongest peak/confidence combinations.
        by_category: dict[str, list[dict[str, Any]]] = {}
        for episode in episodes:
            by_category.setdefault(str(episode.get("type")), []).append(episode)

        selected: list[dict[str, Any]] = []
        for category_episodes in by_category.values():
            strongest = sorted(
                category_episodes,
                key=lambda item: (
                    cls._number(item.get("peak_risk")) * 0.70
                    + cls._number(item.get("confidence")) * 0.30
                ),
                reverse=True,
            )[:5]
            selected.extend(strongest)

        episodes = sorted(
            selected,
            key=lambda x: (
                int(x.get("start_index", 0)),
                int(x.get("peak_index", 0)),
                str(x.get("type")),
            ),
        )

        cls._attach_visual_evidence(episodes, visual_evidence)

        counts = Counter(str(item.get("type")) for item in episodes)
        repeated_types = {
            key: count for key, count in counts.items()
            if count >= 2
        }
        for episode in episodes:
            count = counts[str(episode.get("type"))]
            episode["repeated_in_session"] = count >= 2
            episode["same_type_episode_count"] = count

        peak_episode = max(
            episodes,
            key=lambda x: cls._number(x.get("peak_risk")),
            default=None,
        )

        if not episodes:
            summary = (
                "No Near-Miss pattern met the current historical-analysis "
                "criteria in this Decision Memory session."
            )
        elif peak_episode:
            summary = (
                f"{len(episodes)} Near-Miss episode(s) detected. "
                f"Highest analysed episode: {peak_episode.get('title')} at "
                f"{cls._number(peak_episode.get('peak_risk')):.0%} advisory risk."
            )
        else:
            summary = f"{len(episodes)} Near-Miss episode(s) detected."

        return {
            "version": cls.VERSION,
            "episode_count": len(episodes),
            "by_type": dict(counts),
            "repeated_types": repeated_types,
            "highest_peak_risk": round(
                max(
                    (cls._number(x.get("peak_risk")) for x in episodes),
                    default=0.0,
                ),
                4,
            ),
            "summary": summary,
            "episodes": episodes,
            "safety_boundary": (
                "Near-Miss Memory is retrospective analysis of Decision Memory. "
                "It does not change Monitoring, the trained model or alert thresholds."
            ),
        }
