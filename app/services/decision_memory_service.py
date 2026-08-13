from __future__ import annotations

import csv
import io
import json
import math
import threading
import uuid
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from app.services.near_miss_memory_service import NearMissMemoryService


class DecisionMemoryService:
    """Persistent research trace for Guardian's advisory Intelligence layer.

    V8.2.3 is owned by the application session lifecycle rather than a browser
    page. Monitoring starts/finalises Decision Memory automatically, while a
    lightweight background advisory sampler records evidence. Camera ownership,
    V3 calibration, TemporalStateEngine and AlertManager remain untouched.
    """

    SAMPLE_INTERVAL_SECONDS = 2.0
    FLUSH_INTERVAL_SECONDS = 6.0

    CSV_FIELDS = [
        "timestamp", "elapsed_seconds", "driver_profile", "state", "alert_count",
        "ear", "baseline_ear", "yawn_score", "head_tilt",
        "raw_model_probability", "existing_decision_probability",
        "existing_smoothed_probability", "advisory_risk", "risk_band",
        "decision_confidence", "confidence_level", "signal_quality", "image_quality",
        "perception_confidence", "perception_state", "perception_observation_mode",
        "perception_can_trust_presence", "perception_can_trust_absence",
        "perception_reason_codes",
        "weather", "road_condition", "external_light", "occlusion",
        "raw_automatic_occlusion", "raw_automatic_occlusion_confidence",
        "automatic_occlusion", "automatic_occlusion_confidence",
        "occlusion_temporal_window",
        "eye_visibility_score", "eye_region_brightness_ratio",
        "eye_dark_ratio", "eye_edge_density",
        "manual_occlusion", "resolved_occlusion", "occlusion_source",
        "context_caution", "dominant_evidence", "recommended_action",
    ]

    def __init__(self, root: Path) -> None:
        self.root = root
        self.memory_dir = root / "guardian_data" / "decision_memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active_id: str | None = None
        self._active: dict[str, Any] | None = None
        self._last_monitoring = False
        self._last_sample_clock = 0.0
        self._last_flush_clock = 0.0
        self._pending_visual_events: list[dict[str, Any]] = []

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value if value is not None else default)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    def _new_session(self, metrics: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        session_id = f"decision_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return {
            "schema_version": "8.6-decision-memory-v6",
            "id": session_id,
            "started_at": now.isoformat(timespec="seconds"),
            "ended_at": None,
            "driver_profile": str(metrics.get("driver_profile_name") or "Guest"),
            "label": "",
            "condition": "",
            "notes": "",
            "source": "Guardian automatic advisory sampler",
            "coverage_note": (
                "Samples are captured automatically during Monitoring at the advisory "
                "sampling interval. This is an explainability trace, not every camera frame."
            ),
            "sample_interval_seconds_nominal": self.SAMPLE_INTERVAL_SECONDS,
            "samples": [],
            "summary": {},
        }

    def _sample(self, metrics: dict[str, Any], intelligence: dict[str, Any]) -> dict[str, Any]:
        engine = intelligence.get("decision_engine", {}) or {}
        confidence = engine.get("confidence", {}) or {}
        caution = engine.get("context_caution", {}) or {}
        context = intelligence.get("context", {}) or {}
        quality = intelligence.get("signal_quality", {}) or {}
        environment = intelligence.get("environment", {}) or {}
        perception = intelligence.get("perception_confidence", {}) or {}
        evidence = engine.get("evidence", []) or []
        strongest = max(evidence, key=lambda item: self._number(item.get("contribution")), default={})
        legacy = engine.get("legacy_reference", {}) or {}

        return {
            "timestamp": self._now().isoformat(timespec="milliseconds"),
            "elapsed_seconds": int(self._number(metrics.get("session_seconds"))),
            "driver_profile": str(metrics.get("driver_profile_name") or "Guest"),
            "state": str(metrics.get("state") or "READY"),
            "alert_count": int(self._number(metrics.get("alert_count"))),
            "ear": round(self._number(metrics.get("ear")), 6),
            "baseline_ear": round(self._number(metrics.get("baseline_ear")), 6),
            "yawn_score": round(self._number(metrics.get("yawn_score")), 6),
            "head_tilt": round(self._number(metrics.get("head_tilt")), 6),
            "raw_model_probability": round(self._number(legacy.get("raw_model_probability", metrics.get("raw_probability"))), 6),
            "existing_decision_probability": round(self._number(legacy.get("existing_personalized_probability", metrics.get("decision_probability"))), 6),
            "existing_smoothed_probability": round(self._number(legacy.get("existing_smoothed_probability", metrics.get("fatigue_probability"))), 6),
            "advisory_risk": round(self._number(engine.get("risk_score")), 6),
            "risk_band": str(engine.get("risk_band") or "standby"),
            "decision_confidence": round(self._number(confidence.get("score")), 6),
            "confidence_level": str(confidence.get("level") or "standby"),
            "signal_quality": round(self._number(quality.get("score")), 6),
            "image_quality": round(self._number(environment.get("quality_score")), 6),
            "perception_confidence": round(self._number(perception.get("score")), 6),
            "perception_state": str(perception.get("state") or "standby"),
            "perception_observation_mode": str(perception.get("observation_mode") or "standby"),
            "perception_can_trust_presence": bool(perception.get("can_trust_presence")),
            "perception_can_trust_absence": bool(perception.get("can_trust_absence")),
            "perception_reason_codes": ";".join(
                str(item.get("code") or "")
                for item in perception.get("reason_codes", []) or []
                if item.get("code")
            ),
            "weather": str(context.get("weather") or "unknown"),
            "road_condition": str(context.get("road_condition") or "unknown"),
            "external_light": str(context.get("external_light") or "unknown"),
            "occlusion": str(context.get("occlusion") or "none"),
            "raw_automatic_occlusion": str(
                (context.get("automatic_occlusion") or {}).get("raw_value")
                or metrics.get("raw_automatic_occlusion")
                or "unknown"
            ),
            "raw_automatic_occlusion_confidence": round(
                self._number(
                    (context.get("automatic_occlusion") or {}).get("raw_confidence")
                    or metrics.get("raw_automatic_occlusion_confidence")
                ),
                6,
            ),
            "automatic_occlusion": str(
                (context.get("automatic_occlusion") or {}).get("value")
                or metrics.get("automatic_occlusion")
                or "unknown"
            ),
            "automatic_occlusion_confidence": round(
                self._number(
                    (context.get("automatic_occlusion") or {}).get("confidence")
                    or metrics.get("automatic_occlusion_confidence")
                ),
                6,
            ),
            "occlusion_temporal_window": int(
                self._number(
                    (context.get("automatic_occlusion") or {}).get("temporal_window")
                    or metrics.get("occlusion_temporal_window")
                )
            ),
            "eye_visibility_score": round(
                self._number(
                    (context.get("automatic_occlusion") or {}).get("eye_visibility_score")
                    or metrics.get("eye_visibility_score")
                ),
                6,
            ),
            "eye_region_brightness_ratio": round(
                self._number(metrics.get("eye_region_brightness_ratio")),
                6,
            ),
            "eye_dark_ratio": round(
                self._number(metrics.get("eye_dark_ratio")),
                6,
            ),
            "eye_edge_density": round(
                self._number(metrics.get("eye_edge_density")),
                6,
            ),
            "manual_occlusion": str(context.get("manual_occlusion") or "none"),
            "resolved_occlusion": str(
                context.get("resolved_occlusion")
                or context.get("occlusion")
                or "none"
            ),
            "occlusion_source": str(
                context.get("occlusion_source")
                or ((context.get("sources") or {}).get("occlusion") or {}).get("source")
                or "unknown"
            ),
            "context_caution": str(caution.get("level") or "normal"),
            "dominant_evidence": str(strongest.get("label") or "none"),
            "recommended_action": str(engine.get("action") or ""),
            "evidence": evidence,
        }

    @staticmethod
    def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {
                "sample_count": 0, "observed_seconds": 0,
                "maximum_advisory_risk": 0.0, "average_advisory_risk": 0.0,
                "average_confidence": 0.0, "minimum_confidence": 0.0,
                "average_ear": 0.0, "alert_count": 0,
                "dominant_evidence": "none", "high_risk_samples": 0,
            }
        risks=[float(x.get("advisory_risk",0) or 0) for x in samples]
        conf=[float(x.get("decision_confidence",0) or 0) for x in samples]
        ears=[float(x.get("ear",0) or 0) for x in samples if x.get("ear")]
        counts={}
        for x in samples:
            label=str(x.get("dominant_evidence") or "none"); counts[label]=counts.get(label,0)+1
        start=int(samples[0].get("elapsed_seconds",0) or 0); end=int(samples[-1].get("elapsed_seconds",0) or 0)
        return {
            "sample_count":len(samples), "observed_seconds":max(0,end-start),
            "maximum_advisory_risk":round(max(risks),6),
            "average_advisory_risk":round(sum(risks)/len(risks),6),
            "average_confidence":round(sum(conf)/len(conf),6),
            "minimum_confidence":round(min(conf),6),
            "average_ear":round(sum(ears)/len(ears),6) if ears else 0.0,
            "alert_count":max(int(x.get("alert_count",0) or 0) for x in samples),
            "dominant_evidence":max(counts,key=counts.get),
            "high_risk_samples":sum(1 for x in samples if str(x.get("risk_band"))=="high"),
        }

    @staticmethod
    def _evidence_value(sample: dict[str, Any], key: str) -> float:
        for item in sample.get("evidence", []) or []:
            if item.get("key") == key:
                try: return float(item.get("value",0) or 0)
                except (TypeError,ValueError): return 0.0
        return 0.0

    @classmethod
    def events(cls, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events=[]; previous_band=None; previous_alerts=0; yawn_active=False; eye_active=False; low_conf=False; previous_perception=None
        for index,sample in enumerate(samples):
            band=str(sample.get("risk_band") or "standby")
            if previous_band is not None and band != previous_band:
                events.append({"index":index,"timestamp":sample.get("timestamp"),"type":"risk_change","level":band,"title":f"Risk changed to {band}","detail":sample.get("recommended_action","")})
            previous_band=band
            alerts=int(sample.get("alert_count",0) or 0)
            if alerts > previous_alerts:
                events.append({"index":index,"timestamp":sample.get("timestamp"),"type":"alert","level":"high","title":"Monitoring alert recorded","detail":f"Alert count increased to {alerts}."})
            previous_alerts=alerts
            yawn=cls._evidence_value(sample,"yawn")
            if yawn >= .65 and not yawn_active:
                events.append({"index":index,"timestamp":sample.get("timestamp"),"type":"yawn","level":"watch","title":"Strong yawn evidence","detail":f"Yawn evidence reached {yawn:.0%}."})
                yawn_active=True
            elif yawn < .35: yawn_active=False
            eye=cls._evidence_value(sample,"personal_baseline")
            if eye >= .55 and not eye_active:
                events.append({"index":index,"timestamp":sample.get("timestamp"),"type":"eye","level":"watch","title":"EAR moved below personal baseline","detail":f"Personal deviation evidence reached {eye:.0%}."})
                eye_active=True
            elif eye < .25: eye_active=False
            conf=float(sample.get("decision_confidence",0) or 0)
            if conf < .58 and not low_conf:
                events.append({"index":index,"timestamp":sample.get("timestamp"),"type":"confidence","level":"warning","title":"Decision confidence reduced","detail":f"Confidence fell to {conf:.0%}."})
                low_conf=True
            elif conf >= .68: low_conf=False
            perception=str(sample.get("perception_state") or "").lower()
            if perception in {"trusted","degraded","insufficient"}:
                if previous_perception is not None and perception != previous_perception:
                    events.append({
                        "index":index,
                        "timestamp":sample.get("timestamp"),
                        "type":"perception",
                        "level":perception,
                        "title":f"Perception changed to {perception}",
                        "detail":str(sample.get("perception_reason_codes") or "No perception limitation recorded."),
                    })
                previous_perception=perception
        return events

    def begin(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Start a Decision Memory session from the Monitoring lifecycle."""
        with self._lock:
            if self._active is not None:
                return self.status()

            self._active = self._new_session(metrics)
            self._active_id = self._active["id"]
            self._last_monitoring = True
            self._last_sample_clock = 0.0
            self._last_flush_clock = 0.0
            self._pending_visual_events.clear()

            # Persist the header immediately so the new timestamp appears even
            # before the first advisory sample.
            self._active["summary"] = self._summary([])
            self._save(self._active)
            self._last_flush_clock = monotonic()
            return self.status()

    def record(
        self,
        metrics: dict[str, Any],
        intelligence: dict[str, Any],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Append an advisory snapshot when a Monitoring session is active."""
        now_clock = monotonic()
        with self._lock:
            if self._active is None:
                if not bool(metrics.get("monitoring")):
                    return self.status()
                self.begin(metrics)

            if not bool(metrics.get("monitoring")) and not force:
                return self.status()

            should_sample = (
                force
                or (now_clock - self._last_sample_clock) >= self.SAMPLE_INTERVAL_SECONDS
            )
            if not should_sample:
                return self.status()

            previous_event_count = len(self.events(self._active["samples"]))
            self._active["samples"].append(self._sample(metrics, intelligence))
            self._active["summary"] = self._summary(self._active["samples"])
            self._last_sample_clock = now_clock

            for event in self.events(self._active["samples"])[previous_event_count:]:
                if event.get("type") in {"alert", "yawn", "eye", "risk_change"}:
                    self._pending_visual_events.append(dict(event))

            should_flush = (
                force
                or (now_clock - self._last_flush_clock) >= self.FLUSH_INTERVAL_SECONDS
                or len(self._active["samples"]) <= 1
            )
            if should_flush:
                self._save(self._active)
                self._last_flush_clock = now_clock

            return self.status()

    def finalise(self) -> dict[str, Any]:
        """Finalise the active session even if no browser page is open."""
        with self._lock:
            if self._active is None:
                self._last_monitoring = False
                return self.status()

            self._active["ended_at"] = self._now().isoformat(timespec="seconds")
            self._active["summary"] = self._summary(self._active["samples"])
            self._save(self._active)

            completed_id = self._active_id
            self._active = None
            self._active_id = None
            self._last_monitoring = False
            self._last_sample_clock = 0.0
            self._last_flush_clock = 0.0
            self._pending_visual_events.clear()

            return {
                **self.status(),
                "completed_session_id": completed_id,
            }

    def pop_visual_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._pending_visual_events)
            self._pending_visual_events.clear()
            return events

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "recording": self._active is not None,
                "active_session_id": self._active_id,
                "sample_count": (
                    len(self._active.get("samples", []))
                    if self._active is not None
                    else 0
                ),
                "coverage_note": (
                    "Decision Memory starts and stops automatically with Monitoring."
                ),
                "sample_interval_seconds": self.SAMPLE_INTERVAL_SECONDS,
            }

    def observe(
        self,
        metrics: dict[str, Any],
        intelligence: dict[str, Any],
    ) -> dict[str, Any]:
        """Backward-compatible wrapper for older callers."""
        monitoring = bool(metrics.get("monitoring"))
        if monitoring and self._active is None:
            self.begin(metrics)
        if monitoring:
            return self.record(metrics, intelligence)
        if self._active is not None:
            return self.finalise()
        return self.status()

    def _save(self,payload:dict[str,Any])->None:
        path=self.memory_dir/f"{payload['id']}.json"; temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8'); temporary.replace(path)

    def _read(self,session_id:str)->dict[str,Any]:
        path=self.resolve(session_id,'.json'); payload=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload,dict): raise ValueError('Invalid Decision Memory file.')
        return payload

    def list_sessions(self)->list[dict[str,Any]]:
        rows=[]
        for path in sorted(self.memory_dir.glob('decision_*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            try: payload=json.loads(path.read_text(encoding='utf-8'))
            except (OSError,json.JSONDecodeError): continue
            summary=payload.get('summary',{}) or {}
            rows.append({
                'id':payload.get('id',path.stem),'started_at':payload.get('started_at'),'ended_at':payload.get('ended_at'),
                'driver_profile':payload.get('driver_profile','Guest'),'label':payload.get('label',''),'condition':payload.get('condition',''),'notes':payload.get('notes',''),
                'sample_count':int(summary.get('sample_count',0) or 0),'observed_seconds':int(summary.get('observed_seconds',0) or 0),
                'maximum_advisory_risk':float(summary.get('maximum_advisory_risk',0) or 0),'average_advisory_risk':float(summary.get('average_advisory_risk',0) or 0),
                'average_confidence':float(summary.get('average_confidence',0) or 0),'minimum_confidence':float(summary.get('minimum_confidence',0) or 0),
                'average_ear':float(summary.get('average_ear',0) or 0),'alert_count':int(summary.get('alert_count',0) or 0),
                'dominant_evidence':summary.get('dominant_evidence','none'),'event_count':len(self.events(payload.get('samples',[]) or [])),'visual_evidence_count':self.visual_evidence(str(payload.get('id',path.stem))).get('file_count',0),'near_miss_count':NearMissMemoryService.analyse(payload.get('samples',[]) or []).get('episode_count',0),
                'active':payload.get('id')==self._active_id,
            })
        return rows

    @property
    def visual_evidence_root(self) -> Path:
        return self.root / "guardian_data" / "visual_evidence"

    def visual_evidence(self, session_id: str) -> dict[str, Any]:
        if not session_id.startswith("decision_"):
            raise ValueError("Invalid Decision Memory ID.")
        manifest_path = self.visual_evidence_root / session_id / "manifest.json"
        if not manifest_path.exists():
            return {"available": False, "session_id": session_id, "events": [], "file_count": 0}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        events = payload.get("events", []) if isinstance(payload, dict) else []
        count = sum(len(item.get("files", []) or []) for item in events)
        return {
            "available": bool(count),
            "session_id": session_id,
            "events": events,
            "file_count": count,
            "privacy_note": payload.get("privacy_note", "") if isinstance(payload, dict) else "",
        }

    def resolve_visual_evidence(self, session_id: str, filename: str) -> Path:
        if not session_id.startswith("decision_"):
            raise ValueError("Invalid Decision Memory ID.")
        if Path(filename).name != filename or not filename.lower().endswith(".jpg"):
            raise ValueError("Invalid visual evidence filename.")
        directory = (self.visual_evidence_root / session_id).resolve()
        candidate = (directory / filename).resolve()
        if directory not in candidate.parents or not candidate.exists():
            raise FileNotFoundError(filename)
        return candidate

    def delete_visual_evidence(self, session_id: str) -> dict[str, Any]:
        if not session_id.startswith("decision_"):
            raise ValueError("Invalid Decision Memory ID.")
        directory = self.visual_evidence_root / session_id
        if directory.exists():
            import shutil
            shutil.rmtree(directory)
        return {"deleted": True, "session_id": session_id, "decision_memory_preserved": True}

    def get_session(self,session_id:str)->dict[str,Any]:
        payload=self._read(session_id)
        samples=payload.get('samples',[]) or []
        payload['events']=self.events(samples)
        payload['visual_evidence']=self.visual_evidence(session_id)
        payload['near_miss_memory']=NearMissMemoryService.analyse(
            samples,
            visual_evidence=payload['visual_evidence'],
        )
        return payload

    def update_metadata(self,session_id:str,*,label:str='',condition:str='',notes:str='')->dict[str,Any]:
        with self._lock:
            payload=self._read(session_id); payload['label']=str(label or '').strip()[:120]; payload['condition']=str(condition or '').strip()[:80]; payload['notes']=str(notes or '').strip()[:1000]; self._save(payload); return self.get_session(session_id)

    def comparison(self,first_id:str,second_id:str)->dict[str,Any]:
        first=self.get_session(first_id); second=self.get_session(second_id); a=first.get('summary',{}) or {}; b=second.get('summary',{}) or {}
        def delta(k): return round(float(b.get(k,0) or 0)-float(a.get(k,0) or 0),6)
        return {
            'first':{'id':first['id'],'started_at':first.get('started_at'),'driver_profile':first.get('driver_profile'),'label':first.get('label',''),'condition':first.get('condition',''),'summary':a,'event_count':len(first.get('events',[]))},
            'second':{'id':second['id'],'started_at':second.get('started_at'),'driver_profile':second.get('driver_profile'),'label':second.get('label',''),'condition':second.get('condition',''),'summary':b,'event_count':len(second.get('events',[]))},
            'delta_second_minus_first':{'average_advisory_risk':delta('average_advisory_risk'),'maximum_advisory_risk':delta('maximum_advisory_risk'),'average_confidence':delta('average_confidence'),'minimum_confidence':delta('minimum_confidence'),'average_ear':delta('average_ear'),'alert_count':int(b.get('alert_count',0) or 0)-int(a.get('alert_count',0) or 0),'event_count':len(second.get('events',[]))-len(first.get('events',[]))},
            'series':{'first':self._compact_series(first.get('samples',[])),'second':self._compact_series(second.get('samples',[]))},
        }

    @staticmethod
    def _compact_series(samples:list[dict[str,Any]])->list[dict[str,Any]]:
        if not samples: return []
        step=max(1,len(samples)//180)
        return [{'t':int(x.get('elapsed_seconds',0) or 0),'risk':float(x.get('advisory_risk',0) or 0),'confidence':float(x.get('decision_confidence',0) or 0),'ear':float(x.get('ear',0) or 0)} for x in samples[::step]]

    def aggregate(self)->dict[str,Any]:
        sessions=self.list_sessions()
        if not sessions: return {'session_count':0,'observed_seconds':0,'average_risk':0.0,'peak_risk':0.0,'average_confidence':0.0,'alerts':0,'events':0}
        weights=[max(1,s['sample_count']) for s in sessions]; total=sum(weights)
        return {'session_count':len(sessions),'observed_seconds':sum(s['observed_seconds'] for s in sessions),'average_risk':round(sum(s['average_advisory_risk']*w for s,w in zip(sessions,weights))/total,6),'peak_risk':max(s['maximum_advisory_risk'] for s in sessions),'average_confidence':round(sum(s['average_confidence']*w for s,w in zip(sessions,weights))/total,6),'alerts':sum(s['alert_count'] for s in sessions),'events':sum(s['event_count'] for s in sessions)}

    def resolve(self,session_id:str,suffix:str)->Path:
        if not session_id.startswith('decision_'): raise ValueError('Invalid Decision Memory ID.')
        candidate=(self.memory_dir/f'{session_id}{suffix}').resolve()
        if self.memory_dir.resolve() not in candidate.parents: raise ValueError('Invalid Decision Memory path.')
        if suffix=='.json':
            if not candidate.exists(): raise FileNotFoundError(session_id)
            return candidate
        raise ValueError('Unsupported Decision Memory suffix.')

    def csv_bytes(self,session_id:str)->bytes:
        payload=self._read(session_id); output=io.StringIO(); writer=csv.DictWriter(output,fieldnames=self.CSV_FIELDS); writer.writeheader()
        for sample in payload.get('samples',[]): writer.writerow({k:sample.get(k,'') for k in self.CSV_FIELDS})
        return output.getvalue().encode('utf-8-sig')
