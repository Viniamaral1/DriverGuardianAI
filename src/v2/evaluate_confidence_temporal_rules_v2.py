"""Experiment 25: confidence-aware temporal rule evaluation."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/v2/ablation/driver_guardian_core_behaviour.joblib"
OUT = ROOT / "results/v2/confidence_temporal_rules"

LIVE_LOGS = [
    {
        "path": ROOT / "logs/v2/core_behaviour/driver_guardian_v2_session_20260718_174331.csv",
        "actual_label": "Alert",
        "actual_class": 0,
        "session_name": "live_alert",
    },
    {
        "path": ROOT / "logs/v2/core_behaviour/driver_guardian_v2_session_20260718_174536.csv",
        "actual_label": "Fatigue",
        "actual_class": 1,
        "session_name": "live_fatigue",
    },
]

FEATURES = ["ear", "yawn_score", "head_tilt"]
MODEL_THRESHOLD = 0.64
WINDOW = 12
MIN_HISTORY = 5
INFERENCE_SECONDS = 0.5

DEFAULT = {
    "very_high_probability": 0.95,
    "high_probability": 0.85,
    "warning_probability": 0.70,
    "release_probability": 0.45,
    "very_high_consecutive_required": 6,
    "high_consecutive_required": 8,
    "rolling_average_critical": 0.85,
    "rolling_average_warning": 0.65,
    "high_probability_ratio_critical": 0.70,
    "warning_probability_ratio": 0.55,
    "release_consecutive_required": 4,
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_bundle() -> Dict[str, Any]:
    require(MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    needed = {"pipeline", "fatigue_threshold", "feature_columns"}
    missing = needed.difference(bundle)
    if missing:
        raise KeyError(f"Model bundle missing: {sorted(missing)}")
    if list(bundle["feature_columns"]) != FEATURES:
        raise ValueError(f"Expected features {FEATURES}, got {bundle['feature_columns']}")
    return bundle


def load_logs() -> pd.DataFrame:
    frames = []
    for spec in LIVE_LOGS:
        require(spec["path"])
        df = pd.read_csv(spec["path"])
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"{spec['path'].name} missing {missing}")
        for c in FEATURES:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
        df["session_name"] = spec["session_name"]
        df["actual_label"] = spec["actual_label"]
        df["actual_class"] = spec["actual_class"]
        df["source_log"] = str(spec["path"])
        df["source_row"] = np.arange(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class CurrentRule:
    def __init__(self) -> None:
        self.history = deque(maxlen=WINDOW)
        self.consecutive = 0
        self.active = False

    def update(self, p: float) -> Dict[str, Any]:
        fatigue = p >= MODEL_THRESHOLD
        self.history.append(int(fatigue))
        self.consecutive = self.consecutive + 1 if fatigue else 0
        ratio = float(np.mean(self.history))
        if len(self.history) < MIN_HISTORY:
            return self._result("Monitoring", "none", False, "Collecting history", ratio)
        critical = self.consecutive >= 3 or ratio >= 0.70
        if critical:
            trigger = not self.active
            self.active = True
            return self._result("Fatigue", "critical", trigger, "Current critical condition", ratio)
        self.active = False
        if ratio >= 0.50:
            return self._result("Possible Fatigue", "warning", False, "Current warning ratio", ratio)
        return self._result("Alert", "none", False, "Current rule Alert", ratio)

    def _result(self, state: str, level: str, trigger: bool, reason: str, ratio: float) -> Dict[str, Any]:
        return {"state": state, "alert_level": level, "trigger_alert": trigger, "reason": reason,
                "mean_probability": np.nan, "model_fatigue_ratio": ratio,
                "high_probability_ratio": 0.0, "very_high_probability_ratio": 0.0,
                "consecutive_model_fatigue": self.consecutive,
                "consecutive_high_probability": 0, "consecutive_very_high_probability": 0,
                "consecutive_release": 0, "history_size": len(self.history)}


class MultisignalRule:
    def __init__(self) -> None:
        self.history = deque(maxlen=WINDOW)
        self.low = self.strong = self.model = 0
        self.active = False

    def update(self, p: float, ear: float, yawn: float, tilt: float) -> Dict[str, Any]:
        model = p >= MODEL_THRESHOLD
        low = ear <= 0.245
        strong = ear <= 0.220
        support = yawn >= 0.10 or tilt >= 7.0
        self.history.append({"p": p, "model": model, "low": low, "strong": strong, "support": support})
        self.model = self.model + 1 if model else 0
        self.low = self.low + 1 if low else 0
        self.strong = self.strong + 1 if strong else 0
        model_ratio = float(np.mean([x["model"] for x in self.history]))
        low_ratio = float(np.mean([x["low"] for x in self.history]))
        mean_p = float(np.mean([x["p"] for x in self.history]))
        support_present = any(x["support"] for x in self.history)
        if len(self.history) < MIN_HISTORY:
            return self._result("Monitoring", "none", False, "Collecting multisignal history", mean_p, model_ratio)
        critical = self.strong >= 4 or (self.low >= 3 and support_present) or (model_ratio >= 0.60 and low_ratio >= 0.60 and support_present)
        if critical:
            trigger = not self.active
            self.active = True
            return self._result("Fatigue", "critical", trigger, "Multisignal critical condition", mean_p, model_ratio)
        self.active = False
        warning = model_ratio >= 0.50 or low_ratio >= 0.60
        if warning:
            return self._result("Possible Fatigue", "warning", False, "Multisignal warning", mean_p, model_ratio)
        return self._result("Alert", "none", False, "No multisignal fatigue evidence", mean_p, model_ratio)

    def _result(self, state: str, level: str, trigger: bool, reason: str, mean_p: float, ratio: float) -> Dict[str, Any]:
        return {"state": state, "alert_level": level, "trigger_alert": trigger, "reason": reason,
                "mean_probability": mean_p, "model_fatigue_ratio": ratio,
                "high_probability_ratio": 0.0, "very_high_probability_ratio": 0.0,
                "consecutive_model_fatigue": self.model,
                "consecutive_high_probability": 0, "consecutive_very_high_probability": 0,
                "consecutive_release": 0, "history_size": len(self.history)}


class ConfidenceRule:
    def __init__(self, cfg: Dict[str, float]) -> None:
        self.cfg = dict(cfg)
        self.history = deque(maxlen=WINDOW)
        self.model = self.high = self.very_high = self.release = 0
        self.active = False

    def update(self, p: float) -> Dict[str, Any]:
        c = self.cfg
        self.history.append(float(p))
        self.model = self.model + 1 if p >= MODEL_THRESHOLD else 0
        self.high = self.high + 1 if p >= c["high_probability"] else 0
        self.very_high = self.very_high + 1 if p >= c["very_high_probability"] else 0
        self.release = self.release + 1 if p <= c["release_probability"] else 0
        arr = np.asarray(self.history, dtype=float)
        mean_p = float(arr.mean())
        model_ratio = float(np.mean(arr >= MODEL_THRESHOLD))
        high_ratio = float(np.mean(arr >= c["high_probability"]))
        very_high_ratio = float(np.mean(arr >= c["very_high_probability"]))
        if len(arr) < MIN_HISTORY:
            return self._result("Monitoring", "none", False, "Collecting confidence history", mean_p, model_ratio, high_ratio, very_high_ratio)
        if self.active and self.release < c["release_consecutive_required"]:
            return self._result("Fatigue", "critical", False, "Critical state retained by hysteresis", mean_p, model_ratio, high_ratio, very_high_ratio)
        if self.active and self.release >= c["release_consecutive_required"]:
            self.active = False
        by_very_high = self.very_high >= c["very_high_consecutive_required"]
        by_high = self.high >= c["high_consecutive_required"]
        by_window = mean_p >= c["rolling_average_critical"] and high_ratio >= c["high_probability_ratio_critical"]
        if by_very_high or by_high or by_window:
            trigger = not self.active
            self.active = True
            reason = "Sustained very-high confidence" if by_very_high else "Sustained high confidence" if by_high else "High rolling confidence"
            return self._result("Fatigue", "critical", trigger, reason, mean_p, model_ratio, high_ratio, very_high_ratio)
        warning_ratio = float(np.mean(arr >= c["warning_probability"]))
        warning = mean_p >= c["rolling_average_warning"] or warning_ratio >= c["warning_probability_ratio"] or self.model >= 3
        if warning:
            return self._result("Possible Fatigue", "warning", False, "Elevated confidence without critical persistence", mean_p, model_ratio, high_ratio, very_high_ratio)
        return self._result("Alert", "none", False, "Confidence below warning thresholds", mean_p, model_ratio, high_ratio, very_high_ratio)

    def _result(self, state: str, level: str, trigger: bool, reason: str, mean_p: float,
                model_ratio: float, high_ratio: float, very_high_ratio: float) -> Dict[str, Any]:
        return {"state": state, "alert_level": level, "trigger_alert": trigger, "reason": reason,
                "mean_probability": mean_p, "model_fatigue_ratio": model_ratio,
                "high_probability_ratio": high_ratio, "very_high_probability_ratio": very_high_ratio,
                "consecutive_model_fatigue": self.model,
                "consecutive_high_probability": self.high,
                "consecutive_very_high_probability": self.very_high,
                "consecutive_release": self.release, "history_size": len(self.history)}


def add_decision(out: Dict[str, Any], prefix: str, d: Dict[str, Any]) -> None:
    for key, value in d.items():
        out[f"{prefix}_{key}"] = value


def replay(session: pd.DataFrame, probs: np.ndarray, cfg: Dict[str, float]) -> pd.DataFrame:
    current, multi, conf = CurrentRule(), MultisignalRule(), ConfidenceRule(cfg)
    rows = []
    for i, ((_, row), p) in enumerate(zip(session.iterrows(), probs)):
        p = float(p)
        out = {
            "session_name": row["session_name"], "actual_label": row["actual_label"],
            "actual_class": int(row["actual_class"]), "source_log": row["source_log"],
            "source_row": int(row["source_row"]), "elapsed_seconds": i * INFERENCE_SECONDS,
            "ear": float(row["ear"]), "yawn_score": float(row["yawn_score"]),
            "head_tilt": float(row["head_tilt"]), "fatigue_probability": p,
            "model_prediction": "Fatigue" if p >= MODEL_THRESHOLD else "Alert",
        }
        add_decision(out, "current", current.update(p))
        add_decision(out, "multisignal", multi.update(p, out["ear"], out["yawn_score"], out["head_tilt"]))
        add_decision(out, "confidence", conf.update(p))
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(group: pd.DataFrame, rule: str) -> Dict[str, Any]:
    fatigue = group[f"{rule}_state"].eq("Fatigue")
    critical = group[f"{rule}_alert_level"].eq("critical")
    warning = group[f"{rule}_alert_level"].eq("warning")
    triggers = group[f"{rule}_trigger_alert"].astype(bool)
    label = str(group["actual_label"].iloc[0])
    correct = fatigue.mean() if label == "Fatigue" else (~fatigue).mean()
    first = float(group.loc[critical, "elapsed_seconds"].iloc[0]) if critical.any() else None
    return {
        "rule": rule, "session_name": group["session_name"].iloc[0], "actual_label": label,
        "samples": len(group), "correct_state_rate": float(correct),
        "warning_rows": int(warning.sum()), "warning_rate": float(warning.mean()),
        "critical_rows": int(critical.sum()), "critical_rate": float(critical.mean()),
        "alert_triggers": int(triggers.sum()), "first_critical_alert_seconds": first,
        "model_fatigue_frame_rate": float(group["model_prediction"].eq("Fatigue").mean()),
        "mean_fatigue_probability": float(group["fatigue_probability"].mean()),
    }


def session_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for _, group in rows.groupby("session_name", sort=True):
        for rule in ["current", "multisignal", "confidence"]:
            output.append(summarize(group, rule))
    return pd.DataFrame(output)


def comparison(summary: pd.DataFrame) -> pd.DataFrame:
    output = []
    for label in ["Alert", "Fatigue"]:
        for rule in ["current", "multisignal", "confidence"]:
            s = summary[(summary.actual_label == label) & (summary.rule == rule)]
            if s.empty:
                continue
            output.append({
                "actual_label": label, "rule": rule, "samples": int(s.samples.sum()),
                "mean_correct_state_rate": float(np.average(s.correct_state_rate, weights=s.samples)),
                "mean_warning_rate": float(np.average(s.warning_rate, weights=s.samples)),
                "mean_critical_rate": float(np.average(s.critical_rate, weights=s.samples)),
                "total_alert_triggers": int(s.alert_triggers.sum()),
            })
    return pd.DataFrame(output)


def score_config(live: pd.DataFrame, pipeline, cfg: Dict[str, float]) -> Dict[str, Any]:
    metrics = {}
    for _, session in live.groupby("session_name", sort=True):
        probs = pipeline.predict_proba(session[FEATURES])[:, 1]
        rule = ConfidenceRule(cfg)
        states, criticals = [], []
        for p in probs:
            d = rule.update(float(p))
            states.append(d["state"] == "Fatigue")
            criticals.append(d["alert_level"] == "critical")
        label = session.actual_label.iloc[0]
        if label == "Alert":
            metrics["alert_correct_state_rate"] = float(np.mean(~np.asarray(states)))
            metrics["alert_false_critical_rate"] = float(np.mean(criticals))
        else:
            metrics["fatigue_detection_rate"] = float(np.mean(states))
    bal = (metrics["alert_correct_state_rate"] + metrics["fatigue_detection_rate"]) / 2
    selection = bal - 1.25 * metrics["alert_false_critical_rate"]
    return {**cfg, **metrics, "balanced_temporal_accuracy": bal, "selection_score": selection}


def grid_search(live: pd.DataFrame, pipeline) -> tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    for vh in [0.90, 0.95]:
        for n in [5, 6, 8]:
            for avg in [0.80, 0.85, 0.90]:
                for ratio in [0.60, 0.70, 0.80]:
                    cfg = dict(DEFAULT)
                    cfg.update(very_high_probability=vh, very_high_consecutive_required=n,
                               rolling_average_critical=avg, high_probability_ratio_critical=ratio)
                    rows.append(score_config(live, pipeline, cfg))
    table = pd.DataFrame(rows).sort_values(
        ["selection_score", "balanced_temporal_accuracy", "fatigue_detection_rate"],
        ascending=False,
    ).reset_index(drop=True)
    best = {k: table.iloc[0][k].item() if hasattr(table.iloc[0][k], "item") else table.iloc[0][k] for k in DEFAULT}
    for k in ["very_high_consecutive_required", "high_consecutive_required", "release_consecutive_required"]:
        best[k] = int(best[k])
    return best, table


def plot_grouped(comp: pd.DataFrame, value: str, ylabel: str, title: str, path: Path) -> None:
    pivot = comp.pivot(index="actual_label", columns="rule", values=value).fillna(0)
    labels = list(pivot.index)
    rules = [r for r in ["current", "multisignal", "confidence"] if r in pivot]
    x = np.arange(len(labels)); width = 0.8 / len(rules)
    plt.figure(figsize=(10, 6))
    for i, rule in enumerate(rules):
        offset = (i - (len(rules) - 1) / 2) * width
        plt.bar(x + offset, pivot[rule].to_numpy(), width=width, label=rule)
    plt.xticks(x, labels); plt.ylim(0, 1.05); plt.xlabel("Actual session label")
    plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Experiment 25: Confidence-Aware Temporal Rules")
    print("=" * 72)
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(); pipeline = bundle["pipeline"]
    live = load_logs()
    print("\nLoaded sessions:")
    print(live[["session_name", "actual_label"]].value_counts().to_string())
    print("\nSearching confidence-rule settings...")
    best, grid = grid_search(live, pipeline)
    grid.to_csv(OUT / "threshold_grid_search.csv", index=False)
    print("\nSelected confidence configuration:")
    for k, v in best.items(): print(f"{k}: {v}")
    replayed = []
    for _, session in live.groupby("session_name", sort=True):
        session = session.reset_index(drop=True)
        probs = pipeline.predict_proba(session[FEATURES])[:, 1]
        replayed.append(replay(session, probs, best))
    rows = pd.concat(replayed, ignore_index=True)
    sess = session_summary(rows); comp = comparison(sess)
    rows.to_csv(OUT / "row_level_decisions.csv", index=False)
    sess.to_csv(OUT / "session_summary.csv", index=False)
    comp.to_csv(OUT / "comparison_summary.csv", index=False)
    plot_grouped(comp, "mean_correct_state_rate", "Correct temporal-state rate",
                 "Current vs Multisignal vs Confidence-Aware Temporal Accuracy",
                 OUT / "temporal_accuracy_comparison.png")
    plot_grouped(comp, "mean_critical_rate", "Critical decision rate",
                 "Current vs Multisignal vs Confidence-Aware Critical Rate",
                 OUT / "critical_rate_comparison.png")
    alert = comp[(comp.actual_label == "Alert") & (comp.rule == "confidence")].iloc[0]
    fatigue = comp[(comp.actual_label == "Fatigue") & (comp.rule == "confidence")].iloc[0]
    findings = [
        f"Confidence-aware Alert correct-state rate: {alert.mean_correct_state_rate:.1%}.",
        f"Confidence-aware Alert critical rate: {alert.mean_critical_rate:.1%}.",
        f"Confidence-aware Fatigue detection rate: {fatigue.mean_correct_state_rate:.1%}.",
        f"Confidence-aware Fatigue critical rate: {fatigue.mean_critical_rate:.1%}.",
    ]
    with (OUT / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "project": "DriverGuardianAI V2",
            "experiment": "confidence_aware_temporal_rules",
            "model_path": str(MODEL_PATH),
            "selected_confidence_configuration": best,
            "findings": findings,
            "important_notes": [
                "The same two controlled sessions were used for rule selection and evaluation.",
                "Results may therefore be optimistic.",
                "Collect a new untouched Alert/Fatigue pair before real-time integration.",
                "This remains a research prototype, not safety-critical evidence.",
            ],
        }, f, indent=4)
    print("\nTop grid-search configurations:")
    print(grid.head(10).to_string(index=False))
    print("\nSession summary:")
    print(sess.to_string(index=False))
    print("\nComparison summary:")
    print(comp.to_string(index=False))
    print("\nMain findings:")
    for finding in findings: print(f"- {finding}")
    print("\nResults saved to:")
    print(OUT)


if __name__ == "__main__":
    main()