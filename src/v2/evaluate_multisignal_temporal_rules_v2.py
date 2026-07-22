"""DriverGuardianAI V2 - Experiment 24: Multisignal Temporal Rules."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "v2" / "ablation" / "driver_guardian_core_behaviour.joblib"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "multisignal_temporal_rules"

LIVE_LOGS = [
    {
        "path": PROJECT_ROOT / "logs" / "v2" / "core_behaviour" / "driver_guardian_v2_session_20260718_174331.csv",
        "actual_label": "Alert",
        "actual_class": 0,
        "session_name": "live_alert",
    },
    {
        "path": PROJECT_ROOT / "logs" / "v2" / "core_behaviour" / "driver_guardian_v2_session_20260718_174536.csv",
        "actual_label": "Fatigue",
        "actual_class": 1,
        "session_name": "live_fatigue",
    },
]

FEATURE_COLUMNS = ["ear", "yawn_score", "head_tilt"]
MODEL_THRESHOLD = 0.64
WINDOW_SIZE = 12
MINIMUM_HISTORY = 5
INFERENCE_INTERVAL_SECONDS = 0.50

CURRENT_WARNING_RATIO = 0.50
CURRENT_CRITICAL_RATIO = 0.70
CURRENT_CONSECUTIVE_FATIGUE = 3

LOW_EAR_THRESHOLD = 0.245
STRONG_LOW_EAR_THRESHOLD = 0.220
YAWN_SUPPORT_THRESHOLD = 0.10
HEAD_TILT_SUPPORT_THRESHOLD = 7.0
LOW_EAR_RATIO_REQUIRED = 0.60
MODEL_FATIGUE_RATIO_REQUIRED = 0.60
STRONG_LOW_EAR_CONSECUTIVE = 4
SUPPORTED_LOW_EAR_CONSECUTIVE = 3


@dataclass
class RuleDecision:
    state: str
    alert_level: str
    trigger_alert: bool
    reason: str
    model_fatigue_ratio: float
    low_ear_ratio: float
    strong_low_ear_ratio: float
    consecutive_model_fatigue: int
    consecutive_low_ear: int
    consecutive_strong_low_ear: int
    yawn_support_present: bool
    head_tilt_support_present: bool
    support_present: bool
    history_size: int


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")


def load_model_bundle() -> Dict[str, Any]:
    require_file(MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    for key in ("pipeline", "fatigue_threshold", "feature_columns"):
        if key not in bundle:
            raise KeyError(f"Model bundle is missing key: {key}")
    return bundle


def prepare_live_log(spec: Dict[str, Any]) -> pd.DataFrame:
    path = Path(spec["path"])
    require_file(path)
    df = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    df = df.copy()
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    df["session_name"] = spec["session_name"]
    df["actual_label"] = spec["actual_label"]
    df["actual_class"] = spec["actual_class"]
    df["source_log"] = str(path)
    df["source_row"] = np.arange(len(df), dtype=int)
    return df


def load_live_logs() -> pd.DataFrame:
    return pd.concat([prepare_live_log(spec) for spec in LIVE_LOGS], ignore_index=True)


class CurrentTemporalRule:
    def __init__(self) -> None:
        self.history: Deque[int] = deque(maxlen=WINDOW_SIZE)
        self.consecutive_fatigue = 0
        self.alert_active = False

    def update(self, fatigue_probability: float) -> RuleDecision:
        fatigue = int(fatigue_probability >= MODEL_THRESHOLD)
        self.history.append(fatigue)
        self.consecutive_fatigue = self.consecutive_fatigue + 1 if fatigue else 0
        history_size = len(self.history)
        ratio = float(np.mean(self.history))

        base = dict(
            model_fatigue_ratio=ratio,
            low_ear_ratio=0.0,
            strong_low_ear_ratio=0.0,
            consecutive_model_fatigue=self.consecutive_fatigue,
            consecutive_low_ear=0,
            consecutive_strong_low_ear=0,
            yawn_support_present=False,
            head_tilt_support_present=False,
            support_present=False,
            history_size=history_size,
        )

        if history_size < MINIMUM_HISTORY:
            return RuleDecision("Monitoring", "none", False, "Collecting temporal history.", **base)

        critical = (
            self.consecutive_fatigue >= CURRENT_CONSECUTIVE_FATIGUE
            or ratio >= CURRENT_CRITICAL_RATIO
        )
        if critical:
            trigger = not self.alert_active
            self.alert_active = True
            return RuleDecision(
                "Fatigue", "critical", trigger,
                "Existing model-probability rule reached the critical condition.",
                **base,
            )

        self.alert_active = False
        if ratio >= CURRENT_WARNING_RATIO:
            return RuleDecision(
                "Possible Fatigue", "warning", False,
                "Existing rule observed a majority of Fatigue predictions.",
                **base,
            )

        return RuleDecision("Alert", "none", False, "Recent predictions are predominantly Alert.", **base)


class MultisignalTemporalRule:
    def __init__(self) -> None:
        self.history: Deque[Dict[str, Any]] = deque(maxlen=WINDOW_SIZE)
        self.consecutive_model_fatigue = 0
        self.consecutive_low_ear = 0
        self.consecutive_strong_low_ear = 0
        self.alert_active = False

    def update(self, fatigue_probability: float, ear: float, yawn_score: float, head_tilt: float) -> RuleDecision:
        model_fatigue = fatigue_probability >= MODEL_THRESHOLD
        low_ear = ear <= LOW_EAR_THRESHOLD
        strong_low_ear = ear <= STRONG_LOW_EAR_THRESHOLD
        yawn_support = yawn_score >= YAWN_SUPPORT_THRESHOLD
        head_support = head_tilt >= HEAD_TILT_SUPPORT_THRESHOLD

        self.history.append({
            "model_fatigue": model_fatigue,
            "low_ear": low_ear,
            "strong_low_ear": strong_low_ear,
            "yawn_support": yawn_support,
            "head_support": head_support,
        })

        self.consecutive_model_fatigue = self.consecutive_model_fatigue + 1 if model_fatigue else 0
        self.consecutive_low_ear = self.consecutive_low_ear + 1 if low_ear else 0
        self.consecutive_strong_low_ear = self.consecutive_strong_low_ear + 1 if strong_low_ear else 0

        history_size = len(self.history)
        model_ratio = float(np.mean([row["model_fatigue"] for row in self.history]))
        low_ratio = float(np.mean([row["low_ear"] for row in self.history]))
        strong_ratio = float(np.mean([row["strong_low_ear"] for row in self.history]))
        yawn_present = any(row["yawn_support"] for row in self.history)
        head_present = any(row["head_support"] for row in self.history)
        support_present = yawn_present or head_present

        base = dict(
            model_fatigue_ratio=model_ratio,
            low_ear_ratio=low_ratio,
            strong_low_ear_ratio=strong_ratio,
            consecutive_model_fatigue=self.consecutive_model_fatigue,
            consecutive_low_ear=self.consecutive_low_ear,
            consecutive_strong_low_ear=self.consecutive_strong_low_ear,
            yawn_support_present=yawn_present,
            head_tilt_support_present=head_present,
            support_present=support_present,
            history_size=history_size,
        )

        if history_size < MINIMUM_HISTORY:
            return RuleDecision("Monitoring", "none", False, "Collecting multisignal history.", **base)

        critical_by_strong_eye = self.consecutive_strong_low_ear >= STRONG_LOW_EAR_CONSECUTIVE
        critical_by_supported_eye = (
            self.consecutive_low_ear >= SUPPORTED_LOW_EAR_CONSECUTIVE
            and support_present
        )
        critical_by_window = (
            model_ratio >= MODEL_FATIGUE_RATIO_REQUIRED
            and low_ratio >= LOW_EAR_RATIO_REQUIRED
            and support_present
        )

        if critical_by_strong_eye or critical_by_supported_eye or critical_by_window:
            trigger = not self.alert_active
            self.alert_active = True
            if critical_by_strong_eye:
                reason = "Sustained strongly low EAR detected."
            elif critical_by_supported_eye:
                reason = "Sustained low EAR with supporting yawn or head-tilt evidence."
            else:
                reason = "High model-Fatigue and low-EAR ratios with supporting evidence."
            return RuleDecision("Fatigue", "critical", trigger, reason, **base)

        self.alert_active = False
        warning = (
            model_ratio >= CURRENT_WARNING_RATIO
            or low_ratio >= LOW_EAR_RATIO_REQUIRED
            or self.consecutive_low_ear >= SUPPORTED_LOW_EAR_CONSECUTIVE
        )
        if warning:
            return RuleDecision(
                "Possible Fatigue", "warning", False,
                "Fatigue evidence is present, but critical multisignal evidence is incomplete.",
                **base,
            )

        return RuleDecision("Alert", "none", False, "No sustained multisignal Fatigue evidence.", **base)


def replay_session(session: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    current = CurrentTemporalRule()
    multisignal = MultisignalTemporalRule()
    rows: List[Dict[str, Any]] = []

    for index, ((_, row), probability) in enumerate(zip(session.iterrows(), probabilities)):
        current_decision = current.update(float(probability))
        multi_decision = multisignal.update(
            float(probability), float(row["ear"]), float(row["yawn_score"]), float(row["head_tilt"])
        )
        output: Dict[str, Any] = {
            "session_name": row["session_name"],
            "actual_label": row["actual_label"],
            "actual_class": int(row["actual_class"]),
            "source_log": row["source_log"],
            "source_row": int(row["source_row"]),
            "elapsed_seconds": index * INFERENCE_INTERVAL_SECONDS,
            "ear": float(row["ear"]),
            "yawn_score": float(row["yawn_score"]),
            "head_tilt": float(row["head_tilt"]),
            "fatigue_probability": float(probability),
            "model_prediction": "Fatigue" if probability >= MODEL_THRESHOLD else "Alert",
        }
        for prefix, decision in (("current", current_decision), ("multisignal", multi_decision)):
            for field, value in decision.__dict__.items():
                output[f"{prefix}_{field}"] = value
        rows.append(output)

    return pd.DataFrame(rows)


def summarise_rule(group: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    critical = group[f"{prefix}_alert_level"] == "critical"
    warning = group[f"{prefix}_alert_level"] == "warning"
    triggers = group[f"{prefix}_trigger_alert"].astype(bool)
    fatigue_state = group[f"{prefix}_state"] == "Fatigue"
    actual_label = str(group["actual_label"].iloc[0])
    correct_rate = float(fatigue_state.mean() if actual_label == "Fatigue" else (~fatigue_state).mean())
    first_critical: Optional[float] = None
    if critical.any():
        first_critical = float(group.loc[critical, "elapsed_seconds"].iloc[0])

    return {
        "rule": prefix,
        "session_name": str(group["session_name"].iloc[0]),
        "actual_label": actual_label,
        "samples": int(len(group)),
        "correct_state_rate": correct_rate,
        "warning_rows": int(warning.sum()),
        "warning_rate": float(warning.mean()),
        "critical_rows": int(critical.sum()),
        "critical_rate": float(critical.mean()),
        "alert_triggers": int(triggers.sum()),
        "first_critical_alert_seconds": first_critical,
        "model_fatigue_frame_rate": float((group["model_prediction"] == "Fatigue").mean()),
        "mean_fatigue_probability": float(group["fatigue_probability"].mean()),
    }


def create_session_summary(row_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in row_level.groupby("session_name", sort=True):
        rows.append(summarise_rule(group, "current"))
        rows.append(summarise_rule(group, "multisignal"))
    return pd.DataFrame(rows)


def create_comparison_summary(session_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for actual_label in ("Alert", "Fatigue"):
        for rule in ("current", "multisignal"):
            subset = session_summary[
                (session_summary["actual_label"] == actual_label)
                & (session_summary["rule"] == rule)
            ]
            if subset.empty:
                continue
            weights = subset["samples"]
            rows.append({
                "actual_label": actual_label,
                "rule": rule,
                "samples": int(weights.sum()),
                "mean_correct_state_rate": float(np.average(subset["correct_state_rate"], weights=weights)),
                "mean_warning_rate": float(np.average(subset["warning_rate"], weights=weights)),
                "mean_critical_rate": float(np.average(subset["critical_rate"], weights=weights)),
                "total_alert_triggers": int(subset["alert_triggers"].sum()),
            })
    return pd.DataFrame(rows)


def save_plot(comparison: pd.DataFrame, value_column: str, path: Path, title: str, ylabel: str) -> None:
    pivot = comparison.pivot(index="actual_label", columns="rule", values=value_column).fillna(0.0)
    positions = np.arange(len(pivot.index))
    width = 0.36
    plt.figure(figsize=(9, 6))
    plt.bar(positions - width / 2, pivot.get("current", 0.0), width=width, label="Current rule")
    plt.bar(positions + width / 2, pivot.get("multisignal", 0.0), width=width, label="Multisignal rule")
    plt.xticks(positions, pivot.index)
    plt.ylim(0.0, 1.05)
    plt.ylabel(ylabel)
    plt.xlabel("Actual session label")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Experiment 24: Multisignal Temporal Rule Evaluation")
    print("=" * 72)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    live = load_live_logs()

    print(f"Saved model threshold: {float(bundle['fatigue_threshold']):.2f}")
    print(f"Rule threshold: {MODEL_THRESHOLD:.2f}")
    print("\nLoaded sessions:")
    print(live[["session_name", "actual_label"]].value_counts().to_string())

    replayed = []
    for _, session in live.groupby("session_name", sort=True):
        session = session.reset_index(drop=True)
        probabilities = pipeline.predict_proba(session[FEATURE_COLUMNS])[:, 1]
        replayed.append(replay_session(session, probabilities))

    row_level = pd.concat(replayed, ignore_index=True)
    session_summary = create_session_summary(row_level)
    comparison = create_comparison_summary(session_summary)

    row_level.to_csv(RESULTS_DIRECTORY / "row_level_decisions.csv", index=False)
    session_summary.to_csv(RESULTS_DIRECTORY / "session_summary.csv", index=False)
    comparison.to_csv(RESULTS_DIRECTORY / "comparison_summary.csv", index=False)

    save_plot(
        comparison, "mean_critical_rate",
        RESULTS_DIRECTORY / "critical_rate_comparison.png",
        "Current vs Multisignal Critical Rate",
        "Critical decision rate",
    )
    save_plot(
        comparison, "mean_correct_state_rate",
        RESULTS_DIRECTORY / "fatigue_detection_comparison.png",
        "Current vs Multisignal Temporal Accuracy",
        "Correct temporal-state rate",
    )

    findings = []
    for session_name in session_summary["session_name"].unique():
        subset = session_summary[session_summary["session_name"] == session_name]
        current = subset[subset["rule"] == "current"].iloc[0]
        multi = subset[subset["rule"] == "multisignal"].iloc[0]
        findings.extend([
            f"{session_name}: critical rate changed from {current['critical_rate']:.1%} to {multi['critical_rate']:.1%}.",
            f"{session_name}: correct temporal-state rate changed from {current['correct_state_rate']:.1%} to {multi['correct_state_rate']:.1%}.",
            f"{session_name}: alert triggers changed from {int(current['alert_triggers'])} to {int(multi['alert_triggers'])}.",
        ])

    summary = {
        "project": "DriverGuardianAI V2",
        "experiment": "multisignal_temporal_rule_evaluation",
        "model_path": str(MODEL_PATH),
        "parameters": {
            "model_threshold": MODEL_THRESHOLD,
            "window_size": WINDOW_SIZE,
            "minimum_history": MINIMUM_HISTORY,
            "low_ear_threshold": LOW_EAR_THRESHOLD,
            "strong_low_ear_threshold": STRONG_LOW_EAR_THRESHOLD,
            "yawn_support_threshold": YAWN_SUPPORT_THRESHOLD,
            "head_tilt_support_threshold": HEAD_TILT_SUPPORT_THRESHOLD,
            "low_ear_ratio_required": LOW_EAR_RATIO_REQUIRED,
            "model_fatigue_ratio_required": MODEL_FATIGUE_RATIO_REQUIRED,
            "strong_low_ear_consecutive": STRONG_LOW_EAR_CONSECUTIVE,
            "supported_low_ear_consecutive": SUPPORTED_LOW_EAR_CONSECUTIVE,
        },
        "findings": findings,
        "important_note": "Offline replay of two controlled sessions; not evidence of safety-critical readiness.",
    }
    (RESULTS_DIRECTORY / "experiment_summary.json").write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("\nSession summary:")
    print(session_summary.to_string(index=False))
    print("\nComparison summary:")
    print(comparison.to_string(index=False))
    print("\nMain findings:")
    for finding in findings:
        print(f"- {finding}")
    print("\nResults saved to:")
    print(RESULTS_DIRECTORY)


if __name__ == "__main__":
    main()