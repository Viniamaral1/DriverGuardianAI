"""DriverGuardianAI V2 — Experiment 26: causal temporal features."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data/splits/v2/train.csv"
CALIBRATION_PATH = PROJECT_ROOT / "data/splits/v2/calibration.csv"
TEST_PATH = PROJECT_ROOT / "data/splits/v2/test.csv"
MODEL_PATH = PROJECT_ROOT / "models/v2/driver_guardian_temporal_hgb_v2.joblib"
RESULTS_DIR = PROJECT_ROOT / "results/v2/temporal_features"

LIVE_LOGS = [
    {
        "path": PROJECT_ROOT / "logs/v2/core_behaviour/driver_guardian_v2_session_20260718_174331.csv",
        "actual_label": "Alert",
        "session_name": "live_alert",
    },
    {
        "path": PROJECT_ROOT / "logs/v2/core_behaviour/driver_guardian_v2_session_20260718_174536.csv",
        "actual_label": "Fatigue",
        "session_name": "live_fatigue",
    },
]

TARGET_COLUMN = "fatigue_level"
TARGET_MAP = {
    "Alert": 0,
    "Mild Fatigue": 1,
    "Moderate Fatigue": 1,
    "Severe Fatigue": 1,
    "Fatigue": 1,
}
CLASS_NAMES = ["Alert", "Fatigue"]
SESSION_CANDIDATES = [
    "session_id", "session", "recording_id", "recording", "video_id",
    "video", "source_file", "filename", "file_name",
]
ORDER_CANDIDATES = [
    "timestamp", "elapsed_seconds", "time_seconds", "frame_index",
    "frame_number", "frame", "sample_index", "row_index",
]
PARTICIPANT_CANDIDATES = ["participant_id", "participant", "subject_id", "subject"]
WINDOWS = [5, 12, 30]
LOW_EAR = 0.245
STRONG_LOW_EAR = 0.220
YAWN_ACTIVE = 0.10
TILT_ACTIVE = 7.0
RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.10, 0.951, 0.01), 2)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def first_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    return next((name for name in names if name in df.columns), None)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def bool_numeric(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0}
    )
    return pd.to_numeric(series, errors="coerce").fillna(mapped).astype(float)


def prepare(df: pd.DataFrame, name: str, require_target: bool = True) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    for col in ["ear", "yawn_score", "head_tilt"]:
        if col not in df.columns:
            raise ValueError(f"{name} is missing {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["low_light", "face_confidence"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = bool_numeric(df[col])

    if "condition" not in df.columns:
        df["condition"] = "unknown"
    df["condition"] = df["condition"].fillna("unknown").astype(str).str.strip().str.lower()

    session_col = first_column(df, SESSION_CANDIDATES)
    if session_col is None:
        participant_col = first_column(df, PARTICIPANT_CANDIDATES)
        if participant_col:
            df["_session_id"] = df[participant_col].astype(str) + "__" + df["condition"]
            print(f"WARNING: {name} has no explicit session column; using {participant_col}+condition.")
        else:
            df["_session_id"] = f"{name}_single_sequence"
            print(f"WARNING: {name} has no session/participant column; treating it as one sequence.")
        session_col = "_session_id"
    else:
        df[session_col] = df[session_col].fillna("unknown_session").astype(str)

    order_col = first_column(df, ORDER_CANDIDATES)
    if order_col is None:
        df["_order"] = df.groupby(session_col, sort=False).cumcount()
    else:
        order = pd.to_numeric(df[order_col], errors="coerce")
        fallback = df.groupby(session_col, sort=False).cumcount()
        df["_order"] = order.fillna(fallback)

    df["_original_order"] = np.arange(len(df))
    df = df.sort_values([session_col, "_order", "_original_order"], kind="stable").reset_index(drop=True)

    if require_target:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"{name} is missing {TARGET_COLUMN}")
        labels = df[TARGET_COLUMN].astype(str).str.strip()
        unknown = sorted(set(labels.unique()) - set(TARGET_MAP))
        if unknown:
            raise ValueError(f"Unsupported labels in {name}: {unknown}")
        df["binary_target"] = labels.map(TARGET_MAP).astype(int)

    return df, session_col


def roll(grouped, window: int, method: str) -> pd.Series:
    obj = grouped.rolling(window=window, min_periods=1)
    result = getattr(obj, method)() if method != "std" else obj.std(ddof=0)
    return result.reset_index(level=0, drop=True)


def engineer(df: pd.DataFrame, session_col: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    grouped = out.groupby(session_col, sort=False, group_keys=False)

    for feature in ["ear", "yawn_score", "head_tilt"]:
        delta = grouped[feature].diff().fillna(0.0)
        out[f"{feature}_delta"] = delta
        out[f"{feature}_abs_delta"] = delta.abs()
        out[f"{feature}_acceleration"] = delta.groupby(out[session_col], sort=False).diff().fillna(0.0)

    out["low_ear_flag"] = (out["ear"] <= LOW_EAR).astype(float)
    out["strong_low_ear_flag"] = (out["ear"] <= STRONG_LOW_EAR).astype(float)
    out["yawn_activity_flag"] = (out["yawn_score"] >= YAWN_ACTIVE).astype(float)
    out["head_tilt_activity_flag"] = (out["head_tilt"] >= TILT_ACTIVE).astype(float)

    for window in WINDOWS:
        for feature in ["ear", "yawn_score", "head_tilt"]:
            grouped_feature = out.groupby(session_col, sort=False)[feature]
            for method in ["mean", "std", "min", "max"]:
                out[f"{feature}_{method}_w{window}"] = roll(grouped_feature, window, method)

        for feature in ["ear_abs_delta", "yawn_score_abs_delta", "head_tilt_abs_delta"]:
            grouped_feature = out.groupby(session_col, sort=False)[feature]
            out[f"{feature}_mean_w{window}"] = roll(grouped_feature, window, "mean")
            out[f"{feature}_max_w{window}"] = roll(grouped_feature, window, "max")

        for feature in ["low_ear_flag", "strong_low_ear_flag", "yawn_activity_flag", "head_tilt_activity_flag"]:
            grouped_feature = out.groupby(session_col, sort=False)[feature]
            out[f"{feature}_ratio_w{window}"] = roll(grouped_feature, window, "mean")

    out["session_row_index"] = out.groupby(session_col, sort=False).cumcount().astype(float)

    base = ["ear", "yawn_score", "head_tilt", "low_light", "face_confidence"]
    prefixes = (
        "ear_", "yawn_score_", "head_tilt_", "low_ear_", "strong_low_ear_",
        "yawn_activity_", "head_tilt_activity_",
    )
    engineered = [c for c in out.columns if c.startswith(prefixes) or c == "session_row_index"]
    numeric = list(dict.fromkeys([c for c in base + engineered if c in out.columns]))
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out, numeric


def one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(numeric: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", one_hot()),
                ]),
                ["condition"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    result = {
        "threshold": float(threshold),
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_fatigue": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_fatigue": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(1.0 - specificity),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probability))
        result["average_precision"] = float(average_precision_score(y, probability))
    return result


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLDS:
        row = metrics(y, probability, float(threshold))
        row["selection_score"] = row["balanced_accuracy"] + 0.02 * row["specificity"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["selection_score", "balanced_accuracy", "specificity", "recall_sensitivity"],
        ascending=False,
    ).reset_index(drop=True)
    return float(table.iloc[0]["threshold"]), table


def permutation_importance(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
    threshold: float,
    repeats: int = 3,
) -> pd.DataFrame:
    base_pred = (pipeline.predict_proba(x)[:, 1] >= threshold).astype(int)
    baseline = balanced_accuracy_score(y, base_pred)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for feature in x.columns:
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            pred = (pipeline.predict_proba(shuffled)[:, 1] >= threshold).astype(int)
            drops.append(baseline - balanced_accuracy_score(y, pred))
        rows.append({
            "feature": feature,
            "mean_balanced_accuracy_drop": float(np.mean(drops)),
            "std_balanced_accuracy_drop": float(np.std(drops)),
            "baseline_balanced_accuracy": float(baseline),
        })
    return pd.DataFrame(rows).sort_values("mean_balanced_accuracy_drop", ascending=False).reset_index(drop=True)


def confusion_plot(y: np.ndarray, pred: np.ndarray) -> None:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Temporal Feature HGB — Untouched Test")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_confusion_matrix.png", dpi=300)
    plt.close()


def importance_plot(table: pd.DataFrame) -> None:
    top = table.head(20).sort_values("mean_balanced_accuracy_drop")
    plt.figure(figsize=(11, 8))
    plt.barh(top["feature"], top["mean_balanced_accuracy_drop"])
    plt.xlabel("Balanced-accuracy reduction after permutation")
    plt.title("Temporal Feature Model — Permutation Importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=300)
    plt.close()


def optional_live() -> Optional[pd.DataFrame]:
    if not all(item["path"].exists() for item in LIVE_LOGS):
        return None
    frames = []
    for item in LIVE_LOGS:
        frame = pd.read_csv(item["path"])
        frame[TARGET_COLUMN] = item["actual_label"]
        frame["session_id"] = item["session_name"]
        frame["session_name"] = item["session_name"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Experiment 26: Temporal Feature Engineering")
    print("=" * 72)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    for path in [TRAIN_PATH, CALIBRATION_PATH, TEST_PATH]:
        require(path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_cal = pd.read_csv(CALIBRATION_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    print(f"Train rows: {len(raw_train)}")
    print(f"Calibration rows: {len(raw_cal)}")
    print(f"Test rows: {len(raw_test)}")

    train, train_session = prepare(raw_train, "training split")
    calibration, cal_session = prepare(raw_cal, "calibration split")
    test, test_session = prepare(raw_test, "test split")

    train, numeric = engineer(train, train_session)
    calibration, numeric_cal = engineer(calibration, cal_session)
    test, numeric_test = engineer(test, test_session)
    if numeric != numeric_cal or numeric != numeric_test:
        raise RuntimeError("Engineered feature orders do not match across splits.")

    selected = numeric + ["condition"]
    write_json(RESULTS_DIR / "engineered_feature_columns.json", {
        "numeric_features": numeric,
        "categorical_features": ["condition"],
        "selected_features": selected,
        "windows": WINDOWS,
        "causal_only": True,
        "hands_detected_excluded": True,
    })
    print(f"Model input columns: {len(selected)}")

    x_train, y_train = train[selected], train["binary_target"].to_numpy(int)
    x_cal, y_cal = calibration[selected], calibration["binary_target"].to_numpy(int)
    x_test, y_test = test[selected], test["binary_target"].to_numpy(int)

    pipeline = build_pipeline(numeric)
    print("Training Histogram Gradient Boosting...")
    pipeline.fit(x_train, y_train)

    cal_prob = pipeline.predict_proba(x_cal)[:, 1]
    threshold, threshold_table = choose_threshold(y_cal, cal_prob)
    threshold_table.to_csv(RESULTS_DIR / "threshold_search.csv", index=False)
    cal_metrics = metrics(y_cal, cal_prob, threshold)
    write_json(RESULTS_DIR / "calibration_metrics.json", cal_metrics)
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Calibration balanced accuracy: {cal_metrics['balanced_accuracy']:.3f}")

    test_prob = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = metrics(y_test, test_prob, threshold)
    write_json(RESULTS_DIR / "test_metrics.json", test_metrics)

    test_output = test.copy()
    test_output["fatigue_probability"] = test_prob
    test_output["predicted_class"] = test_pred
    test_output["predicted_label"] = np.where(test_pred == 1, "Fatigue", "Alert")
    test_output["correct_prediction"] = test_pred == y_test
    test_output.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    print("\nUntouched Test Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred, labels=[0, 1]))
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=CLASS_NAMES, zero_division=0))
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")
    confusion_plot(y_test, test_pred)

    print("\nCalculating permutation importance...")
    importance = permutation_importance(pipeline, x_test, y_test, threshold)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    importance_plot(importance)
    print("\nTop engineered features:")
    print(importance.head(20).to_string(index=False))

    live_metrics: Optional[dict[str, Any]] = None
    live_raw = optional_live()
    if live_raw is not None:
        live, live_session = prepare(live_raw, "development live logs")
        live, live_numeric = engineer(live, live_session)
        if live_numeric != numeric:
            raise RuntimeError("Live engineered feature order does not match training.")
        y_live = live["binary_target"].to_numpy(int)
        live_prob = pipeline.predict_proba(live[selected])[:, 1]
        live_pred = (live_prob >= threshold).astype(int)
        live_metrics = metrics(y_live, live_prob, threshold)
        session_rows = []
        live_output = live.copy()
        live_output["fatigue_probability"] = live_prob
        live_output["predicted_class"] = live_pred
        live_output["predicted_label"] = np.where(live_pred == 1, "Fatigue", "Alert")
        live_output["correct_prediction"] = live_pred == y_live
        for name, group in live_output.groupby("session_name", sort=True):
            group_metrics = metrics(
                group["binary_target"].to_numpy(int),
                group["fatigue_probability"].to_numpy(float),
                threshold,
            )
            group_metrics["session_name"] = str(name)
            session_rows.append(group_metrics)
        live_metrics["session_metrics"] = session_rows
        write_json(RESULTS_DIR / "live_metrics.json", live_metrics)
        live_output.to_csv(RESULTS_DIR / "live_predictions.csv", index=False)
        print("\nDevelopment live metrics:")
        print(pd.DataFrame(session_rows).to_string(index=False))

    bundle = {
        "pipeline": pipeline,
        "model_name": "hist_gradient_boosting_temporal_v2",
        "feature_columns": selected,
        "numeric_features": numeric,
        "categorical_features": ["condition"],
        "class_names": CLASS_NAMES,
        "fatigue_threshold": threshold,
        "windows": WINDOWS,
        "feature_thresholds": {
            "low_ear": LOW_EAR,
            "strong_low_ear": STRONG_LOW_EAR,
            "yawn_active": YAWN_ACTIVE,
            "head_tilt_active": TILT_ACTIVE,
        },
        "test_metrics": test_metrics,
        "live_metrics": live_metrics,
        "causal_temporal_features": True,
    }
    joblib.dump(bundle, MODEL_PATH)

    write_json(RESULTS_DIR / "experiment_summary.json", {
        "experiment": "temporal_feature_engineering_v2",
        "model_path": str(MODEL_PATH),
        "selected_threshold": threshold,
        "feature_count": len(selected),
        "windows": WINDOWS,
        "test_metrics": test_metrics,
        "live_metrics": live_metrics,
        "top_features": importance.head(20).to_dict(orient="records"),
        "notes": [
            "All rolling features are causal and reset at session boundaries.",
            "hands_detected is excluded because it behaved as a shortcut.",
            "Existing live logs are development data, not untouched final validation.",
            "This remains a research prototype, not a safety-certified system.",
        ],
    })

    print("\n" + "=" * 72)
    print("Experiment 26 completed successfully.")
    print("=" * 72)
    print(f"\nModel saved to:\n{MODEL_PATH}")
    print(f"\nResults saved to:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()