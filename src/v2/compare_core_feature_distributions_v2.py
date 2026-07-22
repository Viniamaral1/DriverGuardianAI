"""
DriverGuardianAI V2 — Core Feature Distribution Comparison.

Compares EAR, yawn_score, and head_tilt across:
- Training Alert
- Training Fatigue
- Live Alert
- Live Fatigue

Outputs statistics, pairwise effect sizes, distribution overlap,
Jensen-Shannon distance, range checks, and publication-ready plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "v2" / "train.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "v2" / "core_feature_distributions"

LIVE_LOGS = [
    {
        "path": PROJECT_ROOT / "logs" / "v2" / "driver_guardian_v2_session_20260717_154546.csv",
        "actual_label": "Alert",
        "group_name": "Live Alert",
    },
    {
        "path": PROJECT_ROOT / "logs" / "v2" / "driver_guardian_v2_session_20260717_154904.csv",
        "actual_label": "Fatigue",
        "group_name": "Live Fatigue",
    },
]

TARGET_COLUMN = "fatigue_level"
TARGET_MAPPING = {
    "Alert": "Alert",
    "Mild Fatigue": "Fatigue",
    "Moderate Fatigue": "Fatigue",
}
CORE_FEATURES = ["ear", "yawn_score", "head_tilt"]
GROUP_ORDER = ["Training Alert", "Training Fatigue", "Live Alert", "Live Fatigue"]
PAIRWISE_TESTS = [
    ("Training Alert", "Training Fatigue", "training_class_separation"),
    ("Live Alert", "Live Fatigue", "live_class_separation"),
    ("Training Alert", "Live Alert", "alert_domain_shift"),
    ("Training Fatigue", "Live Fatigue", "fatigue_domain_shift"),
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")


def prepare_numeric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [f for f in CORE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing features: {missing}")
    df = df.copy()
    for feature in CORE_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna(subset=CORE_FEATURES)
    if df.empty:
        raise ValueError(f"{name} has no valid core-feature rows.")
    return df


def load_data() -> pd.DataFrame:
    require_file(TRAIN_PATH)
    train = prepare_numeric(pd.read_csv(TRAIN_PATH), "training split")
    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"Training data is missing {TARGET_COLUMN}.")
    unknown = set(train[TARGET_COLUMN].astype(str).unique()) - set(TARGET_MAPPING)
    if unknown:
        raise ValueError(f"Unsupported training labels: {sorted(unknown)}")
    train["binary_label"] = train[TARGET_COLUMN].astype(str).map(TARGET_MAPPING)
    train["group_name"] = train["binary_label"].map(
        {"Alert": "Training Alert", "Fatigue": "Training Fatigue"}
    )

    live_frames = []
    for spec in LIVE_LOGS:
        path = Path(spec["path"])
        require_file(path)
        live = prepare_numeric(pd.read_csv(path), path.name)
        live["binary_label"] = spec["actual_label"]
        live["group_name"] = spec["group_name"]
        live_frames.append(live)

    combined = pd.concat(
        [
            train[[*CORE_FEATURES, "binary_label", "group_name"]],
            *[x[[*CORE_FEATURES, "binary_label", "group_name"]] for x in live_frames],
        ],
        ignore_index=True,
    )
    combined["group_name"] = pd.Categorical(
        combined["group_name"], categories=GROUP_ORDER, ordered=True
    )
    return combined


def group_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CORE_FEATURES:
        for group in GROUP_ORDER:
            values = data.loc[data["group_name"] == group, feature].dropna()
            rows.append(
                {
                    "feature": feature,
                    "group_name": group,
                    "samples": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "p01": float(values.quantile(0.01)),
                    "p05": float(values.quantile(0.05)),
                    "median": float(values.median()),
                    "p95": float(values.quantile(0.95)),
                    "p99": float(values.quantile(0.99)),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = (((len(a)-1)*np.var(a, ddof=1)) + ((len(b)-1)*np.var(b, ddof=1))) / (len(a)+len(b)-2)
    mean_diff = float(np.mean(b) - np.mean(a))
    if pooled_var <= 1e-15:
        return 0.0 if abs(mean_diff) <= 1e-15 else float(np.sign(mean_diff) * np.inf)
    return float(mean_diff / np.sqrt(pooled_var))


def shared_histograms(a: np.ndarray, b: np.ndarray, bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    all_values = np.concatenate([a, b])
    low, high = float(all_values.min()), float(all_values.max())
    if high <= low:
        high = low + 1e-9
    edges = np.linspace(low, high, bins + 1)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    return pa.astype(float), pb.astype(float)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = shared_histograms(a, b)
    return float(np.minimum(pa, pb).sum())


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = shared_histograms(a, b)
    midpoint = (pa + pb) / 2.0
    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
    return float(np.sqrt(max(0.0, 0.5 * (kl(pa, midpoint) + kl(pb, midpoint)))))


def pairwise_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CORE_FEATURES:
        for group_a, group_b, comparison_type in PAIRWISE_TESTS:
            a = data.loc[data["group_name"] == group_a, feature].dropna().to_numpy(float)
            b = data.loc[data["group_name"] == group_b, feature].dropna().to_numpy(float)
            effect = cohens_d(a, b)
            rows.append(
                {
                    "feature": feature,
                    "comparison_type": comparison_type,
                    "group_a": group_a,
                    "group_b": group_b,
                    "samples_a": int(len(a)),
                    "samples_b": int(len(b)),
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_difference_b_minus_a": float(b.mean() - a.mean()),
                    "cohens_d_b_minus_a": float(effect),
                    "absolute_effect_size": float(abs(effect)),
                    "distribution_overlap": overlap(a, b),
                    "jensen_shannon_distance": js_distance(a, b),
                }
            )
    return pd.DataFrame(rows)


def live_outside_ranges(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CORE_FEATURES:
        for training_group, live_group, class_name in [
            ("Training Alert", "Live Alert", "Alert"),
            ("Training Fatigue", "Live Fatigue", "Fatigue"),
        ]:
            train = data.loc[data["group_name"] == training_group, feature].dropna()
            live = data.loc[data["group_name"] == live_group, feature].dropna()
            p01, p05, p95, p99 = [float(train.quantile(q)) for q in (0.01, 0.05, 0.95, 0.99)]
            rows.append(
                {
                    "feature": feature,
                    "class_name": class_name,
                    "training_group": training_group,
                    "live_group": live_group,
                    "training_p01": p01,
                    "training_p05": p05,
                    "training_p95": p95,
                    "training_p99": p99,
                    "live_mean": float(live.mean()),
                    "live_median": float(live.median()),
                    "live_outside_training_p05_p95": float(((live < p05) | (live > p95)).mean()),
                    "live_outside_training_p01_p99": float(((live < p01) | (live > p99)).mean()),
                }
            )
    return pd.DataFrame(rows)


def kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.zeros_like(grid)
    std = float(np.std(values, ddof=1))
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    scale = max(scale, 1e-3)
    bandwidth = max(0.9 * scale * len(values) ** (-1/5), 1e-6)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * np.sqrt(2*np.pi))


def save_feature_plots(data: pd.DataFrame, feature: str) -> None:
    group_values = {
        group: data.loc[data["group_name"] == group, feature].dropna().to_numpy(float)
        for group in GROUP_ORDER
    }
    all_values = np.concatenate(list(group_values.values()))
    low, high = np.quantile(all_values, [0.005, 0.995])
    if high <= low:
        high = low + 1e-6
    grid = np.linspace(low, high, 300)

    plt.figure(figsize=(11, 6))
    for group in GROUP_ORDER:
        plt.plot(grid, kde(group_values[group], grid), label=group)
    plt.xlabel(feature)
    plt.ylabel("Estimated density")
    plt.title(f"{feature} — Training and Live Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{feature}_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.boxplot([group_values[g] for g in GROUP_ORDER], tick_labels=GROUP_ORDER, showfliers=False)
    plt.ylabel(feature)
    plt.title(f"{feature} — Group Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{feature}_boxplot.png", dpi=300)
    plt.close()


def save_summary_plots(comparisons: pd.DataFrame) -> None:
    positions = np.arange(len(CORE_FEATURES))
    width = 0.38

    training = comparisons[comparisons["comparison_type"] == "training_class_separation"].set_index("feature").reindex(CORE_FEATURES)
    live = comparisons[comparisons["comparison_type"] == "live_class_separation"].set_index("feature").reindex(CORE_FEATURES)
    plt.figure(figsize=(10, 6))
    plt.bar(positions-width/2, training["absolute_effect_size"], width=width, label="Training class separation")
    plt.bar(positions+width/2, live["absolute_effect_size"], width=width, label="Live class separation")
    plt.xticks(positions, CORE_FEATURES)
    plt.ylabel("Absolute Cohen's d")
    plt.title("Training vs Live Class Separation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_separation_comparison.png", dpi=300)
    plt.close()

    alert = comparisons[comparisons["comparison_type"] == "alert_domain_shift"].set_index("feature").reindex(CORE_FEATURES)
    fatigue = comparisons[comparisons["comparison_type"] == "fatigue_domain_shift"].set_index("feature").reindex(CORE_FEATURES)
    plt.figure(figsize=(10, 6))
    plt.bar(positions-width/2, alert["absolute_effect_size"], width=width, label="Training Alert vs Live Alert")
    plt.bar(positions+width/2, fatigue["absolute_effect_size"], width=width, label="Training Fatigue vs Live Fatigue")
    plt.xticks(positions, CORE_FEATURES)
    plt.ylabel("Absolute Cohen's d")
    plt.title("Class-Matched Training-to-Live Shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "live_vs_training_shift.png", dpi=300)
    plt.close()


def build_findings(comparisons: pd.DataFrame, outside: pd.DataFrame) -> List[str]:
    findings = []
    for comparison_type, label in [
        ("training_class_separation", "strongest training class-separation"),
        ("live_class_separation", "strongest live class-separation"),
        ("alert_domain_shift", "largest Alert-domain shift"),
        ("fatigue_domain_shift", "largest Fatigue-domain shift"),
    ]:
        row = comparisons[comparisons["comparison_type"] == comparison_type].sort_values("absolute_effect_size", ascending=False).iloc[0]
        findings.append(f"The {label} feature was {row['feature']} (absolute Cohen's d {row['absolute_effect_size']:.2f}).")
    for _, row in outside.iterrows():
        findings.append(
            f"{row['live_group']}: {row['live_outside_training_p05_p95']:.1%} of {row['feature']} values were outside the matching training 5th–95th percentile range."
        )
    return findings


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Core Feature Distribution Comparison")
    print("=" * 72)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    print("\nGroup sizes:")
    print(data["group_name"].value_counts(sort=False).to_string())

    stats = group_statistics(data)
    comparisons = pairwise_comparisons(data)
    outside = live_outside_ranges(data)

    stats.to_csv(RESULTS_DIR / "group_statistics.csv", index=False)
    comparisons.to_csv(RESULTS_DIR / "pairwise_comparisons.csv", index=False)
    outside.to_csv(RESULTS_DIR / "live_outside_training_ranges.csv", index=False)

    for feature in CORE_FEATURES:
        save_feature_plots(data, feature)
    save_summary_plots(comparisons)

    findings = build_findings(comparisons, outside)
    summary = {
        "project": "DriverGuardianAI V2",
        "experiment": "core_feature_distribution_comparison",
        "training_path": str(TRAIN_PATH),
        "live_log_configuration": [
            {
                "path": str(item["path"]),
                "actual_label": item["actual_label"],
                "group_name": item["group_name"],
            }
            for item in LIVE_LOGS
        ],
        "group_sizes": {str(k): int(v) for k, v in data["group_name"].value_counts(sort=False).items()},
        "features": CORE_FEATURES,
        "findings": findings,
        "important_notes": [
            "Training Fatigue combines Mild and Moderate Fatigue.",
            "Live sessions are deployment-transfer diagnostics, not independent participant validation.",
            "Confirm the live-session labels before interpreting the results.",
        ],
    }
    with (RESULTS_DIR / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nGroup statistics:")
    print(stats.to_string(index=False))
    print("\nPairwise comparisons:")
    print(comparisons[[
        "feature", "comparison_type", "group_a", "group_b", "mean_a", "mean_b",
        "cohens_d_b_minus_a", "absolute_effect_size", "distribution_overlap",
        "jensen_shannon_distance"
    ]].to_string(index=False))
    print("\nLive values outside matching training ranges:")
    print(outside.to_string(index=False))
    print("\nMain findings:")
    for finding in findings:
        print(f"- {finding}")

    print("\n" + "=" * 72)
    print("Core feature distribution comparison completed.")
    print("=" * 72)
    print("\nResults saved to:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()