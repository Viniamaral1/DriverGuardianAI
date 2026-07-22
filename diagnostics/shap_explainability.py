"""
SHAP explainability for DriverGuardianAI.

Experiment 14 — corrected version
---------------------------------
This version explains the calibrated Histogram Gradient Boosting
classifier in the numeric feature space produced by its fitted
preprocessing pipeline.

Why this version is necessary
-----------------------------
The raw model input contains a text feature:

    condition = none / glasses / dark / hat

SHAP's tabular masker uses numeric comparisons internally and cannot
directly process a mixed array containing both numbers and strings.

This script therefore:

1. Loads the saved calibrated scikit-learn Pipeline.
2. Applies the fitted preprocessor to the selected samples.
3. Retrieves the transformed numeric feature names.
4. Explains the fitted classifier using numeric transformed data.
5. Saves global and individual SHAP explanations.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_exp3.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "driver_guardian_calibrated_hgb.joblib"
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "experiment14_shap_explainability"

GLOBAL_IMPORTANCE_PATH = OUTPUT_DIRECTORY / "global_shap_importance.csv"
EXPLANATION_SAMPLES_PATH = OUTPUT_DIRECTORY / "explained_samples.csv"
SUMMARY_JSON_PATH = OUTPUT_DIRECTORY / "shap_summary.json"
GLOBAL_BAR_PLOT_PATH = OUTPUT_DIRECTORY / "global_shap_importance.png"
BEESWARM_PLOT_PATH = OUTPUT_DIRECTORY / "shap_beeswarm.png"
ALERT_WATERFALL_PATH = OUTPUT_DIRECTORY / "representative_alert_waterfall.png"
FATIGUE_WATERFALL_PATH = OUTPUT_DIRECTORY / "representative_fatigue_waterfall.png"
FALSE_POSITIVE_WATERFALL_PATH = OUTPUT_DIRECTORY / "false_positive_waterfall.png"
FALSE_NEGATIVE_WATERFALL_PATH = OUTPUT_DIRECTORY / "false_negative_waterfall.png"

RANDOM_STATE = 42
TARGET_COLUMN = "fatigue_level"
FEATURE_COLUMNS = [
    "ear", "yawn_score", "head_tilt", "hands_detected",
    "condition", "low_light", "face_confidence", "blink_count",
]
NUMERIC_FEATURES = [
    "ear", "yawn_score", "head_tilt", "hands_detected",
    "low_light", "face_confidence", "blink_count",
]
CLASS_NAMES = ["Alert", "Fatigue"]
TEST_PARTICIPANT_FRACTION = 0.25
BACKGROUND_SAMPLE_SIZE = 100
EXPLANATION_SAMPLE_SIZE = 200
MAX_EVALUATIONS = 3000


def extract_participant(source_file):
    if pd.isna(source_file):
        raise ValueError("A missing source_file value was found.")
    value = str(source_file).strip()
    if not value:
        raise ValueError("An empty source_file value was found.")
    return value.split("_", maxsplit=1)[0].lower()


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataframe = pd.read_csv(DATASET_PATH)
    required_columns = {"source_file", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    dataframe = dataframe.copy()
    dataframe["participant"] = dataframe["source_file"].apply(extract_participant)

    target_mapping = {
        "Alert": 0,
        "Mild Fatigue": 1,
        "Moderate Fatigue": 1,
        "Severe Fatigue": 1,
    }
    unknown_targets = set(
        dataframe[TARGET_COLUMN].dropna().astype(str).unique()
    ).difference(target_mapping.keys())
    if unknown_targets:
        raise ValueError(f"Unknown target labels: {sorted(unknown_targets)}")

    dataframe["binary_target"] = dataframe[TARGET_COLUMN].astype(str).map(target_mapping)
    dataframe["low_light"] = (
        dataframe["low_light"].astype(str).str.lower().map(
            {
                "true": 1.0, "false": 0.0,
                "1": 1.0, "0": 0.0,
                "1.0": 1.0, "0.0": 0.0,
            }
        )
    )

    for feature in NUMERIC_FEATURES:
        dataframe[feature] = pd.to_numeric(dataframe[feature], errors="coerce")

    dataframe["condition"] = dataframe["condition"].astype(str)
    return dataframe


def create_test_split(dataframe):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_PARTICIPANT_FRACTION,
        random_state=RANDOM_STATE,
    )
    _, test_indices = next(
        splitter.split(dataframe, groups=dataframe["participant"])
    )
    test_dataframe = dataframe.iloc[test_indices].copy()
    test_participants = sorted(test_dataframe["participant"].unique().tolist())
    return test_dataframe, test_participants


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Calibrated model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    required_keys = {"pipeline", "fatigue_threshold", "feature_columns", "class_names"}
    missing_keys = required_keys.difference(bundle.keys())
    if missing_keys:
        raise KeyError(f"Model bundle is missing keys: {sorted(missing_keys)}")

    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(
            "Saved model feature order does not match the explainability script."
        )

    pipeline = bundle["pipeline"]
    if "preprocessor" not in pipeline.named_steps:
        raise KeyError("The saved pipeline does not contain a 'preprocessor' step.")
    if "classifier" not in pipeline.named_steps:
        raise KeyError("The saved pipeline does not contain a 'classifier' step.")

    return bundle


def clean_transformed_feature_name(name):
    name = str(name)
    for prefix in ["numeric__", "categorical__"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def get_transformed_feature_names(preprocessor):
    try:
        names = preprocessor.get_feature_names_out()
    except Exception as error:
        raise RuntimeError(
            "Could not retrieve transformed feature names from the fitted preprocessor."
        ) from error
    return [clean_transformed_feature_name(name) for name in names]


def prepare_explanation_samples(test_dataframe, pipeline, threshold):
    features = test_dataframe[FEATURE_COLUMNS].copy()
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    scored = test_dataframe.copy()
    scored["fatigue_probability"] = probabilities
    scored["calibrated_prediction"] = predictions
    scored["is_correct"] = scored["binary_target"] == scored["calibrated_prediction"]

    alert_pool = scored[scored["binary_target"] == 0]
    fatigue_pool = scored[scored["binary_target"] == 1]
    half_size = max(1, EXPLANATION_SAMPLE_SIZE // 2)

    alert_sample = alert_pool.sample(
        n=min(half_size, len(alert_pool)),
        random_state=RANDOM_STATE,
    )
    fatigue_sample = fatigue_pool.sample(
        n=min(half_size, len(fatigue_pool)),
        random_state=RANDOM_STATE,
    )

    explanation_dataframe = pd.concat(
        [alert_sample, fatigue_sample], ignore_index=True
    )

    representative_alert = alert_pool.sort_values(
        "fatigue_probability", ascending=True
    ).iloc[0]
    representative_fatigue = fatigue_pool.sort_values(
        "fatigue_probability", ascending=False
    ).iloc[0]

    false_positive_pool = scored[
        (scored["binary_target"] == 0)
        & (scored["calibrated_prediction"] == 1)
    ]
    false_negative_pool = scored[
        (scored["binary_target"] == 1)
        & (scored["calibrated_prediction"] == 0)
    ]

    false_positive = None
    if not false_positive_pool.empty:
        false_positive = false_positive_pool.sort_values(
            "fatigue_probability", ascending=False
        ).iloc[0]

    false_negative = None
    if not false_negative_pool.empty:
        false_negative = false_negative_pool.sort_values(
            "fatigue_probability", ascending=True
        ).iloc[0]

    return {
        "scored_test": scored,
        "explanation_dataframe": explanation_dataframe,
        "representative_alert": representative_alert,
        "representative_fatigue": representative_fatigue,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def classifier_fatigue_probability(classifier):
    def predict_probability(values):
        values = np.asarray(values, dtype=np.float64)
        return classifier.predict_proba(values)[:, 1]
    return predict_probability


def build_explainer(classifier, background_transformed, transformed_feature_names):
    try:
        print("Trying SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(
            classifier,
            data=background_transformed,
            feature_names=transformed_feature_names,
            model_output="probability",
        )
        return explainer, "tree"
    except Exception as tree_error:
        print("TreeExplainer was not compatible with this SHAP/sklearn version.")
        print("Using numeric PermutationExplainer instead.")
        print(f"TreeExplainer detail: {tree_error}")

        masker = shap.maskers.Independent(
            np.asarray(background_transformed, dtype=np.float64)
        )
        explainer = shap.Explainer(
            classifier_fatigue_probability(classifier),
            masker,
            algorithm="permutation",
            feature_names=transformed_feature_names,
            output_names=["Fatigue probability"],
        )
        return explainer, "permutation"


def explain_batch(explainer, transformed_values, explainer_type):
    values = np.asarray(transformed_values, dtype=np.float64)
    if explainer_type == "tree":
        explanation = explainer(values)
    else:
        explanation = explainer(
            values,
            max_evals=MAX_EVALUATIONS,
            silent=False,
        )

    if explanation.values.ndim == 3:
        explanation.values = explanation.values[:, :, 0]
        base_values = np.asarray(explanation.base_values)
        if base_values.ndim > 1:
            explanation.base_values = base_values[:, 0]

    return explanation


def explain_single_sample(
    explainer,
    explainer_type,
    preprocessor,
    row,
    transformed_feature_names,
    output_path,
):
    raw_features = pd.DataFrame(
        [{feature: row[feature] for feature in FEATURE_COLUMNS}]
    )
    transformed = np.asarray(
        preprocessor.transform(raw_features),
        dtype=np.float64,
    )
    explanation = explain_batch(explainer, transformed, explainer_type)

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(
        explanation[0],
        max_display=min(12, len(transformed_feature_names)),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    contribution_values = np.asarray(explanation.values[0], dtype=float)
    ranked_indices = np.argsort(np.abs(contribution_values))[::-1]

    top_contributions = []
    for index in ranked_indices:
        contribution = float(contribution_values[index])
        top_contributions.append(
            {
                "feature": transformed_feature_names[index],
                "shap_value": contribution,
                "direction": "toward Fatigue" if contribution > 0 else "toward Alert",
            }
        )

    return {
        "actual_class": CLASS_NAMES[int(row["binary_target"])],
        "predicted_class": CLASS_NAMES[int(row["calibrated_prediction"])],
        "fatigue_probability": float(row["fatigue_probability"]),
        "source_file": str(row["source_file"]),
        "participant": str(row["participant"]),
        "top_contributions": top_contributions,
    }


def main():
    print("=" * 72)
    print("DriverGuardianAI")
    print("Experiment 14: SHAP Explainability")
    print("=" * 72)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset()
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    threshold = float(bundle["fatigue_threshold"])
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_feature_names = get_transformed_feature_names(preprocessor)

    test_dataframe, test_participants = create_test_split(dataframe)

    print("\nUntouched test participants:")
    for participant in test_participants:
        print(f"  - {participant}")

    print(f"\nTest samples: {len(test_dataframe)}")
    print(f"Calibrated threshold: {threshold:.2f}")

    print("\nTransformed features:")
    for feature_name in transformed_feature_names:
        print(f"  - {feature_name}")

    selected = prepare_explanation_samples(test_dataframe, pipeline, threshold)
    scored_test = selected["scored_test"]
    explanation_dataframe = selected["explanation_dataframe"]

    background_raw = scored_test[FEATURE_COLUMNS].sample(
        n=min(BACKGROUND_SAMPLE_SIZE, len(scored_test)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    explanation_raw = explanation_dataframe[FEATURE_COLUMNS].reset_index(drop=True)

    print("\nApplying fitted preprocessing...")
    background_transformed = np.asarray(
        preprocessor.transform(background_raw),
        dtype=np.float64,
    )
    explanation_transformed = np.asarray(
        preprocessor.transform(explanation_raw),
        dtype=np.float64,
    )

    if background_transformed.shape[1] != len(transformed_feature_names):
        raise RuntimeError(
            "The transformed feature matrix and feature-name list do not match."
        )

    print("\nBuilding SHAP explainer...")
    explainer, explainer_type = build_explainer(
        classifier,
        background_transformed,
        transformed_feature_names,
    )
    print(f"Explainer type: {explainer_type}")

    print("\nCalculating global SHAP explanations...")
    shap_values = explain_batch(
        explainer,
        explanation_transformed,
        explainer_type,
    )

    mean_absolute_values = np.mean(np.abs(shap_values.values), axis=0)
    global_importance = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "mean_absolute_shap": mean_absolute_values,
        }
    ).sort_values("mean_absolute_shap", ascending=False)

    total_importance = float(global_importance["mean_absolute_shap"].sum())
    if total_importance > 0:
        global_importance["relative_importance"] = (
            global_importance["mean_absolute_shap"] / total_importance
        )
    else:
        global_importance["relative_importance"] = 0.0

    global_importance.to_csv(GLOBAL_IMPORTANCE_PATH, index=False)

    plt.figure(figsize=(10, 7))
    shap.plots.bar(
        shap_values,
        max_display=min(15, len(transformed_feature_names)),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(GLOBAL_BAR_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 8))
    shap.plots.beeswarm(
        shap_values,
        max_display=min(15, len(transformed_feature_names)),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(BEESWARM_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nCreating individual explanations...")
    individual_summaries = {}

    individual_summaries["representative_alert"] = explain_single_sample(
        explainer,
        explainer_type,
        preprocessor,
        selected["representative_alert"],
        transformed_feature_names,
        ALERT_WATERFALL_PATH,
    )

    individual_summaries["representative_fatigue"] = explain_single_sample(
        explainer,
        explainer_type,
        preprocessor,
        selected["representative_fatigue"],
        transformed_feature_names,
        FATIGUE_WATERFALL_PATH,
    )

    if selected["false_positive"] is not None:
        individual_summaries["false_positive"] = explain_single_sample(
            explainer,
            explainer_type,
            preprocessor,
            selected["false_positive"],
            transformed_feature_names,
            FALSE_POSITIVE_WATERFALL_PATH,
        )

    if selected["false_negative"] is not None:
        individual_summaries["false_negative"] = explain_single_sample(
            explainer,
            explainer_type,
            preprocessor,
            selected["false_negative"],
            transformed_feature_names,
            FALSE_NEGATIVE_WATERFALL_PATH,
        )

    scored_test[
        [
            "source_file", "participant", TARGET_COLUMN,
            "binary_target", "fatigue_probability",
            "calibrated_prediction", "is_correct",
            *FEATURE_COLUMNS,
        ]
    ].to_csv(EXPLANATION_SAMPLES_PATH, index=False)

    summary = {
        "experiment": "experiment14_shap_explainability",
        "model": "calibrated_hist_gradient_boosting",
        "explainer_type": explainer_type,
        "fatigue_threshold": threshold,
        "test_participants": test_participants,
        "raw_feature_columns": FEATURE_COLUMNS,
        "transformed_feature_columns": transformed_feature_names,
        "global_feature_importance": global_importance.to_dict(orient="records"),
        "individual_explanations": individual_summaries,
        "important_note": (
            "SHAP values explain how model inputs influenced predicted "
            "Fatigue probability. They do not establish medical or causal effects."
        ),
    }

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nGlobal SHAP importance:")
    print(global_importance.to_string(index=False))

    print("\n" + "=" * 72)
    print("Experiment 14 completed successfully.")
    print("=" * 72)
    print(f"\nResults saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()