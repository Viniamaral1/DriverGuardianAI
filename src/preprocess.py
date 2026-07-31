"""
Preprocessing utilities for DriverGuardianAI.

Production preprocessing pipeline.

Pipeline
--------
1. Remove unused columns.
2. Merge rare classes.
3. Encode categorical variables.
4. Scale numerical variables.
5. Save preprocessing artifacts.
6. Load preprocessing artifacts.
7. Preprocess a single sample for inference.
"""

import os
import pickle

import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

# ==========================================================
# Configuration
# ==========================================================

DROP_COLUMNS = [
    "timestamp",
    "source_file",
    "state",
    "fatigue_score"
]

TARGET_COLUMN = "fatigue_level"


# ==========================================================
# Cleaning
# ==========================================================

def _clean_dataframe(df):

    df = df.copy()

    df.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
        inplace=True
    )

    if TARGET_COLUMN in df.columns:

        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(
            {
                "Severe Fatigue": "Moderate Fatigue"
            }
        )

    return df


# ==========================================================
# Fit preprocessing
# ==========================================================

def fit_preprocessor(train_df):

    train_df = _clean_dataframe(train_df)

    preprocessing = {
        "feature_encoders": {},
        "target_encoder": None,
        "scaler": None,
        "numeric_columns": None
    }

    # ------------------------------------------------------

    target_encoder = LabelEncoder()

    train_df[TARGET_COLUMN] = target_encoder.fit_transform(
        train_df[TARGET_COLUMN]
    )

    preprocessing["target_encoder"] = target_encoder

    # ------------------------------------------------------

    categorical_columns = []

    for column in train_df.columns:

        if column == TARGET_COLUMN:
            continue

        if train_df[column].dtype == object:
            categorical_columns.append(column)

    for column in categorical_columns:

        encoder = LabelEncoder()

        train_df[column] = encoder.fit_transform(
            train_df[column].astype(str)
        )

        preprocessing["feature_encoders"][column] = encoder

    # ------------------------------------------------------

    bool_columns = train_df.select_dtypes(
        include=["bool"]
    ).columns

    for column in bool_columns:

        train_df[column] = train_df[column].astype(int)

    # ------------------------------------------------------

    feature_columns = [
        c for c in train_df.columns
        if c != TARGET_COLUMN
    ]

    train_df[feature_columns] = train_df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    train_df.fillna(0, inplace=True)

    numeric_columns = train_df[
        feature_columns
    ].columns.tolist()

    preprocessing["numeric_columns"] = numeric_columns

    # ------------------------------------------------------

    scaler = StandardScaler()

    train_df[numeric_columns] = scaler.fit_transform(
        train_df[numeric_columns]
    )

    preprocessing["scaler"] = scaler

    return train_df, preprocessing


# ==========================================================
# Transform Dataset
# ==========================================================

def transform_dataset(df, preprocessing):

    df = _clean_dataframe(df)

    if TARGET_COLUMN in df.columns:

        encoder = preprocessing["target_encoder"]

        values = df[TARGET_COLUMN].astype(str)

        values = values.where(
            values.isin(encoder.classes_),
            encoder.classes_[0]
        )

        df[TARGET_COLUMN] = encoder.transform(values)

    # ------------------------------------------------------

    for column, encoder in preprocessing["feature_encoders"].items():

        values = df[column].astype(str)

        values = values.where(
            values.isin(encoder.classes_),
            encoder.classes_[0]
        )

        df[column] = encoder.transform(values)

    # ------------------------------------------------------

    bool_columns = df.select_dtypes(
        include=["bool"]
    ).columns

    for column in bool_columns:

        df[column] = df[column].astype(int)

    # ------------------------------------------------------

    numeric_columns = preprocessing["numeric_columns"]

    df[numeric_columns] = df[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df.fillna(0, inplace=True)

    scaler = preprocessing["scaler"]

    df[numeric_columns] = scaler.transform(
        df[numeric_columns]
    )

    return df


# ==========================================================
# Save preprocessing
# ==========================================================

def save_preprocessing(preprocessing, save_folder="models"):

    os.makedirs(save_folder, exist_ok=True)

    with open(
        os.path.join(save_folder, "preprocessing.pkl"),
        "wb"
    ) as f:

        pickle.dump(preprocessing, f)


# ==========================================================
# Load preprocessing
# ==========================================================

def load_preprocessing(save_folder="models"):

    with open(
        os.path.join(save_folder, "preprocessing.pkl"),
        "rb"
    ) as f:

        preprocessing = pickle.load(f)

    return preprocessing


# ==========================================================
# Prepare one sample for prediction
# ==========================================================

def preprocess_sample(sample_dict, preprocessing):

    df = pd.DataFrame([sample_dict])

    df = transform_dataset(df, preprocessing)

    return df.values.astype("float32")