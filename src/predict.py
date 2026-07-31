"""
Production inference utilities for DriverGuardianAI.

This module:
- Loads the trained PyTorch model.
- Loads the complete preprocessing pipeline.
- Validates incoming features.
- Applies the same preprocessing used during training.
- Returns the predicted fatigue class, confidence and probabilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.fatigue_model import FatigueResidualNN
from src.preprocess import transform_dataset
from src.utils import load_preprocessor


class Predictor:
    """
    Production inference engine for DriverGuardianAI.
    """

    CLASS_DISPLAY_NAMES = {
        "Alert": "Alert",
        "Mild Fatigue": "Mild Fatigue",
        "Moderate Fatigue": "Moderate/Severe Fatigue",
        "Severe Fatigue": "Moderate/Severe Fatigue"
    }

    def __init__(
        self,
        model_path="models/driver_guardian_best.pth",
        preprocessing_path="models/preprocessing.pkl",
        hidden_dims=None,
        dropout=0.30,
        num_classes=3,
        device=None
    ):
        """
        Initialise the predictor.

        Parameters
        ----------
        model_path : str
            Path to the saved PyTorch state dictionary.

        preprocessing_path : str
            Path to the saved preprocessing dictionary.

        hidden_dims : list[int], optional
            Must match the architecture used during training.

        dropout : float
            Must match the training configuration.

        num_classes : int
            Number of output classes.

        device : str or torch.device, optional
            CPU or CUDA device.
        """

        self.model_path = Path(model_path)
        self.preprocessing_path = Path(preprocessing_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file was not found: {self.model_path}"
            )

        if not self.preprocessing_path.exists():
            raise FileNotFoundError(
                "Preprocessing file was not found: "
                f"{self.preprocessing_path}"
            )

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_classes = num_classes

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # --------------------------------------------------
        # Load fitted preprocessing pipeline
        # --------------------------------------------------

        self.preprocessing = load_preprocessor(
            str(self.preprocessing_path)
        )

        required_preprocessing_keys = {
            "feature_encoders",
            "target_encoder",
            "scaler",
            "numeric_columns"
        }

        missing_keys = required_preprocessing_keys.difference(
            self.preprocessing.keys()
        )

        if missing_keys:
            raise KeyError(
                "The saved preprocessing pipeline is missing: "
                f"{sorted(missing_keys)}"
            )

        self.target_encoder = self.preprocessing["target_encoder"]

        self.feature_columns = list(
            self.preprocessing["numeric_columns"]
        )

        if not self.feature_columns:
            raise ValueError(
                "No feature columns were found in preprocessing.pkl."
            )

        self.input_dim = len(self.feature_columns)

        # --------------------------------------------------
        # Build the same architecture used during training
        # --------------------------------------------------

        self.model = FatigueResidualNN(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            num_classes=self.num_classes
        )

        state_dict = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.")
        print(f"Expected features ({self.input_dim}): {self.feature_columns}")

    def _validate_features(self, features):
        """
        Validate and convert incoming features to a DataFrame.
        """

        if isinstance(features, dict):
            dataframe = pd.DataFrame([features])

        elif isinstance(features, pd.DataFrame):
            if features.empty:
                raise ValueError(
                    "The supplied DataFrame is empty."
                )

            dataframe = features.copy()

        else:
            raise TypeError(
                "Features must be supplied as a dictionary "
                "or pandas DataFrame."
            )

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing required features: "
                f"{missing_features}"
            )

        # Ignore any unexpected extra columns and preserve
        # exactly the same feature order used during training.
        return dataframe[self.feature_columns].copy()

    def _preprocess_features(self, features):
        """
        Apply the fitted training preprocessing pipeline.
        """

        dataframe = self._validate_features(features)

        processed = transform_dataset(
            dataframe,
            self.preprocessing
        )

        missing_after_transform = [
            feature
            for feature in self.feature_columns
            if feature not in processed.columns
        ]

        if missing_after_transform:
            raise ValueError(
                "Features disappeared during preprocessing: "
                f"{missing_after_transform}"
            )

        processed = processed[
            self.feature_columns
        ].astype("float32")

        values = processed.to_numpy(
            dtype=np.float32
        )

        if values.shape[1] != self.input_dim:
            raise ValueError(
                "Processed feature count does not match the model. "
                f"Expected {self.input_dim}, received {values.shape[1]}."
            )

        return values

    def predict(self, features):
        """
        Predict fatigue for one sample.

        Returns
        -------
        dict
            Predicted class, confidence and class probabilities.
        """

        processed_features = self._preprocess_features(
            features
        )

        if processed_features.shape[0] != 1:
            raise ValueError(
                "predict() accepts exactly one sample. "
                "Use predict_batch() for multiple samples."
            )

        input_tensor = torch.from_numpy(
            processed_features
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0].cpu().numpy()

        predicted_class = int(
            np.argmax(probabilities)
        )

        raw_label = self.target_encoder.inverse_transform(
            [predicted_class]
        )[0]

        prediction = self.CLASS_DISPLAY_NAMES.get(
            str(raw_label),
            str(raw_label)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        probability_labels = self.target_encoder.inverse_transform(
            np.arange(len(probabilities))
        )

        probability_dictionary = {}

        for label, probability in zip(
            probability_labels,
            probabilities
        ):
            display_label = self.CLASS_DISPLAY_NAMES.get(
                str(label),
                str(label)
            )

            probability_dictionary[display_label] = float(
                probability
            )

        return {
            "prediction": prediction,
            "raw_prediction": str(raw_label),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probability_dictionary,
            "feature_order": self.feature_columns.copy()
        }

    def predict_batch(self, features):
        """
        Predict fatigue for multiple samples.
        """

        processed_features = self._preprocess_features(
            features
        )

        input_tensor = torch.from_numpy(
            processed_features
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            ).cpu().numpy()

        results = []

        probability_labels = self.target_encoder.inverse_transform(
            np.arange(probabilities.shape[1])
        )

        for sample_probabilities in probabilities:
            predicted_class = int(
                np.argmax(sample_probabilities)
            )

            raw_label = self.target_encoder.inverse_transform(
                [predicted_class]
            )[0]

            prediction = self.CLASS_DISPLAY_NAMES.get(
                str(raw_label),
                str(raw_label)
            )

            probability_dictionary = {}

            for label, probability in zip(
                probability_labels,
                sample_probabilities
            ):
                display_label = self.CLASS_DISPLAY_NAMES.get(
                    str(label),
                    str(label)
                )

                probability_dictionary[display_label] = float(
                    probability
                )

            results.append(
                {
                    "prediction": prediction,
                    "raw_prediction": str(raw_label),
                    "predicted_class": predicted_class,
                    "confidence": float(
                        sample_probabilities[predicted_class]
                    ),
                    "probabilities": probability_dictionary
                }
            )

        return results