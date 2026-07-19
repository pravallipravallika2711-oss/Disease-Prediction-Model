import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X_train: np.ndarray, y_train: np.ndarray):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


def prepare_training_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    try:
        if len(np.unique(y)) > 1 and len(y) >= 2 * len(np.unique(y)):
            return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    except ValueError:
        pass

    return train_test_split(X, y, test_size=test_size, random_state=42)


def save_model(model, output_path: str, metadata_path: str, label_mapping: dict):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_file)

    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(label_mapping, handle, indent=2)


def load_model(model_path: str, metadata_path: str):
    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        label_mapping = json.load(handle)

    index_to_label = {value: key for key, value in label_mapping.items()}
    return model, index_to_label


def predict_probabilities(model, feature_vector: np.ndarray):
    probabilities = model.predict_proba([feature_vector])[0]
    return probabilities
