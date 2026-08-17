from pathlib import Path
import json

import joblib


CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = (
    CURRENT_DIR.parent
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_prediction_pipeline.joblib"
)


METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)


THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)


model = joblib.load(
    MODEL_PATH
)


with open(
    METADATA_PATH,
    "r"
) as file:
    metadata = json.load(file)


threshold = (
    metadata[
        "decision_threshold"
    ]
)


print(
    "Model loaded successfully"
)


def get_model():

    return model


def get_threshold():

    return threshold