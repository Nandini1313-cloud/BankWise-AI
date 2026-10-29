from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"


def get_model_path():
    comparison_path = ROOT / "reports" / "model_comparison.json"

    import json

    with comparison_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    best = max(results, key=lambda item: item["f1_yes"])
    filename = best["model"].lower().replace(" ", "_") + "_model.joblib"
    return MODELS_DIR / filename


@pytest.fixture(scope="module")
def model():
    path = get_model_path()
    assert path.exists(), f"Trained model missing: {path}"
    return joblib.load(path)


@pytest.fixture(scope="module")
def sample_customer():
    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop(columns=["y", "duration"])
    return X.iloc[[0]]


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_model_file_exists():
    assert get_model_path().exists()


def test_model_accepts_customer_data(model, sample_customer):
    probability = model.predict_proba(sample_customer)
    assert probability.shape == (1, 2)


def test_prediction_probability_is_valid(model, sample_customer):
    probability = model.predict_proba(sample_customer)[0, 1]
    assert 0.0 <= probability <= 1.0


def test_model_returns_valid_class(model, sample_customer):
    prediction = model.predict(sample_customer)
    assert prediction[0] in [0, 1]
