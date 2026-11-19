"""Model inputs, encoding and refusal behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from models import breast_cancer, heart_disease, skin_disease
from models.schema import (
    BREAST_FEATURE_KEYS,
    HEART_FEATURES,
    HEART_FEATURE_KEYS,
    validate_heart_measurements,
)
from tests.conftest import ROOT, png_bytes

HEART_CSV = ROOT / "diag-agents" / "data" / "heart.csv"

HIGH_RISK = {  # 67-year-old man, asymptomatic, 3 vessels, reversible defect
    "age": 67, "sex": 1, "chest_pain_type": 4, "resting_blood_pressure": 160,
    "cholesterol": 286, "fasting_blood_sugar": 0, "resting_ecg": 2, "max_heart_rate": 108,
    "exercise_induced_angina": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3,
}
LOW_RISK = {  # 41-year-old woman, atypical angina, normal tests
    "age": 41, "sex": 0, "chest_pain_type": 2, "resting_blood_pressure": 130,
    "cholesterol": 204, "fasting_blood_sugar": 0, "resting_ecg": 2, "max_heart_rate": 172,
    "exercise_induced_angina": 0, "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 3,
}


# ── schema ────────────────────────────────────────────────────────────────────

def test_schema_codes_match_the_training_data():
    """The encoding the app sends must be the encoding in the training CSV."""
    table = pd.read_csv(HEART_CSV, na_values="?")
    for feature in HEART_FEATURES:
        values = table[feature.column].dropna()
        if feature.categorical:
            assert set(values.astype(int)) == set(feature.codes), feature.key
        else:
            assert values.between(feature.low, feature.high).all(), feature.key


def test_validation_drops_out_of_range_and_unknown_values():
    clean, rejected = validate_heart_measurements({
        "age": 54, "sex": 1, "chest_pain_type": 0,      # 0 is not a valid UCI code
        "cholesterol": 0, "thal": 2, "oldpeak": "1.5", "shoe_size": 9, "ca": None,
    })
    assert clean == {"age": 54, "sex": 1, "oldpeak": 1.5}
    assert set(rejected) == {"chest_pain_type", "cholesterol", "thal", "shoe_size"}


# ── heart ─────────────────────────────────────────────────────────────────────

def test_heart_not_scored_from_intake_fields_alone():
    result = heart_disease.predict({"age": 70, "sex": 1, "resting_blood_pressure": 170})
    assert result["diagnosis"] == "insufficient data"
    assert result["confidence"] is None
    assert "chest pain type" in result["details"]


def test_heart_separates_clear_cases():
    high = heart_disease.predict(HIGH_RISK)
    low = heart_disease.predict(LOW_RISK)
    assert high["diagnosis"] == "elevated likelihood of heart disease"
    assert low["diagnosis"] == "lower likelihood of heart disease"
    assert high["probability_of_disease"] > 0.8 > 0.2 > low["probability_of_disease"]


def test_heart_runs_with_partial_measurements_and_says_so():
    partial = {k: HIGH_RISK[k] for k in ("age", "sex", "chest_pain_type", "resting_blood_pressure")}
    result = heart_disease.predict(partial)
    assert result["features_missing"] == [k for k in HEART_FEATURE_KEYS if k not in partial]
    assert "4 of 13" in result["summary"]
    assert 0.0 < result["probability_of_disease"] < 1.0


def test_heart_model_agrees_with_training_labels():
    """Guards against the feature order or encoding drifting from the model."""
    table = pd.read_csv(HEART_CSV, na_values="?")
    truth = (table["num"] > 0).to_numpy()
    predicted = []
    for row in table.to_dict("records"):
        measurements = {
            f.key: (None if pd.isna(row[f.column]) else row[f.column]) for f in HEART_FEATURES
        }
        predicted.append(heart_disease.predict(measurements)["probability_of_disease"] >= 0.5)
    assert (np.array(predicted) == truth).mean() > 0.85


# ── breast ────────────────────────────────────────────────────────────────────

def test_breast_tabular_requires_all_features():
    result = breast_cancer.predict(features={"radius_mean": 14.0})
    assert result["diagnosis"] == "insufficient data"


def test_breast_tabular_matches_known_samples():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    correct = 0
    for row, target in zip(data.data[:60], data.target[:60]):
        result = breast_cancer.predict(features=dict(zip(BREAST_FEATURE_KEYS, row)))
        correct += (result["diagnosis"] == "malignant pattern") == (target == 0)
    assert correct >= 56


def test_colour_photo_is_not_scored_as_a_mammogram(tmp_path):
    path = tmp_path / "selfie.png"
    path.write_bytes(png_bytes((400, 400), (200, 120, 90)))
    result = breast_cancer.predict(image_path=str(path))
    assert result["diagnosis"] == "image not assessed"
    assert result["confidence"] is None


def test_evenly_lit_grayscale_is_not_scored_as_a_mammogram(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (400, 400), 128).save(path)
    with pytest.raises(breast_cancer.NotAMammogramError):
        breast_cancer.load_mammogram(str(path))


def test_mammogram_like_image_passes_the_input_check(tmp_path):
    pixels = np.zeros((600, 400), dtype=np.uint8)
    pixels[100:500, 0:250] = 140
    path = tmp_path / "scan.png"
    Image.fromarray(pixels).save(path)
    assert breast_cancer.load_mammogram(str(path)).shape == (600, 400)


@pytest.mark.skipif(
    not (breast_cancer.MAMMOSCREEN_DIR / "model.safetensors").exists(),
    reason="mammogram weights not downloaded (python scripts/download_models.py)",
)
def test_mammoscreen_weights_load_and_score(tmp_path):
    pixels = np.zeros((1024, 768), dtype=np.uint8)
    pixels[100:900, 0:500] = 120
    path = tmp_path / "scan.png"
    Image.fromarray(pixels).save(path)
    result = breast_cancer.predict(image_path=str(path))
    assert result["diagnosis"] in {"flagged for follow-up", "not flagged by the screening model"}
    assert 0.0 <= result["screening_score"] <= 1.0


# ── skin ──────────────────────────────────────────────────────────────────────

def test_grayscale_image_is_not_scored_as_a_skin_photo(tmp_path):
    path = tmp_path / "scan.png"
    Image.new("L", (300, 300), 90).save(path)
    result = skin_disease.predict(str(path))
    assert result["diagnosis"] == "image not assessed"


@pytest.mark.skipif(
    not (skin_disease.MODEL_DIR / "model.safetensors").exists(),
    reason="skin weights not downloaded (python scripts/download_models.py)",
)
def test_skin_model_loads_and_returns_probabilities(tmp_path):
    path = tmp_path / "skin.png"
    path.write_bytes(png_bytes((300, 300), (190, 140, 120)))
    result = skin_disease.predict(str(path))
    assert result["diagnosis"] in {name for name, _ in skin_disease.CLASSES}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
