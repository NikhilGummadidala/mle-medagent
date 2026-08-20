"""
Heart Disease specialist — loads a trained RandomForestClassifier from disk
and predicts heart disease risk from 13 clinical features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "heart_disease_rf.pkl"

# Matches USER_MEASUREMENT_KEYS order in langgraph_multi_agent.py
FEATURE_KEYS = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_induced_angina", "oldpeak", "slope", "ca", "thal",
]

# Stand-ins for features the patient hasn't given us. These follow the original
# UCI Cleveland encoding the model was trained on (see diag-agents/data/heart.csv),
# NOT the 0-indexed Kaggle variant: chest_pain_type 1-4, slope 1-3, thal 3/6/7.
FEATURE_DEFAULTS: dict[str, int | float] = {
    "age": 50,
    "sex": 0,
    "chest_pain_type": 4,   # asymptomatic
    "resting_blood_pressure": 120,
    "cholesterol": 200,
    "fasting_blood_sugar": 0,
    "resting_ecg": 0,       # normal
    "max_heart_rate": 140,
    "exercise_induced_angina": 0,
    "oldpeak": 0.0,
    "slope": 2,             # flat
    "ca": 0.0,
    "thal": 3.0,            # normal
}

_model = None


def load_model() -> Any:
    """Load the RandomForest model from disk. Cached after first call."""
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Heart disease model not found at {MODEL_PATH}. "
            "Run the Heart_Disease_Model notebook and export the model first."
        )

    _model = joblib.load(MODEL_PATH)
    logger.info("Heart disease model loaded from %s", MODEL_PATH)
    return _model


def predict(measurements: dict[str, int | float | None]) -> dict[str, Any]:
    """
    Run inference on heart disease measurements.

    Parameters
    ----------
    measurements : dict
        Keys from USER_MEASUREMENT_KEYS. None values are filled with defaults
        before being passed to the model.

    Returns
    -------
    dict with keys: diagnosis, confidence, probabilities, details, features_used
    """
    model = load_model()

    # Build feature vector in correct order, fill None with sensible defaults
    feature_vector = []
    provided: list[str] = []
    missing: list[str] = []
    for key in FEATURE_KEYS:
        val = measurements.get(key)
        if val is not None:
            provided.append(key)
            feature_vector.append(val)
        else:
            missing.append(key)
            feature_vector.append(FEATURE_DEFAULTS[key])

    X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    label = "high risk of heart disease" if prediction == 1 else "low risk of heart disease"
    confidence = float(max(probabilities))

    return {
        "diagnosis": label,
        "confidence": confidence,
        "probabilities": {
            "low_risk": float(probabilities[0]),
            "high_risk": float(probabilities[1]),
        },
        "details": (
            f"RandomForest inference on {len(FEATURE_KEYS)} features "
            f"({len(provided)} measured, {len(missing)} estimated from population defaults)."
        ),
        "features_used": feature_vector,
        "features_provided": provided,
        "features_missing": missing,
    }
