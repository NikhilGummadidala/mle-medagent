"""Heart disease specialist.

Gradient-boosted trees trained by ``scripts/train_heart_model.py`` on the UCI
Cleveland data, using the encoding defined in ``models/schema.py``. Missing
measurements are passed as NaN (the model was trained for that); nothing is
filled in with invented defaults, and the model is not run at all unless the
minimum informative inputs are present.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from models.schema import (
    HEART_FEATURE_BY_KEY,
    HEART_FEATURE_KEYS,
    HEART_HELPFUL_KEYS,
    HEART_MIN_PROVIDED,
    HEART_REQUIRED_KEYS,
    validate_heart_measurements,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_PATH = MODEL_DIR / "heart_disease_hgb.joblib"
METADATA_PATH = MODEL_DIR / "heart_disease_hgb.metadata.json"

_model = None
_metadata: dict[str, Any] = {}
_lock = threading.Lock()


def load_model() -> Any:
    """Load the model from disk. Cached after first call."""
    global _model, _metadata
    with _lock:
        if _model is not None:
            return _model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Heart disease model file is missing. Run: python scripts/train_heart_model.py"
            )
        _model = joblib.load(MODEL_PATH)
        if METADATA_PATH.exists():
            _metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        logger.info("Heart disease model loaded from %s", MODEL_PATH)
        return _model


def _label(key: str) -> str:
    return HEART_FEATURE_BY_KEY[key].label


def insufficient_data(provided: list[str], rejected: dict[str, str] | None = None) -> dict[str, Any]:
    needed = [k for k in HEART_REQUIRED_KEYS if k not in provided]
    parts = []
    if needed:
        parts.append("Still needed: " + ", ".join(_label(k) for k in needed) + ".")
    if "chest_pain_type" in needed:
        parts.append(
            "For chest pain type, tell me whether you get chest pain or pressure, whether "
            "physical effort brings it on, and whether rest relieves it (or say you have "
            "no chest pain)."
        )
    if rejected:
        parts.append(
            "Ignored as implausible: "
            + "; ".join(f"{k} ({why})" for k, why in rejected.items()) + "."
        )
    return {
        "diagnosis": "insufficient data",
        "confidence": None,
        "summary": "No heart assessment was run: not enough measurements were provided.",
        "details": " ".join(parts),
        "features_provided": provided,
        "features_missing": [k for k in HEART_FEATURE_KEYS if k not in provided],
        "features_rejected": rejected or {},
    }


REQUIRED_QUESTIONS = {
    "age": "How old are you?",
    "sex": "What is your sex (male or female)? The model was built using it.",
    "chest_pain_type": (
        "Do you get chest pain, pressure or tightness? If you do: does physical effort "
        "(walking, climbing stairs) bring it on, and does it ease when you rest?"
    ),
}
HELPFUL_QUESTIONS = {
    "resting_blood_pressure": "Do you know your blood pressure (for example 120/80)?",
    "cholesterol": "Do you know your total cholesterol?",
}


def follow_up_questions(
    measurements: dict[str, Any] | None, helpful_already_asked: bool = False,
) -> tuple[list[str], str | None]:
    """Questions to put to the patient before estimating.

    Returns ``(questions, stage)``. Stage "required" means the model cannot run
    without the answers. Stage "helpful" is asked once, when neither blood
    pressure nor cholesterol is known; the patient can decline and still get an
    estimate. An empty list means: go ahead.
    """
    clean, _ = validate_heart_measurements(measurements)
    missing_required = [k for k in HEART_REQUIRED_KEYS if k not in clean]
    if missing_required:
        return [REQUIRED_QUESTIONS[k] for k in missing_required], "required"
    if not helpful_already_asked and not any(k in clean for k in HEART_HELPFUL_KEYS):
        return [HELPFUL_QUESTIONS[k] for k in HEART_HELPFUL_KEYS], "helpful"
    return [], None


def more_information_needed(
    measurements: dict[str, Any] | None, questions: list[str], stage: str,
) -> dict[str, Any]:
    """A turn that asks follow-up questions instead of giving an estimate."""
    clean, rejected = validate_heart_measurements(measurements)
    provided = [k for k in HEART_FEATURE_KEYS if k in clean]
    if stage == "required":
        intro = "Before I can estimate your heart disease likelihood, I need to ask:"
        outro = "I can't give an estimate without these."
    else:
        intro = "I have enough for a first estimate, but it will be more reliable if you can tell me:"
        outro = 'If you don\'t know, just say "skip" and I\'ll estimate with what I have.'
    return {
        "diagnosis": "more information needed",
        "confidence": None,
        "summary": intro,
        "questions": questions,
        "question_stage": stage,
        "details": outro,
        "features_provided": provided,
        "features_missing": [k for k in HEART_FEATURE_KEYS if k not in clean],
        "features_rejected": rejected,
    }


def inputs_sufficient(measurements: dict[str, Any] | None) -> bool:
    """True when the minimum informative measurements are present."""
    clean, _ = validate_heart_measurements(measurements)
    return len(clean) >= HEART_MIN_PROVIDED and all(k in clean for k in HEART_REQUIRED_KEYS)


def predict(measurements: dict[str, Any] | None) -> dict[str, Any]:
    """Estimate the likelihood of coronary heart disease.

    ``measurements`` uses the keys and encoding from ``models/schema.py``.
    Returns a dict with ``diagnosis``, ``confidence`` (probability of the stated
    class, or None when the model was not run), ``summary`` and ``details``.
    """
    clean, rejected = validate_heart_measurements(measurements)
    provided = [k for k in HEART_FEATURE_KEYS if k in clean]
    missing = [k for k in HEART_FEATURE_KEYS if k not in clean]

    if not inputs_sufficient(clean):
        return insufficient_data(provided, rejected)

    model = load_model()
    row = np.array([[clean.get(k, np.nan) for k in HEART_FEATURE_KEYS]], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])
    elevated = probability >= 0.5

    cv = _metadata.get("cross_validation", {})
    full = cv.get("all_13_features", {}).get("auroc")
    minimal = cv.get("minimum_required_plus_bp", {}).get("auroc")
    quality = ""
    if full and minimal:
        quality = (
            f" In cross-validation the model's AUROC was {full:.2f} with all 13 measurements "
            f"and {minimal:.2f} with only the minimum set."
        )
    details = (
        f"Based on {len(provided)} of {len(HEART_FEATURE_KEYS)} measurements "
        f"({', '.join(provided)}). Missing measurements were left blank, not guessed."
        f"{quality} The model was built from 303 patients already referred for heart "
        "testing, so this is not a general-population risk."
    )
    if rejected:
        details += " Ignored as implausible: " + "; ".join(
            f"{k} ({why})" for k, why in rejected.items()
        ) + "."

    return {
        "diagnosis": (
            "elevated likelihood of heart disease" if elevated
            else "lower likelihood of heart disease"
        ),
        "confidence": probability if elevated else 1.0 - probability,
        "probability_of_disease": probability,
        "summary": (
            f"Model estimate of heart disease: {probability:.0%} "
            f"(from {len(provided)} of {len(HEART_FEATURE_KEYS)} measurements)."
        ),
        "details": details,
        "features_provided": provided,
        "features_missing": missing,
        "features_rejected": rejected,
    }
