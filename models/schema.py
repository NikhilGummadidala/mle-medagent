"""Single source of truth for the tabular model inputs.

Every place that names a heart or breast feature (API, LLM extraction, graph
state, training scripts, inference) imports from here, so the encoding the
model was trained on is the encoding the app sends.

Heart features use the original UCI Cleveland encoding found in
``diag-agents/data/heart.csv``. Categorical codes are NOT zero-based.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Feature:
    key: str                      # name used across the app
    column: str                   # column name in the UCI CSV
    label: str                    # human-readable name
    low: float | None = None      # inclusive numeric range (numeric features)
    high: float | None = None
    codes: dict[int, str] = field(default_factory=dict)  # categorical code -> meaning
    integer: bool = True

    @property
    def categorical(self) -> bool:
        return bool(self.codes)


HEART_FEATURES: tuple[Feature, ...] = (
    Feature("age", "age", "age (years)", 18, 100),
    Feature("sex", "sex", "sex", codes={0: "female", 1: "male"}),
    Feature(
        "chest_pain_type", "cp", "chest pain type",
        codes={1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    ),
    Feature("resting_blood_pressure", "trestbps", "resting systolic blood pressure (mmHg)", 70, 250),
    Feature("cholesterol", "chol", "serum cholesterol (mg/dl)", 80, 700),
    Feature("fasting_blood_sugar", "fbs", "fasting blood sugar above 120 mg/dl", codes={0: "no", 1: "yes"}),
    Feature(
        "resting_ecg", "restecg", "resting ECG",
        codes={0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
    ),
    Feature("max_heart_rate", "thalach", "peak heart rate in an exercise test (bpm)", 50, 230),
    Feature("exercise_induced_angina", "exang", "exercise-induced angina", codes={0: "no", 1: "yes"}),
    Feature("oldpeak", "oldpeak", "ST depression induced by exercise (mm)", 0, 10, integer=False),
    Feature("slope", "slope", "slope of the peak exercise ST segment", codes={1: "upsloping", 2: "flat", 3: "downsloping"}),
    Feature("ca", "ca", "major vessels coloured by fluoroscopy", codes={0: "0", 1: "1", 2: "2", 3: "3"}),
    Feature("thal", "thal", "thallium stress test result", codes={3: "normal", 6: "fixed defect", 7: "reversible defect"}),
)

HEART_FEATURE_KEYS: list[str] = [f.key for f in HEART_FEATURES]
HEART_FEATURE_BY_KEY: dict[str, Feature] = {f.key: f for f in HEART_FEATURES}

# The model is not run without these. Cross-validation (see
# saved_models/heart_disease_hgb.metadata.json) shows age/sex/blood pressure
# alone barely separate patients; adding chest pain type is what makes the
# estimate informative.
HEART_REQUIRED_KEYS: tuple[str, ...] = ("age", "sex", "chest_pain_type")
HEART_MIN_PROVIDED = len(HEART_REQUIRED_KEYS)
# Asked for once before an estimate is given, but the patient may not know them.
HEART_HELPFUL_KEYS: tuple[str, ...] = ("resting_blood_pressure", "cholesterol")

# Wisconsin Diagnostic Breast Cancer features, in scikit-learn's column order.
_WDBC_BASE = (
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
)
BREAST_FEATURE_KEYS: list[str] = [
    f"{base}_{suffix}" for suffix in ("mean", "se", "worst") for base in _WDBC_BASE
]


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def validate_heart_measurements(
    raw: dict[str, Any] | None,
) -> tuple[dict[str, int | float], dict[str, str]]:
    """Keep only known, in-range heart measurements.

    Returns ``(clean, rejected)`` where ``rejected`` maps a key to the reason it
    was dropped. Values that are None are simply absent, not rejected.
    """
    clean: dict[str, int | float] = {}
    rejected: dict[str, str] = {}
    for key, value in (raw or {}).items():
        if value is None:
            continue
        feature = HEART_FEATURE_BY_KEY.get(key)
        if feature is None:
            rejected[key] = "unknown measurement"
            continue
        number = _as_number(value)
        if number is None:
            rejected[key] = f"not a number: {value!r}"
            continue
        if feature.categorical:
            if number != int(number) or int(number) not in feature.codes:
                allowed = ", ".join(f"{c}={m}" for c, m in feature.codes.items())
                rejected[key] = f"{value!r} is not one of {allowed}"
                continue
            clean[key] = int(number)
            continue
        if not (feature.low <= number <= feature.high):
            rejected[key] = f"{value!r} is outside {feature.low:g}-{feature.high:g}"
            continue
        clean[key] = int(round(number)) if feature.integer else number
    return clean, rejected


def validate_breast_features(
    raw: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Keep only known, non-negative Wisconsin features."""
    clean: dict[str, float] = {}
    rejected: dict[str, str] = {}
    for key, value in (raw or {}).items():
        if value is None:
            continue
        if key not in BREAST_FEATURE_KEYS:
            rejected[key] = "unknown feature"
            continue
        number = _as_number(value)
        if number is None or number < 0:
            rejected[key] = f"not a non-negative number: {value!r}"
            continue
        clean[key] = number
    return clean, rejected
