"""Train the heart-disease model from the committed UCI Cleveland data.

    python scripts/train_heart_model.py

Writes ``saved_models/heart_disease_hgb.joblib`` and a metadata JSON holding
cross-validated metrics. The model is a gradient-boosted tree ensemble that
accepts missing values natively. It is trained with random feature masking so
its estimate stays sensible when a patient supplies only some measurements,
instead of imputing invented "default" values.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.schema import HEART_FEATURES, HEART_REQUIRED_KEYS  # noqa: E402

DATA = ROOT / "diag-agents" / "data" / "heart.csv"
OUT_MODEL = ROOT / "saved_models" / "heart_disease_hgb.joblib"
OUT_META = ROOT / "saved_models" / "heart_disease_hgb.metadata.json"
SEED = 42
MASK_COPIES = 8

KEYS = [f.key for f in HEART_FEATURES]
ALWAYS_KEPT = [KEYS.index(k) for k in ("age", "sex")]

# Input scenarios evaluated in cross-validation: which measurements the patient gave.
SCENARIOS = {
    "all_13_features": KEYS,
    "age_sex_bp_only": ["age", "sex", "resting_blood_pressure"],
    "minimum_required_plus_bp": [*HEART_REQUIRED_KEYS, "resting_blood_pressure"],
    "no_invasive_tests": [k for k in KEYS if k not in ("ca", "thal")],
}


def load() -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(DATA, na_values="?")
    X = table[[f.column for f in HEART_FEATURES]].to_numpy(dtype=float)
    y = (table["num"] > 0).astype(int).to_numpy()
    return X, y


def mask_augment(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Append copies of the data with random features hidden (set to NaN)."""
    parts_x, parts_y = [X], [y]
    for _ in range(MASK_COPIES):
        masked = X.copy()
        rate = rng.uniform(0.1, 0.9, size=(len(X), 1))
        hide = rng.random(X.shape) < rate
        hide[:, ALWAYS_KEPT] = False
        masked[hide] = np.nan
        parts_x.append(masked)
        parts_y.append(y)
    return np.vstack(parts_x), np.concatenate(parts_y)


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED
    )


def cross_validate(X: np.ndarray, y: np.ndarray) -> dict:
    scores: dict[str, dict[str, list]] = {name: {"auroc": [], "brier": [], "cm": []} for name in SCENARIOS}
    for repeat in range(3):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(folds.split(X, y)):
            rng = np.random.default_rng(SEED + 100 * repeat + fold)
            model = make_model().fit(*mask_augment(X[train], y[train], rng))
            for name, available in SCENARIOS.items():
                view = X[test].copy()
                view[:, [i for i, k in enumerate(KEYS) if k not in available]] = np.nan
                p = model.predict_proba(view)[:, 1]
                scores[name]["auroc"].append(roc_auc_score(y[test], p))
                scores[name]["brier"].append(brier_score_loss(y[test], p))
                scores[name]["cm"].append(confusion_matrix(y[test], p >= 0.5, labels=[0, 1]))
    summary = {}
    for name, s in scores.items():
        tn, fp, fn, tp = np.sum(s["cm"], axis=0).ravel()
        summary[name] = {
            "features": SCENARIOS[name],
            "auroc": round(float(np.mean(s["auroc"])), 3),
            "auroc_std": round(float(np.std(s["auroc"])), 3),
            "brier": round(float(np.mean(s["brier"])), 3),
            "accuracy_at_0.5": round(float((tp + tn) / (tp + tn + fp + fn)), 3),
            "sensitivity_at_0.5": round(float(tp / (tp + fn)), 3),
            "specificity_at_0.5": round(float(tn / (tn + fp)), 3),
        }
    return summary


def main() -> None:
    X, y = load()
    metrics = cross_validate(X, y)
    model = make_model().fit(*mask_augment(X, y, np.random.default_rng(SEED)))
    OUT_MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT_MODEL)
    metadata = {
        "model": "sklearn.ensemble.HistGradientBoostingClassifier",
        "sklearn_version": sklearn.__version__,
        "trained_on": date.today().isoformat(),
        "training_script": "scripts/train_heart_model.py",
        "data": "diag-agents/data/heart.csv (UCI Cleveland heart disease, processed)",
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "n_patients": int(len(y)),
        "n_with_disease": int(y.sum()),
        "target": "angiographic coronary disease present (num > 0)",
        "feature_order": KEYS,
        "encoding": {f.key: f.codes for f in HEART_FEATURES if f.categorical},
        "missing_values": "NaN at inference; trained with random feature masking, no default imputation",
        "evaluation": "5-fold stratified cross-validation repeated 3 times, by input scenario",
        "cross_validation": metrics,
        "weights_sha256": hashlib.sha256(OUT_MODEL.read_bytes()).hexdigest(),
        "limitations": (
            "303 patients referred for coronary angiography at one US clinic in the 1980s; "
            "nearly half had disease, so the output is not a general-population risk. "
            "Not clinically validated."
        ),
    }
    OUT_META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {OUT_MODEL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
