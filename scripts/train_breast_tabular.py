"""Train the tabular breast model on the Wisconsin Diagnostic Breast Cancer data.

    python scripts/train_breast_tabular.py

The data ships with scikit-learn (``sklearn.datasets.load_breast_cancer``,
UCI WDBC, 569 fine-needle-aspirate samples). The saved pipeline contains its
own StandardScaler, so inference takes raw feature values.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.schema import BREAST_FEATURE_KEYS  # noqa: E402

OUT_MODEL = ROOT / "saved_models" / "breast_cancer_tabular.joblib"
OUT_META = ROOT / "saved_models" / "breast_cancer_tabular.metadata.json"
SEED = 42


def expected_sklearn_names() -> list[str]:
    """Our keys in scikit-learn's naming, to prove the column order matches."""
    names = []
    for key in BREAST_FEATURE_KEYS:
        base, suffix = key.rsplit("_", 1)
        base = base.replace("_", " ")
        names.append({"mean": f"mean {base}", "se": f"{base} error", "worst": f"worst {base}"}[suffix])
    return names


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))


def main() -> None:
    data = load_breast_cancer()
    assert list(data.feature_names) == expected_sklearn_names(), "feature order mismatch"
    X = data.data
    # scikit-learn labels malignant as 0; the app's positive class is malignant.
    y = (data.target == 0).astype(int)

    aurocs, matrices = [], []
    for repeat in range(3):
        for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED + repeat).split(X, y):
            model = make_model().fit(X[train], y[train])
            p = model.predict_proba(X[test])[:, 1]
            aurocs.append(roc_auc_score(y[test], p))
            matrices.append(confusion_matrix(y[test], p >= 0.5, labels=[0, 1]))
    tn, fp, fn, tp = np.sum(matrices, axis=0).ravel()
    metrics = {
        "auroc": round(float(np.mean(aurocs)), 4),
        "accuracy_at_0.5": round(float((tp + tn) / (tp + tn + fp + fn)), 4),
        "sensitivity_at_0.5": round(float(tp / (tp + fn)), 4),
        "specificity_at_0.5": round(float(tn / (tn + fp)), 4),
    }

    model = make_model().fit(X, y)
    OUT_MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT_MODEL)
    OUT_META.write_text(json.dumps({
        "model": "StandardScaler + LogisticRegression",
        "sklearn_version": sklearn.__version__,
        "trained_on": date.today().isoformat(),
        "training_script": "scripts/train_breast_tabular.py",
        "data": "sklearn.datasets.load_breast_cancer (UCI Wisconsin Diagnostic Breast Cancer)",
        "n_samples": int(len(y)),
        "n_malignant": int(y.sum()),
        "positive_class": "malignant",
        "feature_order": BREAST_FEATURE_KEYS,
        "evaluation": "5-fold stratified cross-validation repeated 3 times",
        "cross_validation": metrics,
        "weights_sha256": hashlib.sha256(OUT_MODEL.read_bytes()).hexdigest(),
        "limitations": (
            "Inputs are cell-nucleus measurements from a fine-needle aspirate, produced by a "
            "pathology lab, not something a patient can measure. Not clinically validated."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {OUT_MODEL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
