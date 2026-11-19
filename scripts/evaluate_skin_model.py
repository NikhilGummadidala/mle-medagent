"""Measure the skin model on the HAM10000 test split it was not trained on.

    # test split of https://huggingface.co/datasets/marmal88/skin_cancer (about 350 MB)
    curl -L -o test.parquet https://huggingface.co/api/datasets/marmal88/skin_cancer/parquet/default/test/0.parquet
    python scripts/evaluate_skin_model.py test.parquet [more.parquet ...] [--exclude-lesions ids.txt]

``--exclude-lesions`` takes a file of lesion ids (one per line) to drop, used to
remove test lesions that also appear in the training split.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import skin_disease  # noqa: E402

DX_TO_INDEX = {label: i for i, label in enumerate(skin_disease._EXPECTED_LABELS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="+")
    parser.add_argument("--exclude-lesions")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    table = pd.concat([pd.read_parquet(p) for p in args.parquet]).drop_duplicates("image_id")
    if args.exclude_lesions:
        seen = set(Path(args.exclude_lesions).read_text().split())
        before = len(table)
        table = table[~table.lesion_id.isin(seen)]
        print(f"Dropped {before - len(table)} test images whose lesion is in the training data")
    if args.limit:
        table = table.sample(n=min(args.limit, len(table)), random_state=0)
    unknown = set(table.dx) - set(DX_TO_INDEX)
    assert not unknown, f"Unmapped labels: {unknown}"

    from PIL import Image

    truth, probabilities = [], []
    with tempfile.TemporaryDirectory() as scratch:
        for n, row in enumerate(table.itertuples()):
            path = Path(scratch) / "image.png"
            Image.open(io.BytesIO(row.image["bytes"])).convert("RGB").save(path)
            probabilities.append(skin_disease.class_probabilities(str(path)))
            truth.append(DX_TO_INDEX[row.dx])
            if n % 200 == 0:
                print(f"  {n}/{len(table)}", flush=True)

    y = np.array(truth)
    p = np.array(probabilities)
    predicted = p.argmax(axis=1)
    concerning_classes = [i for i, (_, flag) in enumerate(skin_disease.CLASSES) if flag]
    y_concerning = np.isin(y, concerning_classes)
    p_concerning = p[:, concerning_classes].sum(axis=1)
    flagged = p_concerning >= 0.20
    recall = confusion_matrix(y, predicted, labels=range(7), normalize="true").diagonal()
    report = {
        "n_images": int(len(y)),
        "accuracy": round(float((predicted == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predicted)), 4),
        "recall_by_class": {
            skin_disease.CLASSES[i][0]: round(float(recall[i]), 3) for i in range(7)
        },
        "needs_review_vs_benign": {
            "n_needing_review": int(y_concerning.sum()),
            "auroc": round(float(roc_auc_score(y_concerning, p_concerning)), 4),
            "sensitivity_at_0.20": round(float(flagged[y_concerning].mean()), 4),
            "specificity_at_0.20": round(float((~flagged[~y_concerning]).mean()), 4),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
