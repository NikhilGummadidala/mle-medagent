# Models

Four models back the three specialists. Two are small and committed here; two
are third-party open-source weights fetched by `scripts/download_models.py`
(too large for an ordinary Git file) and verified by SHA-256.

| Specialist | Model | Source | Licence | In Git |
| --- | --- | --- | --- | --- |
| Heart | Gradient-boosted trees, `heart_disease_hgb.joblib` | trained here by `scripts/train_heart_model.py` on UCI Cleveland | data: CC BY 4.0 | yes |
| Breast (image) | Mammoscreen, 3x EfficientNetV2-S + mammo-crop | [ianpan/mammoscreen](https://huggingface.co/ianpan/mammoscreen), [ianpan/mammo-crop](https://huggingface.co/ianpan/mammo-crop) | Apache-2.0 | no (250 MB) |
| Breast (lab values) | Logistic regression, `breast_cancer_tabular.joblib` | trained here by `scripts/train_breast_tabular.py` on Wisconsin Diagnostic Breast Cancer | data: CC BY 4.0 | yes |
| Skin | ViT-B/16 fine-tuned on HAM10000 | [Anwarkh1/Skin_Cancer-Image_Classification](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification) | Apache-2.0 | no (343 MB) |

The downloads are pinned to exact repository revisions. Only `.safetensors`
weights and JSON configs are fetched; the network architectures are implemented
in `models/`, so no downloaded code is executed.

None of these models is clinically validated. This is a prototype.

## Heart

- 303 patients, target is angiographic coronary disease. Encoding is the
  original UCI one, defined once in `models/schema.py`.
- Missing measurements are passed as missing. The model is trained with random
  feature masking so it copes with partial input, and it is not run at all
  without age, sex, chest pain type and at least four measurements in total.
- 5-fold cross-validation repeated 3 times (`heart_disease_hgb.metadata.json`):

| Measurements available | AUROC | Accuracy |
| --- | --- | --- |
| All 13 | 0.895 | 81% |
| All except the two invasive tests (ca, thal) | 0.855 | 77% |
| Age, sex, chest pain type, blood pressure | 0.817 | 74% |
| Age, sex, blood pressure only (not allowed in the app) | 0.702 | 64% |

- Limitation: every patient had been referred for heart testing and almost half
  had disease, so the output is not a general-population risk.

This replaces `heart_disease_rf.pkl`, which had been trained on a different
category encoding from the one the app sent it, and whose training notebook did
not produce it.

## Breast, mammogram image

- Published evaluation (model card, mean of three held-out splits of the RSNA
  screening data, two views per breast): AUROC 0.945. At the cut-off the app
  uses (score >= 0.0127): sensitivity 94.3%, specificity 78.7%.
- The app reports a threshold decision ("flagged for follow-up" or "not
  flagged"), not a percentage, because the raw score is not a calibrated
  probability.
- The app sends one view; the card states single-view performance is lower and
  does not quantify it.
- Local check on 65 MIAS film mammograms that neither model was trained on,
  abnormal versus normal: Mammoscreen AUROC 0.65, the previous in-house CNN
  0.51. Mammoscreen flagged 9 of 26 abnormal and 2 of 39 normal images; the old
  CNN called 8 of 26 abnormal and 12 of 39 normal "malignant". This is a small
  sanity check on low-resolution digitised film, and most MIAS abnormalities are
  benign, so it does not test the published cancer-detection figure.
- Inputs that are in colour, tiny, or lack a dark background are refused rather
  than scored.

This replaces `breast_cancer_cnn.pt` (held-out AUROC 0.70, accuracy 63%).

## Breast, lab values

- 569 fine-needle-aspirate samples, 30 features, scaler included in the saved
  pipeline. Cross-validated AUROC 0.996, accuracy 97.5%, sensitivity 95.3%,
  specificity 98.9%.
- Requires all 30 values, which come from a pathology report.

## Skin

- Seven classes: melanoma, basal cell carcinoma, actinic keratosis, melanocytic
  nevus, benign keratosis, dermatofibroma, vascular lesion. It has no class for
  rashes, infections or healthy skin.
- The model card reports 96.95% validation accuracy. That figure is inflated:
  1,025 of the 1,285 images in the dataset's test split are also in its training
  split. Measured here with `scripts/evaluate_skin_model.py` on the 248 test and
  validation images whose lesion does not appear in the training split:

| Metric | Value |
| --- | --- |
| Accuracy | 85.9% |
| Balanced accuracy | 73.4% |
| Melanoma recall | 83% |
| Basal cell carcinoma recall | 30% |
| Benign keratosis recall | 24% |
| "Needs review" (melanoma, BCC, actinic keratosis) vs the rest | AUROC 0.94; at the app's cut-off 19 of 21 flagged, 90% specificity |

- Only 21 of those 248 images are in the "needs review" group, so those numbers
  are rough.
- Trained on dermatoscope images. Ordinary phone photos are out of distribution
  and accuracy on them is unmeasured.

## Reproducing

```sh
pip install -r requirements-test.txt
python scripts/download_models.py          # fetch and verify third-party weights
python scripts/train_heart_model.py        # rewrites heart_disease_hgb.*
python scripts/train_breast_tabular.py     # rewrites breast_cancer_tabular.*
```

The `.joblib` files are tied to the scikit-learn version pinned in
`requirements.txt`; retrain them when that pin changes.
