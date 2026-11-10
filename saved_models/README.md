# Trained specialist weights

`breast_cancer_cnn.pt` is a new CBIS-DDSM research checkpoint trained locally on
2026-11-02 using the existing `BreastCancerCNN` architecture. It is committed as
an ordinary Git file (about 1.6 MB); Git LFS and retraining are not required.

## Use with MedAgent

1. Pull this branch, or pull main after the PR is merged.
2. Install the normal runtime dependencies with `pip install -r requirements.txt`.
3. Restart the backend: `uvicorn api.main:app --host 127.0.0.1 --port 8000`.
4. Upload a mammogram named, for example, `mammogram_example.png`, then submit the
   patient's prompt through the existing MediChat interface.

The existing loader in `models/breast_cancer.py` automatically reads
`saved_models/breast_cancer_cnn.pt`. It does not need a new API or a cloud model.
Restarting clears any older checkpoint cached by a running backend.
The frontend now routes using the original attachment name; uploaded storage
paths are UUIDs and cannot identify the intended specialist. Attachment names
containing `mamm` or `breast` select the breast image path.

The Ollama models required by the existing agent remain `llama3.2` and
`llama3.2-vision`. These are separate from the trained CNN. Runtime support for
`llama3.2-vision` must be verified; the local integration was tested with Ollama
0.24.0 after 0.40.13 rejected the `mllama` architecture. The upstream discussion is
https://github.com/ollama/ollama/issues/16490 . No Ollama binary is included here.

## Integrity and measured results

SHA256 of `breast_cancer_cnn.pt`:

```
90bc418dea7e68c16f4c1ab13ea633597609870ac457345ad62f564eef453284
```

Check it on macOS with `shasum -a 256 saved_models/breast_cancer_cnn.pt`.
The adjacent metadata JSON records the model's provenance and metrics.

- 2,789 full CBIS-DDSM mammograms from 1,456 patients.
- Patient-disjoint internal partitions: 1,920 training / 451 validation / 418 test images.
- 49 completed epochs; epoch 39 selected using minimum validation loss.
- Internal held-out accuracy: 0.6340; AUROC: 0.6962.
- 13 exact-pixel groups with conflicting labels excluded; other exact duplicate
  inputs and unavailable metadata/image matches are logged by the training script.
- No MIAS conversation-test images were used to train or select this checkpoint.

Confidence is `max(p_malignant, 1 - p_malignant)`, an **uncalibrated class score**.
It is not measured accuracy or calibrated patient cancer risk. These internal
CBIS results are not the official CBIS benchmark and do not establish performance
on MIAS. The checkpoint and the LLM explanations have not been clinically validated.
This is a new training run, not the teammate's earlier checkpoint.

## Training source

The original CNN recipe is in `diag-agents/BreastCancerAgent.ipynb`; the local
adaptation is `scripts/train_breast_cnn.py`. The architecture, augmentation,
optimizer, learning rate, batch size and stopping policy follow the CNN section.
The local adaptation joins pathology tables to full-image metadata, splits by
underlying patient identity, excludes contradictory exact duplicates, rejects
invalid images, and aligns resizing with the existing inference preprocessing.

The notebook's original Kaggle slug omitted `image`. The version used here is:
https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
(version 1, public JPG conversion; listed license CC BY-SA 3.0).

Original dataset citation: Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D.
(2016). *Curated Breast Imaging Subset of Digital Database for Screening Mammography
(CBIS-DDSM)*. The Cancer Imaging Archive.
https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

Training dependencies are optional for inference:
`pip install -r requirements-training.txt`. From the repository root, the training
functions can be called as follows (the shipped checkpoint is not overwritten):

```python
import kagglehub
from scripts.train_breast_cnn import prepare, train

source = kagglehub.dataset_download(
    "awsaf49/cbis-ddsm-breast-cancer-image-dataset/versions/1"
)
run = "research_training/cbis_cnn_reproduction"
prepare(source, run)
train(run, max_epochs=50)
# Newly trained state_dict: research_training/cbis_cnn_reproduction/best_weights.pt
```

Mac MPS is used when available; fixed seeds do not guarantee bitwise-identical
GPU training. Source images, patient manifests, transcripts, screenshots and local
account data are excluded from this change. The breast tabular MLP and skin
weights are not supplied by this checkpoint.
