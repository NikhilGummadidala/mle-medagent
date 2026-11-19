"""Breast specialist.

Images: Mammoscreen (https://huggingface.co/ianpan/mammoscreen, Apache-2.0), an
ensemble of three EfficientNetV2-S networks trained on CBIS-DDSM and the RSNA
Screening Mammography data, preceded by the mammo-crop model
(https://huggingface.co/ianpan/mammo-crop, Apache-2.0) that trims background.
The architectures below are re-implemented from those repositories so the
published ``.safetensors`` weights load without executing any downloaded code.

Tabular: a scikit-learn pipeline trained by ``scripts/train_breast_tabular.py``
on the Wisconsin Diagnostic Breast Cancer features.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from models.schema import BREAST_FEATURE_KEYS, validate_breast_features

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MAMMOSCREEN_DIR = MODEL_DIR / "mammoscreen"
MAMMO_CROP_DIR = MODEL_DIR / "mammo_crop"
TABULAR_MODEL_PATH = MODEL_DIR / "breast_cancer_tabular.joblib"
TABULAR_METADATA_PATH = MODEL_DIR / "breast_cancer_tabular.metadata.json"

# Operating point published on the Mammoscreen model card (mean of 3 held-out
# splits of the RSNA screening data): scores at or above this flagged 94.3% of
# cancers while clearing 78.7% of cancer-free breasts. Overall AUROC 0.945.
SCREEN_THRESHOLD = 0.0127
SCREEN_SENSITIVITY = 0.943
SCREEN_SPECIFICITY = 0.787
SCREEN_AUROC = 0.945

DENSITY_LABELS = ("A (almost entirely fatty)", "B (scattered density)",
                  "C (heterogeneously dense)", "D (extremely dense)")

_lock = threading.Lock()
_screen_model = None
_crop_model = None
_tabular_model = None
_tabular_metadata: dict[str, Any] = {}


class NotAMammogramError(ValueError):
    """The image does not look like a mammogram, so it was not scored."""


# ── Architectures (lazy: torch/timm are imported only when an image arrives) ──

def _build_screen_model(config: dict[str, Any]):
    import torch
    import torch.nn as nn
    from timm import create_model

    class MammoNet(nn.Module):
        def __init__(self, image_size, pad_to_aspect_ratio):
            super().__init__()
            self.backbone = create_model(
                config["backbone"], pretrained=False, num_classes=0,
                global_pool="", in_chans=config["in_chans"],
            )
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(p=config["dropout"])
            self.linear = nn.Linear(config["feature_dim"], config["num_classes"])
            self.image_size = tuple(image_size)          # (height, width)
            self.pad_to_aspect_ratio = pad_to_aspect_ratio

        def preprocess(self, image: np.ndarray) -> "torch.Tensor":
            import cv2

            height, width = self.image_size
            if self.pad_to_aspect_ratio:
                # Pad narrow images to the target aspect ratio, then resize.
                h, w = image.shape
                if h / w > height / width:
                    extra = round(h / (height / width)) - w
                    image = np.pad(image, ((0, 0), (extra // 2, extra - extra // 2)))
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                # Fit the longest side, then centre on a black canvas.
                h, w = image.shape
                scale = height / max(h, w)
                image = cv2.resize(
                    image, (max(1, round(w * scale)), max(1, round(h * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )
                h, w = image.shape
                top, left = (height - h) // 2, (width - w) // 2
                image = np.pad(image, ((top, height - h - top), (left, width - w - left)))
            tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
            return (tensor / 255.0 - 0.5) * 2.0          # [0, 255] -> [-1, 1]

        def forward(self, x):
            features = self.pooling(self.backbone(x)).flatten(1)
            logits = self.linear(features)
            return logits[:, 0].sigmoid(), logits[:, 1:].softmax(dim=1)

    class MammoEnsemble(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(config["num_models"]):
                setattr(self, f"net{i}", MammoNet(config["image_sizes"][i], config["pad_to_aspect_ratio"][i]))
            self.num_models = config["num_models"]

    return MammoEnsemble()


def _build_crop_model(config: dict[str, Any]):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from timm import create_model

    class GeM(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * 3)

        def forward(self, x):
            return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.p), 1).pow(1.0 / self.p).flatten(1)

    class MammoCrop(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = create_model(
                config["backbone"], pretrained=False, num_classes=0,
                global_pool="", in_chans=config["in_chans"],
            )
            self.pooling = GeM()
            self.dropout = nn.Dropout(p=config["dropout"])
            self.linear = nn.Linear(config["feature_dim"], config["num_classes"])

        def forward(self, x):
            x = (x / 255.0 - 0.5) * 2.0
            return self.linear(self.pooling(self.backbone(x))).sigmoid()   # normalised x, y, w, h

    return MammoCrop()


def _load_weights(model, directory: Path):
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    return model.eval()


def load_image_models():
    """Load the Mammoscreen ensemble and the crop model. Cached after first call."""
    global _screen_model, _crop_model
    with _lock:
        if _screen_model is not None and _crop_model is not None:
            return _screen_model, _crop_model
        for directory in (MAMMOSCREEN_DIR, MAMMO_CROP_DIR):
            if not (directory / "model.safetensors").exists():
                raise FileNotFoundError(
                    "Mammogram model weights are missing. Run: python scripts/download_models.py"
                )
        screen_config = json.loads((MAMMOSCREEN_DIR / "config.json").read_text(encoding="utf-8"))
        crop_config = json.loads((MAMMO_CROP_DIR / "config.json").read_text(encoding="utf-8"))
        _crop_model = _load_weights(_build_crop_model(crop_config), MAMMO_CROP_DIR)
        _screen_model = _load_weights(_build_screen_model(screen_config), MAMMOSCREEN_DIR)
        logger.info("Mammoscreen and mammo-crop loaded")
        return _screen_model, _crop_model


def load_tabular_model():
    """Load the Wisconsin-feature pipeline. Cached after first call."""
    global _tabular_model, _tabular_metadata
    with _lock:
        if _tabular_model is not None:
            return _tabular_model
        if not TABULAR_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Breast tabular model file is missing. Run: python scripts/train_breast_tabular.py"
            )
        import joblib

        _tabular_model = joblib.load(TABULAR_MODEL_PATH)
        if TABULAR_METADATA_PATH.exists():
            _tabular_metadata = json.loads(TABULAR_METADATA_PATH.read_text(encoding="utf-8"))
        return _tabular_model


# ── Image handling ────────────────────────────────────────────────────────────

def load_mammogram(image_path: str) -> np.ndarray:
    """Read an image as 8-bit grayscale, refusing inputs that are not mammograms.

    Mammograms are grayscale with a large dark background. A colour photo (a
    selfie, a skin picture) would still get a score from the network, so it is
    rejected here instead.
    """
    from PIL import Image

    with Image.open(image_path) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.int16)
    spread = np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean()
    if spread > 4.0:
        raise NotAMammogramError("the image is in colour; mammograms are grayscale")
    gray = rgb.mean(axis=2).astype(np.uint8)
    if min(gray.shape) < 256:
        raise NotAMammogramError("the image is too small to assess (under 256 pixels)")
    # Real mammograms measured at 10-90% near-black background; 3% leaves room
    # for tightly cropped scans while still rejecting evenly lit grayscale photos.
    if (gray < 40).mean() < 0.03:
        raise NotAMammogramError("the image has no dark background around the breast")
    return gray


def _crop(gray: np.ndarray, crop_model) -> np.ndarray:
    import cv2
    import torch

    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
    with torch.inference_mode():
        x, y, w, h = crop_model(torch.from_numpy(small).float()[None, None])[0].tolist()
    height, width = gray.shape
    left, top = int(x * width), int(y * height)
    right, bottom = int((x + w) * width), int((y + h) * height)
    cropped = gray[max(top, 0):min(bottom, height), max(left, 0):min(right, width)]
    # A degenerate box means the crop model failed; fall back to the full image.
    if cropped.shape[0] < height * 0.2 or cropped.shape[1] < width * 0.1:
        return gray
    return cropped


def score_mammogram(image_path: str) -> dict[str, Any]:
    """Return the raw ensemble outputs for one mammogram view."""
    import torch

    gray = load_mammogram(image_path)
    screen_model, crop_model = load_image_models()
    cropped = _crop(gray, crop_model)
    cancer_scores, densities = [], []
    with torch.inference_mode():
        for i in range(screen_model.num_models):
            net = getattr(screen_model, f"net{i}")
            cancer, density = net(net.preprocess(cropped)[None, None])
            cancer_scores.append(float(cancer[0]))
            densities.append(density[0].numpy())
    return {
        "score": float(np.mean(cancer_scores)),
        "density": DENSITY_LABELS[int(np.mean(densities, axis=0).argmax())],
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def _predict_image(image_path: str) -> dict[str, Any]:
    try:
        result = score_mammogram(image_path)
    except NotAMammogramError as error:
        return {
            "diagnosis": "image not assessed",
            "confidence": None,
            "summary": "The breast image was not assessed.",
            "details": (
                f"This does not look like a mammogram ({error}). The breast model only reads "
                "X-ray mammogram images. If this is a skin photo, attach it as a skin photo."
            ),
        }
    flagged = result["score"] >= SCREEN_THRESHOLD
    return {
        "diagnosis": (
            "flagged for follow-up" if flagged else "not flagged by the screening model"
        ),
        "confidence": None,
        "screening_score": result["score"],
        "summary": (
            "The screening model flagged this mammogram as needing a radiologist's review."
            if flagged else
            "The screening model did not flag this mammogram."
        ),
        "details": (
            f"Mammoscreen ensemble, single view. At this cut-off the published evaluation "
            f"caught {SCREEN_SENSITIVITY:.0%} of cancers and cleared {SCREEN_SPECIFICITY:.0%} "
            f"of cancer-free breasts (AUROC {SCREEN_AUROC}), using two views per breast; a "
            f"single view is less accurate. A flag is not a cancer diagnosis and most flags "
            f"are false alarms; no flag does not rule cancer out. "
            f"Estimated breast density: {result['density']}."
        ),
    }


def _predict_tabular(features: dict[str, Any]) -> dict[str, Any]:
    clean, rejected = validate_breast_features(features)
    missing = [k for k in BREAST_FEATURE_KEYS if k not in clean]
    if missing:
        return {
            "diagnosis": "insufficient data",
            "confidence": None,
            "summary": "No breast assessment was run: the lab measurements are incomplete.",
            "details": (
                f"All {len(BREAST_FEATURE_KEYS)} fine-needle-aspirate measurements are required; "
                f"{len(missing)} are missing or invalid (for example {', '.join(missing[:3])}). "
                "These come from a pathology report. Alternatively attach a mammogram image."
            ),
        }
    model = load_tabular_model()
    row = np.array([[clean[k] for k in BREAST_FEATURE_KEYS]], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])
    malignant = probability >= 0.5
    cv = _tabular_metadata.get("cross_validation", {})
    quality = (
        f" Cross-validated accuracy {cv['accuracy_at_0.5']:.1%} (AUROC {cv['auroc']:.3f})."
        if cv else ""
    )
    return {
        "diagnosis": "malignant pattern" if malignant else "benign pattern",
        "confidence": probability if malignant else 1.0 - probability,
        "probability_malignant": probability,
        "summary": f"Model estimate that the sample is malignant: {probability:.0%}.",
        "details": (
            "Logistic regression on 30 fine-needle-aspirate measurements "
            f"(Wisconsin Diagnostic Breast Cancer data).{quality}"
        ),
    }


def predict(
    image_path: str | None = None,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess a mammogram image if given, otherwise the 30 tabular features."""
    if image_path is not None:
        return _predict_image(image_path)
    if features and any(v is not None for v in features.values()):
        return _predict_tabular(features)
    return {
        "diagnosis": "no input provided",
        "confidence": None,
        "summary": "No breast assessment was run.",
        "details": "No mammogram image or lab measurements were provided.",
    }
