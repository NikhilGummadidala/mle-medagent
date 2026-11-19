"""Skin lesion specialist.

A Vision Transformer (ViT-B/16) fine-tuned on HAM10000 dermatoscope images,
published as https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification
(Apache-2.0). It distinguishes seven lesion types and nothing else: it has no
class for rashes, infections, or healthy skin.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "skin_vit"
IMG_SIZE = 224

# Model output index -> (name shown to the patient, needs prompt medical review)
CLASSES: tuple[tuple[str, bool], ...] = (
    ("benign keratosis (seborrheic keratosis / solar lentigo)", False),
    ("basal cell carcinoma", True),
    ("actinic keratosis (pre-cancerous)", True),
    ("vascular lesion", False),
    ("melanocytic nevus (mole)", False),
    ("melanoma", True),
    ("dermatofibroma", False),
)
_EXPECTED_LABELS = (
    "benign_keratosis-like_lesions", "basal_cell_carcinoma", "actinic_keratoses",
    "vascular_lesions", "melanocytic_Nevi", "melanoma", "dermatofibroma",
)

# Measured by scripts/evaluate_skin_model.py; see saved_models/README.md.
EVALUATION_NOTE = (
    "On 248 held-out dermatoscope images it had not seen, it named the right type 86% of "
    "the time and flagged 19 of the 21 lesions needing review."
)

_model = None
_lock = threading.Lock()


class NotASkinPhotoError(ValueError):
    """The image does not look like a colour skin photo, so it was not scored."""


def load_model():
    """Load the ViT classifier from disk. Cached after first call."""
    global _model
    with _lock:
        if _model is not None:
            return _model
        if not (MODEL_DIR / "model.safetensors").exists():
            raise FileNotFoundError(
                "Skin model weights are missing. Run: python scripts/download_models.py"
            )
        from transformers import ViTForImageClassification

        model = ViTForImageClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
        labels = tuple(model.config.id2label[i] for i in range(len(CLASSES)))
        if labels != _EXPECTED_LABELS:
            raise RuntimeError(f"Unexpected skin model label order: {labels}")
        _model = model.eval()
        logger.info("Skin model loaded from %s", MODEL_DIR)
        return _model


def preprocess(image_path: str):
    """Resize to 224x224 and scale pixels to [-1, 1], as the base ViT expects."""
    import torch
    from PIL import Image

    with Image.open(image_path) as handle:
        rgb = handle.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.int16)
        resized = rgb.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    if np.abs(pixels - pixels.mean(axis=2, keepdims=True)).mean() < 2.0:
        raise NotASkinPhotoError("the image is grayscale; a colour photo of the skin is needed")
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def class_probabilities(image_path: str) -> np.ndarray:
    import torch

    model = load_model()
    with torch.inference_mode():
        logits = model(pixel_values=preprocess(image_path)).logits
    return logits.softmax(dim=1)[0].numpy()


def predict(image_path: str) -> dict[str, Any]:
    """Classify a skin lesion photo into one of seven lesion types."""
    try:
        probabilities = class_probabilities(image_path)
    except NotASkinPhotoError as error:
        return {
            "diagnosis": "image not assessed",
            "confidence": None,
            "summary": "The skin photo was not assessed.",
            "details": (
                f"This does not look like a skin photo ({error}). "
                "If it is a mammogram, attach it as a mammogram."
            ),
        }

    order = np.argsort(probabilities)[::-1]
    top = int(order[0])
    name, _ = CLASSES[top]
    concerning = float(sum(p for p, (_, flag) in zip(probabilities, CLASSES) if flag))
    ranked = ", ".join(f"{CLASSES[i][0]} {probabilities[i]:.0%}" for i in order[:3])
    review = concerning >= 0.20

    return {
        "diagnosis": name,
        "confidence": float(probabilities[top]),
        "probability_concerning": concerning,
        "needs_prompt_review": review,
        "summary": (
            f"Closest match: {name} ({probabilities[top]:.0%}). "
            + (
                f"Types that need prompt medical review (melanoma, basal cell carcinoma, "
                f"actinic keratosis) total {concerning:.0%}: please have this looked at soon."
                if review else
                f"Types that need prompt medical review total {concerning:.0%}."
            )
        ),
        "details": (
            f"Top matches: {ranked}. The model only knows seven lesion types and was trained "
            "on dermatoscope close-ups, so it is less reliable on ordinary phone photos and "
            "cannot recognise rashes, infections or healthy skin. " + EVALUATION_NOTE
        ).strip(),
        "probabilities": {CLASSES[i][0]: float(probabilities[i]) for i in range(len(CLASSES))},
    }
