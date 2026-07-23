"""
Breast Cancer specialist — dual-model architecture.

- Primary:   CNN (BreastCancerCNN) — classifies breast scan images.
- Fallback:  MLP (BreastCancerMLP) — classifies from 30 tabular features
             (Wisconsin Breast Cancer dataset) when no image is available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"
CNN_MODEL_PATH = MODEL_DIR / "breast_cancer_cnn.pt"
MLP_MODEL_PATH = MODEL_DIR / "breast_cancer_mlp.pt"
IMG_SIZE = 224

# Wisconsin Breast Cancer feature keys (30 features, must match BREAST_CANCER_FEATURE_KEYS)
FEATURE_KEYS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


# ── CNN Architecture (image input) ───────────────────────────────────────────

class BreastCancerCNN(nn.Module):
    """
    CNN for breast cancer classification (benign vs malignant).
    Input: RGB breast scan image (224×224).
    Output: single sigmoid probability in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── MLP Architecture (tabular input) ─────────────────────────────────────────

class BreastCancerMLP(nn.Module):
    """
    Multi-layer perceptron for breast cancer classification.
    Input:  30 normalised tumour features (Wisconsin dataset).
    Output: single sigmoid probability in [0, 1].
    """

    def __init__(self, input_dim: int = 30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Device ────────────────────────────────────────────────────────────────────

_device = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


# ── Model Loaders (cached singletons) ────────────────────────────────────────

_cnn_model = None
_mlp_model = None


def load_cnn_model() -> BreastCancerCNN:
    """Load the Breast Cancer CNN from disk. Cached after first call."""
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found at {CNN_MODEL_PATH}. "
            "Run the BreastCancerAgent notebook and export the CNN model first."
        )

    device = _get_device()
    model = BreastCancerCNN().to(device)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    _cnn_model = model
    logger.info("Breast cancer CNN loaded from %s", CNN_MODEL_PATH)
    return _cnn_model


def load_mlp_model() -> BreastCancerMLP:
    """Load the Breast Cancer MLP from disk. Cached after first call."""
    global _mlp_model
    if _mlp_model is not None:
        return _mlp_model

    if not MLP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MLP model not found at {MLP_MODEL_PATH}. "
            "Run the BreastCancerAgent notebook and export the MLP model first."
        )

    device = _get_device()
    model = BreastCancerMLP().to(device)
    model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    _mlp_model = model
    logger.info("Breast cancer MLP loaded from %s", MLP_MODEL_PATH)
    return _mlp_model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_image(image_path: str) -> torch.Tensor:
    """Load an image from disk and prepare it for the CNN."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # HWC → CHW, add batch dim
    tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _preprocess_features(features: dict[str, float | None]) -> torch.Tensor:
    """Convert a dict of Wisconsin features into a normalised tensor."""
    values = []
    for key in FEATURE_KEYS:
        val = features.get(key)
        if val is None:
            raise ValueError(
                f"Missing required breast cancer feature: {key}. "
                "All 30 Wisconsin features must be provided for tabular prediction."
            )
        values.append(float(val))

    arr = np.array(values, dtype=np.float32).reshape(1, -1)
    return torch.tensor(arr, dtype=torch.float32)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    image_path: str | None = None,
    features: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """
    Classify breast cancer risk.

    Parameters
    ----------
    image_path : str, optional
        Path to a breast scan image. Used by the CNN (primary model).
    features : dict, optional
        30 Wisconsin breast cancer features. Used by the MLP (fallback).

    Returns
    -------
    dict with keys: diagnosis, confidence, probabilities, details
    """
    device = _get_device()

    if image_path is not None:
        model = load_cnn_model()
        tensor = _preprocess_image(image_path).to(device)
        source = f"CNN classification on {image_path}"
    elif features is not None:
        model = load_mlp_model()
        tensor = _preprocess_features(features).to(device)
        source = "MLP classification on 30 tabular features"
    else:
        return {
            "diagnosis": "no input provided",
            "confidence": 0.0,
            "probabilities": {"benign": 0.0, "malignant": 0.0},
            "details": "No breast scan image or tabular features were provided.",
        }

    with torch.no_grad():
        prob = model(tensor).item()

    label = "malignant" if prob >= 0.5 else "benign"
    confidence = prob if prob >= 0.5 else 1.0 - prob

    return {
        "diagnosis": label,
        "confidence": confidence,
        "probabilities": {
            "benign": 1.0 - prob,
            "malignant": prob,
        },
        "details": f"{source}.",
    }
