"""
Skin Disease specialist — loads a trained MobileNetV2 (TensorFlow/Keras) from disk
and classifies skin lesion images into diagnostic categories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_PATH = MODEL_DIR / "skin_disease_mobilenetv2.keras"
LABELS_PATH = MODEL_DIR / "skin_label_classes.npy"
IMG_SIZE = 224

_model = None
_label_classes = None


def load_model() -> tuple[Any, np.ndarray]:
    """
    Load the Skin Disease MobileNetV2 and label classes from disk.
    Returns (model, label_classes). Cached after first call.
    """
    global _model, _label_classes
    if _model is not None and _label_classes is not None:
        return _model, _label_classes

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Skin disease model not found at {MODEL_PATH}. "
            "Run the skin-disease notebook and export the model first."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Label classes not found at {LABELS_PATH}. "
            "Run the skin-disease notebook and export the label classes first."
        )

    # Lazy import to avoid loading TensorFlow at module-import time
    import tensorflow as tf
    _model = tf.keras.models.load_model(MODEL_PATH)
    _label_classes = np.load(LABELS_PATH, allow_pickle=True)

    logger.info("Skin disease model loaded from %s", MODEL_PATH)
    return _model, _label_classes


def _preprocess_image(image_path: str) -> np.ndarray:
    """Load an image and prepare it for MobileNetV2 inference."""
    import tensorflow as tf

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    # MobileNetV2 expects pixels in [-1, 1]
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img.numpy()


def predict(image_path: str) -> dict[str, Any]:
    """
    Classify a skin lesion image.

    Parameters
    ----------
    image_path : str
        Path to the skin lesion image file.

    Returns
    -------
    dict with keys: diagnosis, confidence, probabilities, details
    """
    model, label_classes = load_model()

    img_array = _preprocess_image(image_path)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])
    predicted_label = label_classes[predicted_idx]

    return {
        "diagnosis": str(predicted_label),
        "confidence": confidence,
        "probabilities": {
            str(label_classes[i]): float(predictions[i])
            for i in range(len(label_classes))
        },
        "details": f"MobileNetV2 classification on {image_path}.",
    }
