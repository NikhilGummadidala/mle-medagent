"""
Model loading and inference for the three specialist agents.

Each module exposes:
  - load_model()  → loads the trained model from disk (cached singleton)
  - predict(...)  → runs inference and returns a structured dict
"""

from models.heart_disease import load_model as load_heart_model
from models.heart_disease import predict as predict_heart
from models.breast_cancer import load_cnn_model as load_breast_cnn
from models.breast_cancer import load_mlp_model as load_breast_mlp
from models.breast_cancer import predict as predict_breast
from models.skin_disease import load_model as load_skin_model
from models.skin_disease import predict as predict_skin

__all__ = [
    "load_heart_model", "predict_heart",
    "load_breast_cnn", "load_breast_mlp", "predict_breast",
    "load_skin_model", "predict_skin",
]
