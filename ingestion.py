"""
Ingestion and file management layer for the MedAgent system.

Provides utilities for:
  - Saving uploaded binary files securely
  - Encoding images to base64 for multimodal LLM processing
  - Building initial LangGraph state from user inputs
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Measurement keys (mirrored from langgraph_multi_agent to avoid circular import)
_USER_MEASUREMENT_KEYS = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_induced_angina", "oldpeak", "slope", "ca", "thal",
]

_BREAST_CANCER_FEATURE_KEYS = [
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


# ── File Management ──────────────────────────────────────────────────────────


def save_uploaded_file(
    file_bytes: bytes,
    original_filename: str,
    upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
) -> str:
    """
    Securely save an uploaded binary file to disk.

    Generates a UUID-prefixed filename to prevent path traversal attacks
    and filename collisions.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    original_filename : str
        Original name from the upload (used only for extension extraction).
    upload_dir : Path or str
        Directory to save into. Created if it does not exist.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = upload_path / safe_name

    save_path.write_bytes(file_bytes)
    logger.info("Saved uploaded file: %s -> %s", original_filename, save_path)
    return str(save_path.resolve())


# ── Base64 Encoding ──────────────────────────────────────────────────────────


def encode_image_to_base64(filepath: str | Path) -> str:
    """
    Read an image file and return its base64-encoded string.

    The result is ready for use with multimodal vision LLMs
    (e.g., ChatOllama with llama3.2-vision).

    Parameters
    ----------
    filepath : str or Path
        Absolute or relative path to the image file.

    Returns
    -------
    str
        Base64-encoded image content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not a supported image type.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    logger.debug("Encoded image to base64: %s (%d bytes)", path, len(encoded))
    return encoded


def get_image_mime_type(filepath: str | Path) -> str:
    """
    Return the MIME type for an image file, suitable for data URIs.

    Parameters
    ----------
    filepath : str or Path
        Path to the image file.

    Returns
    -------
    str
        MIME type string (e.g., "image/jpeg", "image/png").
    """
    ext = Path(filepath).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, mimetypes.guess_type(str(filepath))[0] or "image/jpeg")


# ── State Builder ────────────────────────────────────────────────────────────


def build_initial_state(
    query: str,
    measurements: dict[str, int | float | None] | None = None,
    breast_features: dict[str, float | None] | None = None,
    file_refs: dict[str, str | None] | None = None,
    heart_rate: int | None = None,
    breast_scan_path: str | None = None,
    skin_image_path: str | None = None,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Construct a fully-initialized MedAgentState dict from user inputs.

    Fills defaults for all required fields and populates user-facing aliases.

    Parameters
    ----------
    query : str
        The user's natural language query.
    measurements : dict, optional
        Heart disease measurements. Keys from USER_MEASUREMENT_KEYS.
    breast_features : dict, optional
        30 Wisconsin breast cancer features.
    file_refs : dict, optional
        File references with breast_scan_path and skin_photo_path.
    heart_rate : int, optional
        Heart rate in BPM. Gets merged into measurements.
    breast_scan_path : str, optional
        Path to breast scan image. Overrides file_refs.
    skin_image_path : str, optional
        Path to skin lesion image. Overrides file_refs.
    messages : list, optional
        Pre-existing message history.

    Returns
    -------
    dict
        A complete MedAgentState-compatible dict.
    """
    # Build measurements
    merged_measurements: dict[str, int | float | None] = {
        key: None for key in _USER_MEASUREMENT_KEYS
    }
    if measurements:
        for k, v in measurements.items():
            if k in merged_measurements and v is not None:
                merged_measurements[k] = v

    # Merge heart_rate into max_heart_rate if provided
    if heart_rate is not None:
        merged_measurements["max_heart_rate"] = heart_rate

    # Build breast cancer features
    merged_breast: dict[str, float | None] = {
        key: None for key in _BREAST_CANCER_FEATURE_KEYS
    }
    if breast_features:
        for k, v in breast_features.items():
            if k in merged_breast and v is not None:
                merged_breast[k] = v

    # Build file references
    merged_files: dict[str, str | None] = {
        "breast_scan_path": breast_scan_path,
        "skin_photo_path": skin_image_path,
    }
    if file_refs:
        if merged_files["breast_scan_path"] is None:
            merged_files["breast_scan_path"] = file_refs.get("breast_scan_path")
        if merged_files["skin_photo_path"] is None:
            merged_files["skin_photo_path"] = file_refs.get("skin_photo_path")

    return {
        # Detailed fields
        "raw_user_query": query,
        "user_measurements": merged_measurements,
        "breast_cancer_features": merged_breast,
        "file_references": merged_files,
        "next_agent": "",
        "messages": list(messages or []) + [HumanMessage(content=query)],
        "heart_disease_analysis": {},
        "breast_cancer_analysis": {},
        "skin_disease_analysis": {},
        # User-facing aliases
        "user_query": query,
        "heart_rate": merged_measurements.get("max_heart_rate"),
        "breast_scan_path": merged_files.get("breast_scan_path"),
        "skin_image_path": merged_files.get("skin_photo_path"),
        # LLM explanations (initially empty)
        "heart_diagnosis": None,
        "cancer_diagnosis": None,
        "skin_diagnosis": None,
    }
