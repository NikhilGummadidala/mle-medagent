"""Upload storage for patient images.

Uploads are verified as real images, stored under a per-user directory with a
random name, addressed by an opaque id (never a server path), and deleted
after ``RETENTION_HOURS``.
"""
from __future__ import annotations

import io
import logging
import re
import time
import uuid
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 50_000_000
RETENTION_HOURS = 24

# Pillow format name -> stored extension
_FORMATS = {"JPEG": "jpg", "PNG": "png", "BMP": "bmp", "TIFF": "tiff", "WEBP": "webp"}
_FILE_ID = re.compile(r"^[0-9a-f]{32}\.(?:jpg|png|bmp|tiff|webp)$")


class InvalidUpload(ValueError):
    """The uploaded bytes are not an acceptable image."""


def _user_dir(user_id: int, upload_dir: Path | None = None) -> Path:
    return (upload_dir or UPLOAD_DIR) / str(int(user_id))


def save_upload(data: bytes, user_id: int, upload_dir: Path | None = None) -> str:
    """Validate and store an uploaded image. Returns its opaque file id."""
    if not data:
        raise InvalidUpload("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidUpload(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except Exception as error:  # Pillow raises many types for corrupt input
        raise InvalidUpload("The file is not a readable image.") from error
    if image_format not in _FORMATS:
        raise InvalidUpload("Unsupported image type. Use JPEG, PNG, BMP, TIFF or WebP.")
    if width * height > MAX_PIXELS:
        raise InvalidUpload("The image dimensions are too large.")

    directory = _user_dir(user_id, upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    file_id = f"{uuid.uuid4().hex}.{_FORMATS[image_format]}"
    (directory / file_id).write_bytes(data)
    logger.info("Stored upload %s for user %s", file_id, user_id)
    return file_id


def resolve_upload(file_id: str | None, user_id: int, upload_dir: Path | None = None) -> str | None:
    """Path of a file this user uploaded, or None if the id is not theirs/valid."""
    if not file_id or not _FILE_ID.fullmatch(file_id):
        return None
    path = _user_dir(user_id, upload_dir) / file_id
    return str(path) if path.is_file() else None


def purge_old_uploads(upload_dir: Path | None = None, max_age_hours: float = RETENTION_HOURS) -> int:
    """Delete uploads older than the retention period. Returns how many."""
    root = upload_dir or UPLOAD_DIR
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for path in root.glob("*/*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            logger.warning("Could not remove expired upload %s", path)
    return removed
