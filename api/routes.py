"""
FastAPI route definitions for the MedAgent API.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas import (
    ChatResponse,
    DiagnosisResult,
    HistoryEntry,
    HistoryResponse,
    UploadResponse,
)
from api.graph_runner import create_conversation, get_history, run_graph
from ingestion import save_uploaded_file

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _validate_image(filename: str) -> None:
    """Raise if the file extension is not a supported image type."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    conversation_id: str = Form(...),
    heart_rate: int | None = Form(default=None),
    breast_scan_path: str | None = Form(default=None),
    skin_photo_path: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> ChatResponse:
    """
    Send a message to the MedAgent system.

    Supports both multipart/form-data (with optional file attachment)
    and JSON payloads.
    """
    # Handle optional file attachment
    if file and file.filename:
        _validate_image(file.filename)
        file_bytes = await file.read()
        saved_path = save_uploaded_file(file_bytes, file.filename, UPLOAD_DIR)
        # Route the uploaded image based on filename heuristic or existing paths
        if breast_scan_path is None and skin_photo_path is None:
            # Auto-detect: if filename suggests breast, set breast path
            fname_lower = file.filename.lower()
            if any(kw in fname_lower for kw in ("breast", "mamm", "chest")):
                breast_scan_path = saved_path
            else:
                skin_photo_path = saved_path

    try:
        result = run_graph(
            conversation_id=conversation_id,
            message=message,
            heart_rate=heart_rate,
            breast_scan_path=breast_scan_path,
            skin_photo_path=skin_photo_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {e}")

    def _to_diagnosis(data: dict | None) -> DiagnosisResult | None:
        if not data or not data.get("diagnosis"):
            return None
        return DiagnosisResult(
            diagnosis=data["diagnosis"],
            confidence=data.get("confidence", 0.0),
            details=data.get("details", ""),
        )

    # Build agent-by-agent messages from the result
    agent_messages = result.get("messages", [])
    history_entries = [
        HistoryEntry(role=m["role"], content=m["content"])
        for m in agent_messages
        if m["role"] == "assistant"
    ]

    return ChatResponse(
        conversation_id=conversation_id,
        reply=result["reply"],
        heart_disease=_to_diagnosis(result.get("heart_disease")),
        breast_cancer=_to_diagnosis(result.get("breast_cancer")),
        skin_disease=_to_diagnosis(result.get("skin_disease")),
        routed_to=result["routed_to"],
        messages=history_entries,
        disclaimer=result.get("disclaimer", ""),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload an image file (breast scan or skin photo).
    Returns the saved file path to include in subsequent /chat requests.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    _validate_image(file.filename)
    file_bytes = await file.read()
    saved_path = save_uploaded_file(file_bytes, file.filename, UPLOAD_DIR)

    ext = Path(saved_path).suffix.lower()
    return UploadResponse(
        file_path=saved_path,
        file_type=ext.lstrip("."),
    )


@router.post("/conversations", response_model=dict)
def new_conversation() -> dict:
    """Create a new conversation session and return its ID."""
    return {"conversation_id": create_conversation()}


@router.get("/history/{conversation_id}", response_model=HistoryResponse)
def history(conversation_id: str) -> HistoryResponse:
    """Retrieve the full message history for a conversation."""
    messages = get_history(conversation_id)
    return HistoryResponse(
        conversation_id=conversation_id,
        messages=[HistoryEntry(**m) for m in messages],
    )
