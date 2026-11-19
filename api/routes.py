"""
FastAPI route definitions for the MedAgent API.

Every endpoint except register/login requires a Bearer token. Conversations and
uploads belong to the authenticated user.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.auth import get_current_user, login_user, register_user
from api.graph_runner import create_conversation, get_history, run_graph
from api.profiles import (
    HealthProfile,
    get_profile,
    profile_context,
    profile_measurements,
    save_profile,
)
from api.schemas import (
    ChatResponse,
    DiagnosisResult,
    HistoryEntry,
    HistoryResponse,
    ProfileResponse,
    TokenResponse,
    UploadResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from ingestion import MAX_UPLOAD_BYTES, InvalidUpload, purge_old_uploads, resolve_upload, save_upload

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_user)])
auth_router = APIRouter()

MAX_MESSAGE_CHARS = 4000
MAX_CONTEXT_CHARS = 6000
_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _conversation_id(value: str) -> str:
    if not _CONVERSATION_ID.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid conversation_id.")
    return value


def _owned_upload(field: str, file_id: str | None, user_id: int) -> str | None:
    if not file_id:
        return None
    path = resolve_upload(file_id, user_id)
    if path is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field}: use a file_id returned by /api/upload (uploads expire after a day).",
        )
    return path


def _to_diagnosis(data: dict | None) -> DiagnosisResult | None:
    if not data or not data.get("diagnosis"):
        return None
    return DiagnosisResult(
        diagnosis=data["diagnosis"],
        confidence=data.get("confidence"),
        summary=data.get("summary", ""),
        details=data.get("details", ""),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(..., min_length=1, max_length=MAX_MESSAGE_CHARS),
    conversation_id: str = Form(...),
    breast_scan_id: str | None = Form(default=None),
    skin_photo_id: str | None = Form(default=None),
    triage_context: str | None = Form(default=None, max_length=MAX_CONTEXT_CHARS),
    user: dict = Depends(get_current_user),
) -> ChatResponse:
    """
    Send a message. Images are attached by the file_id returned from /upload.

    The patient's health profile is read from their account, not sent by the
    client. ``triage_context`` carries only this conversation's safety
    screening result.
    """
    conversation_id = _conversation_id(conversation_id)
    breast_scan_path = _owned_upload("breast_scan_id", breast_scan_id, user["id"])
    skin_photo_path = _owned_upload("skin_photo_id", skin_photo_id, user["id"])

    profile = get_profile(user["id"])
    on_file = profile_measurements(profile)
    patient_context = "\n\n".join(
        part for part in ((triage_context or "").strip(), profile_context(profile)) if part
    ) or None

    # The graph makes blocking model/LLM calls that take seconds to minutes;
    # run it in a worker thread so the event loop keeps serving other requests.
    try:
        result = await run_in_threadpool(
            run_graph,
            user_id=user["id"],
            conversation_id=conversation_id,
            message=message,
            heart_rate=on_file["heart_rate"],
            breast_scan_path=breast_scan_path,
            skin_photo_path=skin_photo_path,
            patient_context=patient_context,
            age=on_file["age"],
            sex=on_file["sex"],
            resting_blood_pressure=on_file["resting_blood_pressure"],
        )
    except Exception:
        logger.exception("Chat turn failed for conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="The assistant hit an internal error. Please try again.")

    return ChatResponse(
        conversation_id=conversation_id,
        reply=result["reply"],
        heart_disease=_to_diagnosis(result.get("heart_disease")),
        breast_cancer=_to_diagnosis(result.get("breast_cancer")),
        skin_disease=_to_diagnosis(result.get("skin_disease")),
        routed_to=result["routed_to"],
        messages=[HistoryEntry(**m) for m in result["messages"]],
        disclaimer=result.get("disclaimer", ""),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    kind: Literal["mammogram", "skin"] = Form(...),
    user: dict = Depends(get_current_user),
) -> UploadResponse:
    """
    Upload a mammogram or skin photo. Returns a file_id to pass to /chat as
    breast_scan_id or skin_photo_id.
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        file_id = await run_in_threadpool(save_upload, data, user["id"])
    except InvalidUpload as error:
        raise HTTPException(status_code=400, detail=str(error))
    await run_in_threadpool(purge_old_uploads)
    return UploadResponse(file_id=file_id, kind=kind)


@router.get("/profile", response_model=ProfileResponse)
def read_profile(user: dict = Depends(get_current_user)) -> ProfileResponse:
    """The caller's health profile, or null if they have not filled it in."""
    return ProfileResponse(profile=get_profile(user["id"]))


@router.put("/profile", response_model=ProfileResponse)
def write_profile(profile: HealthProfile, user: dict = Depends(get_current_user)) -> ProfileResponse:
    """Create or replace the caller's health profile."""
    return ProfileResponse(profile=save_profile(user["id"], profile))


@router.post("/conversations", response_model=dict)
def new_conversation() -> dict:
    """Create a new conversation session and return its ID."""
    return {"conversation_id": create_conversation()}


@router.get("/history/{conversation_id}", response_model=HistoryResponse)
def history(conversation_id: str, user: dict = Depends(get_current_user)) -> HistoryResponse:
    """Retrieve the message history of one of the caller's conversations."""
    conversation_id = _conversation_id(conversation_id)
    return HistoryResponse(
        conversation_id=conversation_id,
        messages=[HistoryEntry(**m) for m in get_history(user["id"], conversation_id)],
    )


# ── Auth Endpoints ─────────────────────────────────────────────────────────────

@auth_router.post("/register", response_model=TokenResponse)
def register(payload: UserCreate) -> TokenResponse:
    """Register a new user account."""
    result = register_user(payload.username, payload.password)
    return TokenResponse(
        access_token=result["access_token"],
        user=UserResponse(**result["user"]),
    )


@auth_router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin) -> TokenResponse:
    """Log in with username and password."""
    result = login_user(payload.username, payload.password)
    return TokenResponse(
        access_token=result["access_token"],
        user=UserResponse(**result["user"]),
    )


@auth_router.get("/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user)) -> UserResponse:
    """Return the current authenticated user's info."""
    return UserResponse(**current_user)
