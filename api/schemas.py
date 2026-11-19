"""
Pydantic request/response schemas for the MedAgent API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Response Models ───────────────────────────────────────────────────────────

class DiagnosisResult(BaseModel):
    """A single specialist's output."""
    diagnosis: str
    confidence: float | None = Field(
        default=None,
        description="Model probability of the stated result; null when the model reports "
                    "a threshold decision or was not run",
    )
    summary: str = ""
    details: str = ""


class HistoryEntry(BaseModel):
    """A single message in the conversation history."""
    role: str
    content: str


class ChatResponse(BaseModel):
    """Response returned to the frontend after processing."""
    conversation_id: str
    reply: str
    heart_disease: DiagnosisResult | None = None
    breast_cancer: DiagnosisResult | None = None
    skin_disease: DiagnosisResult | None = None
    routed_to: str
    messages: list[HistoryEntry] = Field(
        default_factory=list,
        description="This turn's specialist steps followed by the final reply",
    )
    disclaimer: str = Field(default="", description="Clinical disclaimer to show with the reply")


class UploadResponse(BaseModel):
    """Response after a file upload."""
    file_id: str
    kind: str


class ProfileResponse(BaseModel):
    """The account's health profile (see api.profiles.HealthProfile), or null."""
    profile: Any | None = None


class HistoryResponse(BaseModel):
    """Full conversation history for a session."""
    conversation_id: str
    messages: list[HistoryEntry]


# ── Auth Models ────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """New user registration payload."""
    username: str = Field(
        ..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.@+-]+$",
        description="Unique username or email (letters, digits, _ . - @ +; no spaces)",
    )
    password: str = Field(..., min_length=8, max_length=72, description="Password")


class UserLogin(BaseModel):
    """Login credentials payload."""
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=128)


class UserResponse(BaseModel):
    """Public user info returned to the client."""
    id: int
    username: str
    created_at: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
