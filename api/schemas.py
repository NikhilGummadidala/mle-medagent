"""
Pydantic request/response schemas for the MedAgent API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat message from the frontend."""
    conversation_id: str = Field(
        description="Unique session/conversation identifier for multi-turn persistence"
    )
    message: str = Field(
        description="Patient's natural language message or query"
    )
    heart_rate: int | None = Field(
        default=None,
        description="Heart rate in BPM (optional dedicated numeric input)"
    )
    breast_scan_path: str | None = Field(
        default=None,
        description="Path/URI to a breast scan image (if uploaded)"
    )
    skin_photo_path: str | None = Field(
        default=None,
        description="Path/URI to a skin lesion photo (if uploaded)"
    )


# ── Response Models ───────────────────────────────────────────────────────────

class DiagnosisResult(BaseModel):
    """A single specialist's diagnosis output."""
    diagnosis: str
    confidence: float
    details: str


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
        description="Agent-by-agent conversation steps for step-by-step display"
    )
    disclaimer: str = Field(
        default="",
        description="Clinical disclaimer appended to the synthesis report"
    )


class UploadResponse(BaseModel):
    """Response after a file upload."""
    file_path: str
    file_type: str


class HistoryEntry(BaseModel):
    """A single message in the conversation history."""
    role: str
    content: str


class HistoryResponse(BaseModel):
    """Full conversation history for a session."""
    conversation_id: str
    messages: list[HistoryEntry]


# ── Auth Models ────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """New user registration payload."""
    username: str = Field(..., min_length=3, max_length=32, description="Unique username")
    password: str = Field(..., min_length=6, max_length=128, description="Password")


class UserLogin(BaseModel):
    """Login credentials payload."""
    username: str
    password: str


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
