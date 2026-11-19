"""
Session-aware graph runner. Wraps the compiled LangGraph with a checkpointer
so measurements, uploaded images and message history persist across the turns
of a conversation. Conversations are namespaced by user id.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_multi_agent import (
    CLINICAL_DISCLAIMER,
    OLLAMA_BASE_URL,
    compile_medagent_graph,
)
from models.schema import validate_heart_measurements
from safety import scan_red_flags

logger = logging.getLogger(__name__)

LLM_OFFLINE_NOTE = (
    "*Note: the language model is offline, so plain-language explanations and general "
    "answers are unavailable. Model results above are unaffected.*"
)

# In-memory persistence: conversations are lost when the server restarts.
_checkpointer = MemorySaver()
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = compile_medagent_graph(checkpointer=_checkpointer)
    return _graph


def llm_available() -> bool:
    """True if the Ollama server answers."""
    if os.environ.get("MEDAGENT_SKIP_LLM_CHECK"):
        return True
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _config(user_id: int, conversation_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"{int(user_id)}:{conversation_id}"}}


def run_graph(
    user_id: int,
    conversation_id: str,
    message: str,
    heart_rate: int | None = None,
    breast_scan_path: str | None = None,
    skin_photo_path: str | None = None,
    patient_context: str | None = None,
    age: int | None = None,
    sex: int | None = None,
    resting_blood_pressure: int | None = None,
) -> dict[str, Any]:
    """
    Run one conversation turn.

    Image paths must already be resolved to files this user uploaded. Returns a
    dict with keys: reply, routed_to, heart_disease, breast_cancer,
    skin_disease, messages (this turn's specialist steps then the reply),
    disclaimer.
    """
    intake, _ = validate_heart_measurements({
        "age": age, "sex": sex, "resting_blood_pressure": resting_blood_pressure,
    })
    new_images = [
        name for name, path in
        (("breast_cancer", breast_scan_path), ("skin_disease", skin_photo_path)) if path
    ]

    turn = {
        "messages": [HumanMessage(content=message)],
        # Merged into what the conversation already knows (see merge_known).
        "user_measurements": intake,
        "breast_cancer_features": {},
        "file_references": {
            "breast_scan_path": breast_scan_path, "skin_photo_path": skin_photo_path,
        },
        # Fresh every turn.
        "raw_user_query": message,
        "patient_context": patient_context,
        "heart_rate": heart_rate,
        "new_images": new_images,
        "new_measurements": [],
        "red_flags": scan_red_flags(message, patient_context),
        "pending": [],
        "consulted": [],
        "intent": {},
        "steps": [],
        "heart_disease_analysis": {},
        "breast_cancer_analysis": {},
        "skin_disease_analysis": {},
        "heart_diagnosis": None,
        "cancer_diagnosis": None,
        "skin_diagnosis": None,
        "next_agent": "",
    }

    online = llm_available()
    result = _get_graph().invoke(turn, _config(user_id, conversation_id))

    reply = result["messages"][-1].content
    if not online:
        reply = f"{reply}\n\n{LLM_OFFLINE_NOTE}"

    steps = [{"role": "assistant", "content": step} for step in result.get("steps", [])]
    return {
        "reply": reply,
        "routed_to": result.get("next_agent") or "unknown",
        "heart_disease": result.get("heart_disease_analysis"),
        "breast_cancer": result.get("breast_cancer_analysis"),
        "skin_disease": result.get("skin_disease_analysis"),
        "messages": [*steps, {"role": "assistant", "content": reply}],
        "disclaimer": CLINICAL_DISCLAIMER,
    }


def get_history(user_id: int, conversation_id: str) -> list[dict[str, str]]:
    """The conversation as ``[{"role": "user"|"assistant", "content": ...}]``."""
    snapshot = _get_graph().get_state(_config(user_id, conversation_id))
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    return [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
    ]


def create_conversation() -> str:
    """Generate a new unique conversation ID."""
    return str(uuid.uuid4())
