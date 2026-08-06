"""
Session-aware graph runner. Wraps the LangGraph compiled graph
with MemorySaver persistence for multi-turn conversations.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_multi_agent import (
    BREAST_CANCER_FEATURE_KEYS,
    USER_MEASUREMENT_KEYS,
    CLINICAL_DISCLAIMER,
    compile_medagent_graph,
)

logger = logging.getLogger(__name__)

# MemorySaver for per-session persistence
_checkpointer = MemorySaver()

# Compiled graph (singleton)
_graph = None


def _get_graph():
    """Lazy-initialize the compiled LangGraph with MemorySaver."""
    global _graph
    if _graph is None:
        _graph = compile_medagent_graph(checkpointer=_checkpointer)
    return _graph


def _check_ollama() -> None:
    """Raise a clear error if the Ollama server is not reachable."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        urllib.request.urlopen(req, timeout=3)
    except (urllib.error.URLError, ConnectionRefusedError, OSError) as e:
        raise ConnectionError(
            "Ollama server is not running. "
            "Start it by running: ollama serve\n"
            "Then make sure the model is pulled: ollama pull llama3.2"
        ) from e


def run_graph(
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
    Invoke the MedAgent graph for a given conversation.

    Parameters
    ----------
    conversation_id : str
        Session identifier for multi-turn persistence.
    message : str
        The patient's latest message.
    heart_rate : int | None
        Optional heart rate in BPM from dedicated input.
    breast_scan_path : str | None
        Optional path to a breast scan image.
    skin_photo_path : str | None
        Optional path to a skin lesion photo.

    Returns
    -------
    dict with keys: reply, routed_to, heart_disease, breast_cancer, skin_disease,
                     messages, disclaimer
    """
    _check_ollama()

    graph = _get_graph()

    # Build initial state with all required fields
    merged_measurements: dict[str, int | float | None] = {
        key: None for key in USER_MEASUREMENT_KEYS
    }
    if heart_rate is not None:
        merged_measurements["max_heart_rate"] = heart_rate
    if age is not None:
        merged_measurements["age"] = age
    if sex is not None:
        merged_measurements["sex"] = sex
    if resting_blood_pressure is not None:
        merged_measurements["resting_blood_pressure"] = resting_blood_pressure

    merged_breast: dict[str, float | None] = {
        key: None for key in BREAST_CANCER_FEATURE_KEYS
    }

    file_refs: dict[str, str | None] = {
        "breast_scan_path": breast_scan_path,
        "skin_photo_path": skin_photo_path,
    }

    # The clipboard rides along with the first message so the router and synthesis
    # nodes see it; raw_user_query stays the patient's own words for extraction.
    initial_text = message
    if patient_context:
        initial_text = f"{patient_context}\n\n---\n\nPatient's question: {message}"

    state = {
        # Detailed fields
        "raw_user_query": message,
        "user_measurements": merged_measurements,
        "breast_cancer_features": merged_breast,
        "file_references": file_refs,
        "next_agent": "",
        "messages": [HumanMessage(content=initial_text)],
        "heart_disease_analysis": {},
        "breast_cancer_analysis": {},
        "skin_disease_analysis": {},
        # User-facing aliases
        "user_query": message,
        "heart_rate": merged_measurements.get("max_heart_rate"),
        "breast_scan_path": breast_scan_path,
        "skin_image_path": skin_photo_path,
        # LLM explanations
        "heart_diagnosis": None,
        "cancer_diagnosis": None,
        "skin_diagnosis": None,
    }

    # Run the graph with MemorySaver thread config
    config = {"configurable": {"thread_id": conversation_id}}
    result = graph.invoke(state, config)

    # Extract the latest AI message as the reply
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_messages[-1].content if ai_messages else "No response generated."

    # Build agent-by-agent message history
    agent_messages = []
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            agent_messages.append({"role": "assistant", "content": m.content})
        elif isinstance(m, HumanMessage):
            agent_messages.append({"role": "user", "content": m.content})

    return {
        "reply": reply,
        "routed_to": result.get("next_agent", "unknown"),
        "heart_disease": result.get("heart_disease_analysis"),
        "breast_cancer": result.get("breast_cancer_analysis"),
        "skin_disease": result.get("skin_disease_analysis"),
        "messages": agent_messages,
        "disclaimer": CLINICAL_DISCLAIMER,
    }


def get_history(conversation_id: str) -> list[dict[str, str]]:
    """
    Return the conversation history for a given session.

    Returns
    -------
    list of {"role": "user"|"assistant", "content": "..."}
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": conversation_id}}

    # Try to get the latest state from the checkpointer
    try:
        state_snapshot = graph.get_state(config)
        if state_snapshot and state_snapshot.values:
            messages = state_snapshot.values.get("messages", [])
            return [
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": m.content,
                }
                for m in messages
                if isinstance(m, (HumanMessage, AIMessage))
            ]
    except Exception:
        pass

    return []


def create_conversation() -> str:
    """Generate a new unique conversation ID."""
    return str(uuid.uuid4())
