"""Shared test setup: an isolated data directory and a scripted stand-in LLM."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must be set before api.auth is imported: keeps tests away from the real user DB.
_DATA = tempfile.mkdtemp(prefix="medagent-test-")
os.environ["MEDAGENT_DATA_DIR"] = _DATA
os.environ["MEDAGENT_SECRET_KEY"] = "test-secret-key"
os.environ["MEDAGENT_SKIP_LLM_CHECK"] = "1"

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import langgraph_multi_agent as graph_module  # noqa: E402


class FakeLLM:
    """Replaces ChatOllama. ``structured`` maps a schema class to the object (or
    callable taking the messages) returned for that schema."""

    def __init__(self, structured=None, text="Plain explanation."):
        self.structured = structured or {}
        self.text = text
        self.calls: list[str] = []
        self._schema = None

    def with_structured_output(self, schema):
        bound = FakeLLM(self.structured, self.text)
        bound.calls = self.calls
        bound._schema = schema
        return bound

    def invoke(self, messages):
        if self._schema is None:
            self.calls.append("text")
            if isinstance(self.text, Exception):
                raise self.text
            return AIMessage(content=self.text)
        self.calls.append(self._schema.__name__)
        value = self.structured.get(self._schema)
        if isinstance(value, Exception):
            raise value
        return value(messages) if callable(value) else value


@pytest.fixture
def fake_llm(monkeypatch):
    def install(structured=None, text="Plain explanation."):
        llm = FakeLLM(structured, text)
        monkeypatch.setattr(graph_module, "get_llm", lambda: llm)
        return llm
    return install


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    import ingestion
    monkeypatch.setattr(ingestion, "UPLOAD_DIR", tmp_path / "uploads")
    return tmp_path / "uploads"


def png_bytes(size=(64, 64), color=(120, 60, 30), mode="RGB") -> bytes:
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    return buffer.getvalue()
