"""HTTP API: authentication, ownership and upload validation."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import langgraph_multi_agent as g
from api import auth
from api.main import app
from tests.conftest import png_bytes

client = TestClient(app)


def register(name=None, password="correct horse"):
    name = name or f"user_{uuid.uuid4().hex[:10]}"
    response = client.post("/api/auth/register", json={"username": name, "password": password})
    assert response.status_code == 200, response.text
    return name, {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def user():
    return register()[1]


def upload(headers, data=None, kind="skin", name="photo.png"):
    return client.post(
        "/api/upload", headers=headers, data={"kind": kind},
        files={"file": (name, data if data is not None else png_bytes(), "image/png")},
    )


# ── authentication ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method, path", [
    ("post", "/api/chat"), ("post", "/api/upload"),
    ("post", "/api/conversations"), ("get", "/api/history/abc"), ("get", "/api/auth/me"),
])
def test_endpoints_require_a_token(method, path):
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_register_login_and_me():
    name, headers = register()
    assert client.get("/api/auth/me", headers=headers).json()["username"] == name
    assert client.post("/api/auth/login", json={"username": name, "password": "correct horse"}).status_code == 200
    assert client.post("/api/auth/login", json={"username": name, "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/register", json={"username": name, "password": "correct horse"}).status_code == 409


def test_weak_or_malformed_registration_is_rejected():
    assert client.post("/api/auth/register", json={"username": "ab", "password": "correct horse"}).status_code == 422
    assert client.post("/api/auth/register", json={"username": "valid_name", "password": "short"}).status_code == 422
    assert client.post("/api/auth/register", json={"username": "<script>", "password": "correct horse"}).status_code == 422


def test_email_style_usernames_are_accepted_and_spaces_are_not():
    name = f"first.last+{uuid.uuid4().hex[:6]}@example.com"
    assert client.post("/api/auth/register", json={"username": name, "password": "correct horse"}).status_code == 200
    assert client.post("/api/auth/login", json={"username": name, "password": "correct horse"}).status_code == 200
    assert client.post("/api/auth/register", json={"username": "has space", "password": "correct horse"}).status_code == 422


def test_hashes_from_the_previous_passlib_setup_still_verify():
    import bcrypt
    legacy = bcrypt.hashpw(b"old password", bcrypt.gensalt()).decode()
    assert auth.verify_password("old password", legacy)
    assert not auth.verify_password("other", legacy)
    assert not auth.verify_password("x", "not-a-hash")


def test_secret_key_is_not_the_old_hardcoded_one():
    assert auth.SECRET_KEY != "medagent-secret-key-change-in-production"


# ── uploads ───────────────────────────────────────────────────────────────────

def test_upload_returns_an_opaque_id_not_a_path(user, upload_dir):
    response = upload(user)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"file_id", "kind"} and body["kind"] == "skin"
    assert "/" not in body["file_id"] and "\\" not in body["file_id"]
    assert len(list(upload_dir.glob("*/*"))) == 1


def test_upload_rejects_non_images_and_oversized_files(user, upload_dir, monkeypatch):
    assert upload(user, data=b"<?php echo 1; ?>", name="evil.png").status_code == 400
    assert upload(user, data=b"").status_code == 400
    assert upload(user, kind="xray").status_code == 422
    import ingestion
    monkeypatch.setattr(ingestion, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr("api.routes.MAX_UPLOAD_BYTES", 100)
    assert upload(user).status_code == 400
    assert list(upload_dir.glob("*/*")) == []


def test_expired_uploads_are_purged(upload_dir):
    import os
    import time
    import ingestion
    file_id = ingestion.save_upload(png_bytes(), user_id=7)
    path = upload_dir / "7" / file_id
    old = time.time() - 48 * 3600
    os.utime(path, (old, old))
    assert ingestion.purge_old_uploads() == 1 and not path.exists()


# ── chat ──────────────────────────────────────────────────────────────────────

def test_chat_rejects_paths_and_other_users_files(user, upload_dir, fake_llm):
    fake_llm({g.Intent: g.Intent(heart=False, breast=False, skin=False, health_question=False)})
    base = {"message": "hi", "conversation_id": "c1"}
    for bad in ("../../data/users.db", r"C:\Windows\win.ini", "/etc/passwd"):
        assert client.post("/api/chat", headers=user, data={**base, "skin_photo_id": bad}).status_code == 400
    file_id = upload(user).json()["file_id"]
    _, other = register()
    assert client.post("/api/chat", headers=other, data={**base, "skin_photo_id": file_id}).status_code == 400
    assert client.post("/api/chat", headers=user, data={**base, "conversation_id": "../x"}).status_code == 400


def test_chat_and_history_round_trip_and_are_private(user, fake_llm):
    fake_llm({g.Intent: g.Intent(heart=False, breast=False, skin=False, health_question=True)}, text="Drink water.")
    conversation = uuid.uuid4().hex
    response = client.post("/api/chat", headers=user, data={"message": "How much water should I drink?", "conversation_id": conversation})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reply"] == "Drink water." and body["routed_to"] == "general"
    assert body["disclaimer"] and body["disclaimer"] not in body["reply"]

    mine = client.get(f"/api/history/{conversation}", headers=user).json()["messages"]
    assert [m["role"] for m in mine] == ["user", "assistant"]
    _, other = register()
    assert client.get(f"/api/history/{conversation}", headers=other).json()["messages"] == []


def test_internal_errors_do_not_leak_details(user, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError(r"secret path C:\models\x.pt")
    monkeypatch.setattr("api.routes.run_graph", boom)
    response = client.post("/api/chat", headers=user, data={"message": "hi", "conversation_id": "c1"})
    assert response.status_code == 500 and "secret" not in response.text


# ── health profile ────────────────────────────────────────────────────────────

PROFILE = {
    "age": 61, "sex": "Male", "height_cm": 178, "weight_kg": 84,
    "conditions": "hypertension", "medications": "lisinopril 10 mg",
    "drug_allergies": "penicillin", "blood_pressure": "150/95", "resting_heart_rate": 72,
}


def test_profile_requires_a_token():
    assert client.get("/api/profile").status_code == 401
    assert client.put("/api/profile", json=PROFILE).status_code == 401


def test_profile_is_empty_until_saved_then_round_trips(user):
    assert client.get("/api/profile", headers=user).json() == {"profile": None}
    saved = client.put("/api/profile", headers=user, json=PROFILE)
    assert saved.status_code == 200, saved.text
    stored = client.get("/api/profile", headers=user).json()["profile"]
    assert stored["age"] == 61 and stored["conditions"] == "hypertension"
    assert stored["food_allergies"] == ""            # unspecified fields default to blank

    client.put("/api/profile", headers=user, json={**PROFILE, "age": 62, "medications": ""})
    stored = client.get("/api/profile", headers=user).json()["profile"]
    assert stored["age"] == 62 and stored["medications"] == ""


def test_profile_belongs_to_its_account(user):
    client.put("/api/profile", headers=user, json=PROFILE)
    _, other = register()
    assert client.get("/api/profile", headers=other).json() == {"profile": None}


@pytest.mark.parametrize("bad", [
    {"sex": "Male"},                                  # age is required
    {"age": 400},
    {"age": 40, "sex": "robot"},
    {"age": 40, "blood_pressure": "high"},
    {"age": 40, "resting_heart_rate": 900},
    {"age": 40, "conditions": "x" * 5000},
])
def test_invalid_profiles_are_rejected(user, bad):
    assert client.put("/api/profile", headers=user, json=bad).status_code == 422


def test_chat_uses_the_profile_on_file_not_client_supplied_values(user, monkeypatch):
    client.put("/api/profile", headers=user, json=PROFILE)
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return {"reply": "ok", "routed_to": "general", "messages": [], "disclaimer": ""}

    monkeypatch.setattr("api.routes.run_graph", capture)
    response = client.post("/api/chat", headers=user, data={
        "message": "hi", "conversation_id": "c1",
        "triage_context": "Self-Reported Urgency: General question",
        "age": "25", "sex": "0",                     # ignored: no longer accepted from the client
    })
    assert response.status_code == 200, response.text
    assert (seen["age"], seen["sex"], seen["resting_blood_pressure"], seen["heart_rate"]) == (61, 1, 150, 72)
    context = seen["patient_context"]
    assert context.startswith("Self-Reported Urgency: General question")
    assert "Medical conditions: hypertension" in context and "drug: penicillin" in context


def test_chat_works_without_a_profile(user, monkeypatch):
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return {"reply": "ok", "routed_to": "general", "messages": [], "disclaimer": ""}

    monkeypatch.setattr("api.routes.run_graph", capture)
    assert client.post("/api/chat", headers=user, data={"message": "hi", "conversation_id": "c1"}).status_code == 200
    assert seen["age"] is None and seen["patient_context"] is None
