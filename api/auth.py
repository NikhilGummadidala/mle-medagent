"""
User authentication: username/password with JWT bearer tokens.

SQLite for storage, bcrypt for password hashing, python-jose for JWTs. The
signing key comes from the MEDAGENT_SECRET_KEY environment variable; if that is
unset a random key is generated once and kept in ``data/secret.key`` (which is
git-ignored), so no secret lives in the source code.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

# ── Config ────────────────────────────────────────────────────────────────────

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

DB_DIR = Path(os.environ.get("MEDAGENT_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
DB_DIR.mkdir(exist_ok=True)
DB_PATH = str(DB_DIR / "users.db")


def _load_secret_key() -> str:
    from_env = os.environ.get("MEDAGENT_SECRET_KEY")
    if from_env:
        return from_env
    key_file = DB_DIR / "secret.key"
    if not key_file.exists():
        key_file.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    return key_file.read_text(encoding="utf-8").strip()


SECRET_KEY = _load_secret_key()
_bearer = HTTPBearer(auto_error=False)


# ── Passwords ─────────────────────────────────────────────────────────────────

def _password_bytes(password: str) -> bytes:
    # bcrypt only reads the first 72 bytes; truncate explicitly so newer bcrypt
    # releases (which raise on longer input) behave like older ones.
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_password_bytes(password), password_hash.encode("ascii"))
    except ValueError:
        return False


# ── Database ──────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _get_user(column: str, value: Any) -> dict[str, Any] | None:
    assert column in ("id", "username")
    conn = _get_db()
    try:
        row = conn.execute(
            f"SELECT id, username, password_hash, created_at FROM users WHERE {column} = ?",
            (value,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _create_user(username: str, password_hash: str) -> dict[str, Any]:
    conn = _get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "username": username,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )
    finally:
        conn.close()


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _create_token(user_id: int, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# ── Public API ────────────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> dict[str, Any]:
    """Register a new user. Returns user info + access_token."""
    user = _create_user(username, hash_password(password))
    return {
        "access_token": _create_token(user["id"], user["username"]),
        "token_type": "bearer",
        "user": user,
    }


def login_user(username: str, password: str) -> dict[str, Any]:
    """Authenticate a user. Returns user info + access_token on success."""
    user = _get_user("username", username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return {
        "access_token": _create_token(user["id"], user["username"]),
        "token_type": "bearer",
        "user": {k: user[k] for k in ("id", "username", "created_at")},
    }


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """FastAPI dependency: the user identified by the Bearer token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    payload = _decode_token(credentials.credentials)
    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    user = _get_user("id", user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return {k: user[k] for k in ("id", "username", "created_at")}


# Initialize the database on import
_init_db()
