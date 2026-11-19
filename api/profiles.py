"""
Health profile: the long-lived patient information that belongs to an account
(age, sex, conditions, medications, allergies ...), stored next to the user's
credentials. It is filled in once after registration, edited from the Profile
panel, and attached by the server to every chat turn.

What is going on today (symptoms, how long, how bad) is not part of the
profile; the patient says that in the chat.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from api.auth import _get_db

_SHORT = 80
_LONG = 1000


class HealthProfile(BaseModel):
    age: int = Field(..., ge=0, le=130)
    sex: Literal["", "Male", "Female", "Other", "Prefer not to say"] = ""
    height_cm: float | None = Field(default=None, gt=0, le=280)
    weight_kg: float | None = Field(default=None, gt=0, le=700)
    pregnancy: str = Field(default="", max_length=_SHORT)
    conditions: str = Field(default="", max_length=_LONG)
    medications: str = Field(default="", max_length=_LONG)
    drug_allergies: str = Field(default="", max_length=_LONG)
    food_allergies: str = Field(default="", max_length=_LONG)
    surgeries: str = Field(default="", max_length=_LONG)
    family_history: str = Field(default="", max_length=_LONG)
    smoking: str = Field(default="", max_length=_SHORT)
    alcohol: str = Field(default="", max_length=_SHORT)
    recreational_drugs: str = Field(default="", max_length=_SHORT * 3)
    blood_pressure: str = Field(default="", max_length=12, description='Usual reading, e.g. "120/80"')
    resting_heart_rate: int | None = Field(default=None, ge=20, le=250)

    @field_validator(
        "pregnancy", "conditions", "medications", "drug_allergies", "food_allergies",
        "surgeries", "family_history", "smoking", "alcohol", "recreational_drugs",
        "blood_pressure", mode="before",
    )
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("blood_pressure")
    @classmethod
    def _blood_pressure_format(cls, value: str) -> str:
        if value and not re.fullmatch(r"\d{2,3}\s*/\s*\d{2,3}", value):
            raise ValueError('Enter blood pressure like "120/80"')
        return value


def _init_table() -> None:
    conn = _get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS health_profiles (
                user_id INTEGER PRIMARY KEY REFERENCES users(id),
                data TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    finally:
        conn.close()


def get_profile(user_id: int) -> HealthProfile | None:
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT data FROM health_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        return HealthProfile(**json.loads(row["data"]))
    except (ValueError, TypeError):
        return None


def save_profile(user_id: int, profile: HealthProfile) -> HealthProfile:
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO health_profiles (user_id, data) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET data = excluded.data, updated_at = datetime('now')",
            (user_id, profile.model_dump_json()),
        )
        conn.commit()
    finally:
        conn.close()
    return profile


def profile_context(profile: HealthProfile | None) -> str:
    """The profile as the plain-text summary shown to the LLM."""
    if profile is None:
        return ""
    lines = ["Patient profile (self-reported, on file):", f"Age: {profile.age}"]

    def add(label: str, value: Any, suffix: str = "") -> None:
        if value not in (None, ""):
            lines.append(f"{label}: {value}{suffix}")

    add("Sex", profile.sex)
    add("Height", profile.height_cm, " cm")
    add("Weight", profile.weight_kg, " kg")
    add("Pregnancy status", profile.pregnancy)
    lines.append(f"Medical conditions: {profile.conditions or 'None reported'}")
    lines.append(f"Medications: {profile.medications or 'None reported'}")
    allergies = "; ".join(
        f"{kind}: {value}" for kind, value in
        (("drug", profile.drug_allergies), ("food", profile.food_allergies)) if value
    )
    lines.append(f"Allergies: {allergies or 'None reported'}")
    add("Past surgeries", profile.surgeries)
    add("Family medical history", profile.family_history)
    add("Smoking status", profile.smoking)
    add("Alcohol use", profile.alcohol)
    add("Recreational drug use", profile.recreational_drugs)
    add("Usual blood pressure", profile.blood_pressure)
    add("Usual resting heart rate", profile.resting_heart_rate, " bpm")
    return "\n".join(lines)


def profile_measurements(profile: HealthProfile | None) -> dict[str, int | None]:
    """The profile fields that feed the heart model and the graph."""
    if profile is None:
        return {"age": None, "sex": None, "resting_blood_pressure": None, "heart_rate": None}
    systolic = re.match(r"(\d{2,3})", profile.blood_pressure)
    return {
        "age": profile.age,
        "sex": {"Male": 1, "Female": 0}.get(profile.sex),
        "resting_blood_pressure": int(systolic.group(1)) if systolic else None,
        "heart_rate": profile.resting_heart_rate,
    }


_init_table()
