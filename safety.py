"""Server-side emergency screening.

The browser runs the same screen before a message is sent, but the API can be
called directly, so the backend checks every message too and puts an emergency
notice at the top of the reply. Patterns mirror ``_RED_FLAG_PATTERNS`` in
``medichat/index.html``; keep the two in step.
"""
from __future__ import annotations

import re

_I = re.IGNORECASE

RED_FLAG_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("chest", "Chest pain or pressure", re.compile(
        r"\bheart attack\b"
        r"|\bchest\b[^.!?]{0,25}\b(pain|painful|hurt|hurts|hurting|ache|aches|aching|tight|tightness"
        r"|pressure|heavy|heaviness|squeez\w*|crushing|burning)\b"
        r"|\b(pain|pressure|tightness|heaviness|crushing)\b[^.!?]{0,20}\bchest\b", _I)),
    ("breath", "Trouble breathing", re.compile(
        r"\bcan'?t breathe\b|\bcannot breathe\b|\btrouble breathing\b"
        r"|\bstruggling to breathe\b|\bgasping for air\b", _I)),
    ("stroke", "Possible signs of a stroke", re.compile(
        r"\bstroke\b|\bslurred speech\b|\bface (is )?droop|\bnumb on one side\b"
        r"|\bone side of (my|his|her|their) (face|body)\b", _I)),
    ("bleed", "Heavy bleeding", re.compile(
        r"\bwon'?t stop bleeding\b|\bbleeding (heavily|badly|a lot|non-?stop)\b"
        r"|\bcoughing up blood\b|\bvomiting blood\b", _I)),
    ("faint", "Fainting or unresponsiveness", re.compile(
        r"\bpassed out\b|\bunconscious\b|\bunresponsive\b|\bfainted\b|\bwon'?t wake up\b", _I)),
    ("allergy", "Severe allergic reaction", re.compile(
        r"\banaphyla|\bthroat (is )?closing\b|\b(face|tongue|lips) (is |are )?swelling\b", _I)),
    ("seizure", "A seizure", re.compile(r"\bseizure\b|\bconvulsion|\bseizing\b", _I)),
    ("head", "Serious head or neck injury", re.compile(
        r"\bhead injury\b|\bskull fracture\b|\b(broke|broken) (my |his |her |their )?neck\b"
        r"|\bneck injury\b", _I)),
    ("selfharm", "Thoughts of self-harm", re.compile(
        r"\bkill myself\b|\bsuicid|\bend my life\b|\b(harm|hurt) myself\b|\bwant to die\b|\boverdos", _I)),
)

# "I don't have chest pain" / "no trouble breathing" must not raise the alarm.
# A hit is ignored when a denial sits just before it in the same sentence.
_DENIED_BEFORE = re.compile(
    r"\b(?:no|not|never|without|don'?t|doesn'?t|didn'?t|do not|does not|did not|haven'?t|hasn'?t"
    r"|isn'?t|aren'?t|wasn'?t|denies|free of)\b(?:(?!\b(?:but|however|though|although|yet|except)\b)[^.!?;,]){0,25}$", _I)


def _denied(message: str, match: re.Match[str]) -> bool:
    return _DENIED_BEFORE.search(message[:match.start()]) is not None


# Only treat a match as a report when the message is about a person, so that
# "what is a stroke?" does not trigger the notice.
_ABOUT_A_PERSON = re.compile(
    r"\b(i|i'm|im|i've|ive|my|me|we|he|she|they|his|her|their|mom|dad|son|daughter"
    r"|wife|husband|partner|friend|child|baby)\b", _I)

# Sent by the browser's intake summary when the user reported red flags and continued.
_CLIENT_REPORTED = re.compile(r"RED FLAGS REPORTED[^\n]*?—\s*([^\n]+)")


def scan_red_flags(message: str, patient_context: str | None = None) -> list[dict[str, str]]:
    """Return the emergency categories present, as ``[{"key", "label"}]``."""
    hits: dict[str, str] = {}
    if message and _ABOUT_A_PERSON.search(message):
        for key, label, pattern in RED_FLAG_PATTERNS:
            if any(not _denied(message, match) for match in pattern.finditer(message)):
                hits[key] = label
    if patient_context:
        reported = _CLIENT_REPORTED.search(patient_context)
        if reported:
            for label in reported.group(1).split(";"):
                label = label.strip()
                if label:
                    hits.setdefault(f"reported:{label.lower()}", label)
    return [{"key": key, "label": label} for key, label in hits.items()]


def emergency_notice(flags: list[dict[str, str]]) -> str:
    """The notice placed above any reply when red flags are present."""
    if not flags:
        return ""
    labels = "; ".join(flag["label"] for flag in flags)
    text = (
        f"**🚨 This may be an emergency ({labels}).** If this is happening now, call 911 "
        "(or your local emergency number) or go to the nearest emergency room. Do not wait "
        "for this app; it cannot send help."
    )
    if any(flag["key"] == "selfharm" or "harming" in flag["label"].lower() for flag in flags):
        text += (
            " If you are thinking about harming yourself, call or text 988 (Suicide & Crisis "
            "Lifeline, US) at any time. You do not have to face this alone."
        )
    return text
