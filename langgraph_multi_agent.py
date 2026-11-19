"""
Multi-agent medical assistant built on LangGraph.

Flow for one turn:

    Extract -> Plan -> [specialist ...] -> Synthesis
                    -> General (health question, no specialist input)
                    -> SmallTalk (greeting / thanks)

* Extract pulls measurements out of the message with an LLM, then keeps only
  values that are in range and actually appear in the patient's words.
* Plan decides once which specialists run this turn. Each specialist runs at
  most once, in a fixed order, so the graph cannot loop.
* Specialists run the ML models. An LLM is used only to explain a result the
  model actually produced, never to interpret an image or stand in for a model.
* Synthesis assembles the reply deterministically from the model outputs.

Measurements and uploaded images persist across turns of a conversation.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, add_messages
from pydantic import BaseModel, Field

from models.breast_cancer import predict as predict_breast_cancer
from models.heart_disease import follow_up_questions as heart_follow_up_questions
from models.heart_disease import inputs_sufficient as heart_inputs_sufficient
from models.heart_disease import insufficient_data as heart_insufficient_data
from models.heart_disease import more_information_needed as heart_more_information_needed
from models.heart_disease import REQUIRED_QUESTIONS as HEART_REQUIRED_QUESTIONS
from models.heart_disease import predict as predict_heart_disease
from models.schema import (
    BREAST_FEATURE_KEYS,
    HEART_FEATURE_BY_KEY,
    HEART_FEATURE_KEYS,
    validate_breast_features,
    validate_heart_measurements,
)
from models.skin_disease import predict as predict_skin_disease
from safety import emergency_notice

logger = logging.getLogger(__name__)

TEXT_MODEL = os.environ.get("MEDAGENT_TEXT_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
HISTORY_MESSAGES = 6          # prior messages shown to the LLM for context
SPECIALIST_ORDER = ("heart_disease", "breast_cancer", "skin_disease")

# Kept under their old names for callers that import them from here.
USER_MEASUREMENT_KEYS = HEART_FEATURE_KEYS
BREAST_CANCER_FEATURE_KEYS = BREAST_FEATURE_KEYS


def get_llm() -> ChatOllama:
    """The text LLM. A function so tests can substitute a stub."""
    return ChatOllama(
        model=TEXT_MODEL, temperature=0, base_url=OLLAMA_BASE_URL, num_ctx=8192,
    )


# ── State ─────────────────────────────────────────────────────────────────────

def merge_known(old: dict | None, new: dict | None) -> dict:
    """Reducer: keep earlier values, let new non-null values override them."""
    merged = dict(old or {})
    merged.update({k: v for k, v in (new or {}).items() if v is not None})
    return merged


class MedAgentState(TypedDict, total=False):
    # Conversation: the patient's own messages and the final reply of each turn.
    messages: Annotated[list[BaseMessage], add_messages]

    # Persist across turns (merged by the reducer).
    user_measurements: Annotated[dict[str, int | float], merge_known]
    breast_cancer_features: Annotated[dict[str, float], merge_known]
    file_references: Annotated[dict[str, str], merge_known]   # breast_scan_path / skin_photo_path
    awaiting: str | None                # specialist whose follow-up questions are open
    heart_helpful_asked: bool           # the optional heart questions were already put once

    # Supplied fresh every turn.
    raw_user_query: str
    patient_context: str | None
    heart_rate: int | None              # resting pulse: context only, not a model feature
    new_images: list[str]               # specialists that received an image this turn
    new_measurements: list[str]         # heart measurements stated in this message
    red_flags: list[dict[str, str]]

    # Produced during the turn (reset by the runner at the start of each turn).
    pending: list[str]
    consulted: list[str]
    intent: dict[str, bool]
    steps: list[str]
    heart_disease_analysis: dict[str, Any]
    breast_cancer_analysis: dict[str, Any]
    skin_disease_analysis: dict[str, Any]
    heart_diagnosis: str | None
    cancer_diagnosis: str | None
    skin_diagnosis: str | None
    next_agent: str


# ── Assessment sanitization ───────────────────────────────────────────────────

SAFE_ASSESSMENT_FALLBACK = (
    "Please interpret the ML result shown above with caution and consult a "
    "qualified healthcare professional for a complete evaluation."
)

_NUMERIC_CLAIM_WORDS = (
    r"confiden[ct]\w*|probab\w*|likely|likelihood|chance\w*|risk|odds"
)

_CONFIDENCE_PATTERNS = (
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    re.compile(r"\b\d+(?:\.\d+)?\s*percent\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:{_NUMERIC_CLAIM_WORDS})\b[^.\n]{{0,60}}\b0\.\d+\b"
        rf"|\b0\.\d+\b[^.\n]{{0,60}}\b(?:{_NUMERIC_CLAIM_WORDS})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b\d+\s+(?:in|out\s+of)\s+\d+\b[^.\n]{{0,30}}\b(?:{_NUMERIC_CLAIM_WORDS})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+(?:am|was|feel|felt)|we\s+(?:are|were|feel|felt))\s+"
        r"(?:very|pretty|fairly|quite|extremely|highly|really)?\s*"
        r"(?:confident|certain|sure)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:highly|very|extremely)\s+(?:likely|unlikely|probable)\b", re.IGNORECASE),
    re.compile(r"\b(?:almost|nearly)\s+certain\b", re.IGNORECASE),
    re.compile(r"\b(?:almost\s+)?certainly\b", re.IGNORECASE),
)

_RATIO_STRIP = re.compile(r"\b(?:odds|risk|hazard)\s+ratios?\b", re.IGNORECASE)


def sanitize_assessment(text: Any) -> str:
    """Deterministically reject LLM explanation text that introduces its own
    confidence, probability, likelihood, or risk claims. Returns a fixed
    fallback on any hit, otherwise the original text."""
    if not isinstance(text, str) or not text.strip():
        return SAFE_ASSESSMENT_FALLBACK
    stripped = _RATIO_STRIP.sub(" ", text)
    for pattern in _CONFIDENCE_PATTERNS:
        if pattern.search(stripped):
            return SAFE_ASSESSMENT_FALLBACK
    return text.strip()


# ── Extraction ────────────────────────────────────────────────────────────────

class ExtractedInputs(BaseModel):
    """Measurements the patient stated. Every field is null unless stated."""
    age: float | None = None
    sex: Literal["male", "female"] | None = None
    chest_pain_type: Literal[
        "typical angina", "atypical angina", "non-anginal pain", "asymptomatic"
    ] | None = Field(default=None, description=(
        "typical angina = pressure-like chest pain brought on by exertion and relieved by rest; "
        "atypical angina = chest pain with only some of those features; "
        "non-anginal pain = chest pain unrelated to exertion; "
        "asymptomatic = the patient says they have no chest pain"))
    resting_blood_pressure: float | None = Field(default=None, description="systolic, mmHg")
    cholesterol: float | None = Field(default=None, description="total serum cholesterol, mg/dl")
    fasting_blood_sugar_above_120: bool | None = None
    resting_ecg: Literal[
        "normal", "ST-T wave abnormality", "left ventricular hypertrophy"
    ] | None = None
    max_heart_rate: float | None = Field(default=None, description=(
        "peak heart rate reached in an exercise/stress test ONLY; never a resting pulse"))
    exercise_induced_angina: bool | None = None
    oldpeak: float | None = Field(default=None, description="ST depression in mm")
    slope: Literal["upsloping", "flat", "downsloping"] | None = None
    ca: int | None = Field(default=None, description="major vessels coloured by fluoroscopy, 0-3")
    thal: Literal["normal", "fixed defect", "reversible defect"] | None = None
    breast_cancer_features: dict[str, float | None] = Field(
        default_factory=dict,
        description="Fine-needle-aspirate lab features if listed, e.g. radius_mean, texture_worst",
    )


EXTRACTOR_SYSTEM_PROMPT = """You extract measurements from a patient's message for a medical app.

Rules:
- Fill a field ONLY if the patient explicitly states it. Otherwise leave it null.
- Never guess, infer, or use typical values. Do not fill a field with 0 or false because it was not mentioned.
- Blood pressure "140/90" -> resting_blood_pressure 140.
- max_heart_rate is only the peak rate from an exercise/stress test. A resting or current pulse ("my heart rate is 90") is NOT max_heart_rate; leave it null.
- breast_cancer_features: only if the patient lists pathology lab values by name."""

# A categorical or yes/no field is accepted only if the message mentions its topic.
_TOPIC_EVIDENCE = {
    "sex": r"\b(male|female|man|woman|boy|girl|gentleman|lady|he|she|mr|mrs|ms)\b",
    "chest_pain_type": r"chest|angina|asymptomatic",
    "fasting_blood_sugar": r"sugar|glucose|diabet",
    "resting_ecg": r"\b(ecg|ekg)\b|electrocardio|st-t|hypertrophy",
    "exercise_induced_angina": r"exerc|exert|stairs|walk|running|jog|activity|angina",
    "slope": r"slop",
    "thal": r"thal|defect",
}
_CATEGORY_FIELDS = ("sex", "chest_pain_type", "resting_ecg", "slope", "thal")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers_in(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(text)}


def _code_for(key: str, meaning: str) -> int | None:
    for code, label in HEART_FEATURE_BY_KEY[key].codes.items():
        if label == meaning:
            return code
    return None


def ground_extraction(extracted: ExtractedInputs, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn the LLM's extraction into model inputs, dropping anything the
    patient's text does not support. Numbers must literally appear in the
    message; categories and yes/no answers need their topic mentioned."""
    lowered = text.lower()
    stated = _numbers_in(text)
    heart: dict[str, Any] = {}

    def topic(key: str) -> bool:
        return re.search(_TOPIC_EVIDENCE[key], lowered) is not None

    for key in ("age", "resting_blood_pressure", "cholesterol", "max_heart_rate", "oldpeak", "ca"):
        value = getattr(extracted, key)
        if value is not None and float(value) in stated:
            heart[key] = value
    if "ca" in heart and not re.search(r"vessel|fluoroscop", lowered):
        del heart["ca"]

    for key in _CATEGORY_FIELDS:
        meaning = getattr(extracted, key)
        if meaning is not None and topic(key):
            heart[key] = _code_for(key, meaning)
    if extracted.fasting_blood_sugar_above_120 is not None and topic("fasting_blood_sugar"):
        heart["fasting_blood_sugar"] = int(extracted.fasting_blood_sugar_above_120)
    if extracted.exercise_induced_angina is not None and topic("exercise_induced_angina"):
        heart["exercise_induced_angina"] = int(extracted.exercise_induced_angina)

    breast = {
        key: value for key, value in extracted.breast_cancer_features.items()
        if value is not None and float(value) in stated
    }
    heart, _ = validate_heart_measurements(heart)
    breast, _ = validate_breast_features(breast)
    return heart, breast


_YesNo = Literal["yes", "no", "not stated"]


class ChestPainFacts(BaseModel):
    """Three plain facts about the patient's chest pain."""
    has_chest_pain: _YesNo = Field(description=(
        "Does the patient report chest pain, pressure, tightness or discomfort?"))
    brought_on_by_exertion: _YesNo = Field(description=(
        "Does physical effort (walking, stairs, jogging, exercise) bring the chest pain on?"))
    relieved_by_rest: _YesNo = Field(description=(
        "Does the chest pain go away when the patient rests or stops the activity?"))


# Worked examples matter here: without them the small local model leaves the
# answers blank for clearly described symptoms.
CHEST_PAIN_PROMPT = """Read the patient's message and answer three questions about their chest pain with "yes", "no" or "not stated".

Examples:
Message: "My chest hurts when I walk uphill and eases when I sit down."
has_chest_pain: yes, brought_on_by_exertion: yes, relieved_by_rest: yes

Message: "I get a stabbing pain in my chest while watching TV, nothing to do with activity."
has_chest_pain: yes, brought_on_by_exertion: no, relieved_by_rest: not stated

Message: "Tightness in my chest when I run, and it keeps going after I stop."
has_chest_pain: yes, brought_on_by_exertion: yes, relieved_by_rest: no

Message: "I have chest pain."
has_chest_pain: yes, brought_on_by_exertion: not stated, relieved_by_rest: not stated

Message: "What is angina?"
has_chest_pain: not stated, brought_on_by_exertion: not stated, relieved_by_rest: not stated"""

_CHEST_TOPIC = re.compile(r"chest|angina", re.IGNORECASE)
_DENIAL = r"(?:no|not any|never had|without|don'?t (?:have|get)(?: any)?|do not (?:have|get)(?: any)?)"
_NO_CHEST_PAIN = re.compile(
    rf"\b{_DENIAL}\s+chest\s+(?:pain|discomfort|pressure|tightness)"
    rf"|\b{_DENIAL}\s+(?:pain|discomfort|pressure|tightness)\s+in\s+(?:my|the)\s+chest",
    re.IGNORECASE,
)
_SEX_WORDS = (
    (1, re.compile(r"\b(male|man|boy|gentleman)\b", re.IGNORECASE)),
    (0, re.compile(r"\b(female|woman|girl|lady)\b", re.IGNORECASE)),
)


def classify_chest_pain(facts: ChestPainFacts) -> dict[str, int]:
    """Standard clinical rule: chest pain that is both provoked by exertion and
    relieved by rest is typical angina; one of the two is atypical; neither is
    non-anginal. Returns nothing unless the facts are stated."""
    if facts.has_chest_pain == "no":
        return {"chest_pain_type": 4, "exercise_induced_angina": 0}
    if facts.has_chest_pain != "yes":
        return {}
    exertion, rest = facts.brought_on_by_exertion, facts.relieved_by_rest
    result: dict[str, int] = {}
    if exertion != "not stated":
        result["exercise_induced_angina"] = int(exertion == "yes")
    if exertion == "yes" and rest == "yes":
        result["chest_pain_type"] = 1
    elif exertion == "yes" or rest == "yes":
        result["chest_pain_type"] = 2
    elif exertion == "no":
        result["chest_pain_type"] = 3
    return result


_AGE = re.compile(
    r"\b(\d{1,3})[\s-]*(?:years?|yrs?|y/?o)\b|\b(?:i am|i'm|im|aged?)\s+(\d{1,3})\b(?!\s*(?:%|bpm|mg|mm|kg|lb|cm))",
    re.IGNORECASE,
)


_BLOOD_PRESSURE = re.compile(r"(?<![\d/])(\d{2,3})\s*/\s*\d{2,3}(?![\d/])")      # 118/76 -> 118
_CHOLESTEROL = re.compile(
    r"\bcholesterol\b[\s:=]*(?:(?:is|was|of|at|level|reading)\b[\s:=]*)*(\d{2,3})\b", re.IGNORECASE)


def rule_based_extraction(text: str) -> dict[str, int]:
    """Facts that plain wording settles without an LLM."""
    found: dict[str, int] = {}
    matches = [code for code, pattern in _SEX_WORDS if pattern.search(text)]
    if len(matches) == 1:
        found["sex"] = matches[0]
    if _NO_CHEST_PAIN.search(text):
        found["chest_pain_type"] = 4
    ages = {int(first or second) for first, second in _AGE.findall(text)}
    if len(ages) == 1:
        found["age"] = ages.pop()
    pressures = {int(systolic) for systolic in _BLOOD_PRESSURE.findall(text)}
    if len(pressures) == 1:
        found["resting_blood_pressure"] = pressures.pop()
    cholesterol = {int(value) for value in _CHOLESTEROL.findall(text)}
    if len(cholesterol) == 1:
        found["cholesterol"] = cholesterol.pop()
    return found


_BARE_NUMBER = re.compile(r"^[^\d/]*(\d{1,3})(?![\d/.])[^\d/]*$")
_DECLINED = re.compile(
    r"\b(skip|don'?t know|do not know|dont know|not sure|no idea|unknown|n/?a|prefer not|"
    r"never (?:had it )?(?:checked|measured|tested))\b|^\W*(no|nope|none|neither)\W*$",
    re.IGNORECASE,
)


def _answering_chest_question(state: MedAgentState) -> bool:
    """True when the previous turn asked the chest pain question and it is still open."""
    return (
        state.get("awaiting") == "heart_disease"
        and "chest_pain_type" not in (state.get("user_measurements") or {})
    )


def answers_to_follow_up(state: MedAgentState, query: str, found: dict[str, Any]) -> dict[str, Any]:
    """Read a reply to the heart follow-up questions, which is often terse
    ("54", "no", "yes, on the stairs") and only makes sense given what was asked."""
    if state.get("awaiting") != "heart_disease":
        return {}
    known = {**(state.get("user_measurements") or {}), **found}
    answers: dict[str, Any] = {}
    # A lone number answers "How old are you?" when age was the open question.
    if "age" not in known:
        bare = _BARE_NUMBER.match(query)
        if bare:
            answers["age"] = int(bare.group(1))
    if "chest_pain_type" not in known:
        if _NO_CHEST_PAIN.search(query):
            answers["chest_pain_type"] = 4
        else:
            try:
                facts = get_llm().with_structured_output(ChestPainFacts).invoke([
                    SystemMessage(content=CHEST_PAIN_PROMPT),
                    HumanMessage(content=(
                        f'The patient was asked: "{HEART_REQUIRED_QUESTIONS["chest_pain_type"]}"\n'
                        f'Message: "{query}"'
                    )),
                ])
                if isinstance(facts, ChestPainFacts):
                    answers.update(classify_chest_pain(facts))
            except Exception:
                logger.exception("Could not read the chest pain answer; it will be asked again")
    return answers


_WORTH_EXTRACTING = re.compile(
    r"\d|" + "|".join(_TOPIC_EVIDENCE[k] for k in _TOPIC_EVIDENCE if k != "sex"), re.IGNORECASE
)


def extraction_node(state: MedAgentState) -> dict[str, Any]:
    """Parse measurements from the patient's own words."""
    query = state["raw_user_query"]
    heart: dict[str, Any] = {}
    breast: dict[str, Any] = {}
    if _WORTH_EXTRACTING.search(query):
        try:
            extracted = get_llm().with_structured_output(ExtractedInputs).invoke([
                SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
                HumanMessage(content=f'Patient message:\n"{query}"'),
            ])
            if isinstance(extracted, ExtractedInputs):
                heart, breast = ground_extraction(extracted, query)
        except Exception:
            logger.exception("Measurement extraction failed; continuing without it")

    # Small local models reliably pull out numbers but tend to leave the
    # categorical fields empty, so chest pain is asked as three yes/no facts
    # and mapped to a type by rule.
    if (
        "chest_pain_type" not in heart and _CHEST_TOPIC.search(query)
        and not _NO_CHEST_PAIN.search(query) and not _answering_chest_question(state)
    ):
        try:
            facts = get_llm().with_structured_output(ChestPainFacts).invoke([
                SystemMessage(content=CHEST_PAIN_PROMPT),
                HumanMessage(content=f'Message: "{query}"'),
            ])
            if isinstance(facts, ChestPainFacts):
                heart = {**classify_chest_pain(facts), **heart}
        except Exception:
            logger.exception("Chest pain classification failed; continuing without it")

    heart = {**rule_based_extraction(query), **heart}
    heart = {**answers_to_follow_up(state, query, heart), **heart}
    heart, _ = validate_heart_measurements(heart)
    return {
        "user_measurements": heart,
        "breast_cancer_features": breast,
        "new_measurements": sorted(heart),
    }


# ── Planning ──────────────────────────────────────────────────────────────────

class Intent(BaseModel):
    heart: bool = Field(description=(
        "The patient describes THEIR OWN heart symptoms or cardiac measurements, or asks "
        "about their own heart risk"))
    breast: bool = Field(description=(
        "The patient asks for an assessment of THEIR OWN breast concern, mammogram, or "
        "breast pathology results"))
    skin: bool = Field(description=(
        "The patient asks for an assessment of THEIR OWN skin lesion, mole, or skin photo"))
    health_question: bool = Field(description=(
        "The message is any health or medical question or concern (false for greetings, "
        "thanks, small talk, or unrelated topics)"))


PLANNER_SYSTEM_PROMPT = """You classify the latest message from a patient using a medical app.
Set each flag strictly according to its description. A general knowledge question such as
"what causes heart disease?" is a health_question but NOT a request to assess the patient's own heart.
Greetings and thanks set every flag to false."""

_KEYWORDS = {
    "heart": r"\b(heart|cardiac|chest pain|angina|blood pressure|cholesterol|palpitation)",
    "breast": r"\b(breast|mammogra|lump)",
    "skin": r"\b(skin|mole|rash|lesion|melanoma|spot on my)",
}
_SMALL_TALK = re.compile(
    r"^\W*(hi|hello|hey|thanks|thank you|thx|ok|okay|bye|goodbye|good (morning|afternoon|evening))"
    r"( there| so much| very much| a lot)?\W*$",
    re.IGNORECASE,
)


def _keyword_intent(text: str) -> dict[str, bool]:
    lowered = text.lower()
    intent = {name: re.search(pattern, lowered) is not None for name, pattern in _KEYWORDS.items()}
    intent["health_question"] = not _SMALL_TALK.match(text)
    return intent


def _recent_history(state: MedAgentState) -> list[BaseMessage]:
    """Earlier messages of this conversation (excludes the current one)."""
    return list(state.get("messages", []))[:-1][-HISTORY_MESSAGES:]


def plan_node(state: MedAgentState) -> dict[str, Any]:
    """Decide which specialists run this turn. Runs once per turn."""
    query = state["raw_user_query"]
    if _SMALL_TALK.match(query):
        intent = dict.fromkeys(("heart", "breast", "skin", "health_question"), False)
    else:
        # Only the latest message is classified. Showing the model the earlier
        # reports made it request the same specialist again on "thanks".
        try:
            result = get_llm().with_structured_output(Intent).invoke([
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ])
            intent = result.model_dump() if isinstance(result, Intent) else _keyword_intent(query)
        except Exception:
            logger.exception("Intent classification failed; using keyword fallback")
            intent = _keyword_intent(query)

    new_images = set(state.get("new_images") or [])
    files = state.get("file_references") or {}
    measurements = state.get("user_measurements") or {}
    breast_features = state.get("breast_cancer_features") or {}

    new_measurements = state.get("new_measurements") or []
    small_talk = not any(intent.values())
    # A reply to open follow-up questions: it supplied something, or it is short
    # ("no", "skip", "yes on the stairs"). A long unrelated question is not.
    # (The classifier calls a bare "58" small talk, so only an actual greeting
    # or thanks is excluded here.)
    answering = (
        state.get("awaiting") == "heart_disease" and not _SMALL_TALK.match(query)
        and (bool(new_measurements) or len(query.split()) <= 12)
    )
    wanted = {
        # Heart: when the patient asks about their own heart (they are asked
        # follow-up questions until there is enough to estimate), is answering
        # those questions, or states a cardiac measurement that completes the set.
        "heart_disease": intent["heart"] or answering or (
            bool(set(new_measurements) - {"age", "sex"}) and heart_inputs_sufficient(measurements)
        ),
        # Images: a new upload is always assessed; an earlier one only when asked about.
        "breast_cancer": "breast_cancer" in new_images
        or (intent["breast"] and bool(files.get("breast_scan_path") or breast_features)),
        "skin_disease": "skin_disease" in new_images
        or (intent["skin"] and bool(files.get("skin_photo_path"))),
    }
    pending = [name for name in SPECIALIST_ORDER if wanted[name]]
    update: dict[str, Any] = {"pending": pending, "intent": intent, "consulted": []}
    if state.get("awaiting") and state["awaiting"] not in pending and not small_talk:
        update["awaiting"] = None          # the patient moved on to something else
    return update


def route_next(state: MedAgentState) -> str:
    consulted = state.get("consulted") or []
    for name in state.get("pending") or []:
        if name not in consulted:
            return name
    if consulted:
        return "synthesis"
    if (state.get("intent") or {}).get("health_question"):
        return "general"
    return "small_talk"


# ── Specialists ───────────────────────────────────────────────────────────────

NO_RESULT = {
    "insufficient data", "more information needed", "model unavailable", "image not assessed", "no input provided",
}

_EXPLAINER_RULES = (
    "You are an explanation layer, NOT a diagnostic model. A separate ML model produced the "
    "result below. In 2-3 plain sentences, explain to the patient what that result means and "
    "what a sensible next step is. Do not question, redo, or replace the result. Do not "
    "describe an image; you cannot see one. NEVER state or estimate any number, percentage, "
    "probability, confidence, likelihood or risk level."
)


def _explain(topic: str, analysis: dict[str, Any], extra: str = "") -> str | None:
    """Plain-language explanation of a result the model actually produced."""
    if analysis["diagnosis"] in NO_RESULT:
        return None
    try:
        response = get_llm().invoke([
            SystemMessage(content=f"{_EXPLAINER_RULES} Topic: {topic}."),
            HumanMessage(content=f"ML result: {analysis['diagnosis']}\n{extra}".strip()),
        ])
        return sanitize_assessment(response.content)
    except Exception:
        logger.exception("Explanation LLM failed for %s", topic)
        return None


def _unavailable(error: Exception) -> dict[str, Any]:
    logger.error("Model unavailable: %s", error)
    return {
        "diagnosis": "model unavailable",
        "confidence": None,
        "summary": "This assessment is not available right now.",
        "details": "The model for this assessment is not installed on the server.",
    }


def _step(title: str, analysis: dict[str, Any], explanation: str | None) -> str:
    if analysis.get("questions"):
        return f"**{title}:**\n- Result: {analysis['diagnosis']}\n- Asked: " + " | ".join(analysis["questions"])
    lines = [f"**{title}:**", f"- Result: {analysis['diagnosis']}", f"- {analysis['summary']}",
             f"- Details: {analysis['details']}"]
    if explanation:
        lines += ["", f"**Explanation:** {explanation}"]
    return "\n".join(lines)


def _finish(state: MedAgentState, name: str, title: str, analysis: dict[str, Any],
            explanation: str | None, explanation_key: str) -> dict[str, Any]:
    return {
        f"{name}_analysis": analysis,
        explanation_key: explanation,
        "consulted": [*(state.get("consulted") or []), name],
        "steps": [*(state.get("steps") or []), _step(title, analysis, explanation)],
    }


def heart_disease_node(state: MedAgentState) -> dict[str, Any]:
    measurements = state.get("user_measurements") or {}
    helpful_asked = bool(state.get("heart_helpful_asked"))
    questions, stage = heart_follow_up_questions(measurements, helpful_asked)
    was_asked = state.get("awaiting") == "heart_disease"
    declined = was_asked and _DECLINED.search(state["raw_user_query"]) is not None
    follow_up: dict[str, Any] = {"awaiting": None}

    if questions and stage == "required" and declined and not state.get("new_measurements"):
        # They can't or won't answer something the model cannot do without.
        analysis = heart_insufficient_data(sorted(measurements))
    elif questions:
        analysis = heart_more_information_needed(measurements, questions, stage)
        follow_up = {"awaiting": "heart_disease"}
        if stage == "helpful":
            follow_up["heart_helpful_asked"] = True
    else:
        try:
            analysis = predict_heart_disease(measurements)
        except FileNotFoundError as error:
            analysis = _unavailable(error)
    provided = analysis.get("features_provided", [])
    missing = analysis.get("features_missing", [])
    explanation = _explain(
        "heart disease likelihood", analysis,
        f"Measurements the patient gave: {', '.join(provided)}\n"
        f"Measurements not available: {', '.join(missing) or 'none'}\n"
        "If measurements are not available, say the result is preliminary.",
    )
    return {
        **_finish(state, "heart_disease", "Heart Disease Analysis", analysis, explanation, "heart_diagnosis"),
        **follow_up,
    }


def breast_cancer_node(state: MedAgentState) -> dict[str, Any]:
    image_path = (state.get("file_references") or {}).get("breast_scan_path")
    try:
        analysis = predict_breast_cancer(
            image_path=image_path, features=state.get("breast_cancer_features") or {},
        )
    except FileNotFoundError as error:
        analysis = _unavailable(error)
    if analysis["diagnosis"] == "flagged for follow-up":
        guidance = (
            "A flag means a radiologist should review the image. It is not a cancer "
            "diagnosis and most flags turn out to be nothing serious."
        )
    else:
        guidance = (
            "Do NOT tell the patient this is good news, that they are healthy, or that the "
            "model was accurate. A screening model can miss cancers. Say only that nothing "
            "was flagged, that routine screening should continue, and that any lump, pain "
            "or change should still be checked by a doctor."
        )
    explanation = _explain("breast screening", analysis, guidance)
    return _finish(state, "breast_cancer", "Breast Cancer Analysis", analysis, explanation, "cancer_diagnosis")


def skin_disease_node(state: MedAgentState) -> dict[str, Any]:
    image_path = (state.get("file_references") or {}).get("skin_photo_path")
    if not image_path:
        analysis = {
            "diagnosis": "no input provided",
            "confidence": None,
            "summary": "No skin assessment was run.",
            "details": "No skin photo was provided. Attach a clear, close-up colour photo.",
        }
    else:
        try:
            analysis = predict_skin_disease(image_path)
        except FileNotFoundError as error:
            analysis = _unavailable(error)
    explanation = _explain(
        "skin lesion classification", analysis,
        "Needs prompt medical review: " + ("yes" if analysis.get("needs_prompt_review") else "no"),
    )
    return _finish(state, "skin_disease", "Skin Disease Analysis", analysis, explanation, "skin_diagnosis")


# ── Final reply nodes ─────────────────────────────────────────────────────────

CLINICAL_DISCLAIMER = (
    "**⚕️ CLINICAL DISCLAIMER:** This is an automated preliminary assessment, "
    "not a formal diagnosis. The analysis was generated by AI models and should "
    "not be used as the basis for medical decisions. Please consult a qualified "
    "healthcare professional for definitive evaluation and treatment decisions."
)

NO_ANALYSIS_REPLY = (
    "Hello! I can answer general health questions, estimate heart disease likelihood from "
    "your measurements, screen a mammogram, and classify a photo of a skin lesion. Tell me "
    "what's going on, or attach an image and choose its type."
)

ASK_SKIN_PHOTO = (
    "**To have the skin model look at it:** can you attach a clear, close-up colour photo of "
    "the spot (use the paperclip or camera button) and mark it as \"Skin photo\"?"
)
ASK_MAMMOGRAM = (
    "**To have the screening model look at it:** can you attach your mammogram image (use the "
    "paperclip button) and mark it as \"Mammogram\"?"
)

THANKS_REPLY = "You're welcome. Tell me if there's anything else you'd like to ask or check."
GOODBYE_REPLY = "Take care. If your symptoms change or worsen, contact a healthcare professional."

GENERAL_UNAVAILABLE_REPLY = (
    "I couldn't generate an answer just now. Please try again in a moment, and contact a "
    "healthcare professional if your question is urgent."
)

_CARDS = (
    ("heart_disease", "## ❤️ Heart Disease Assessment", "heart_diagnosis"),
    ("breast_cancer", "## 🎗️ Breast Screening", "cancer_diagnosis"),
    ("skin_disease", "## 🩺 Skin Lesion Analysis", "skin_diagnosis"),
)


def _reply(state: MedAgentState, body: str, routed: str) -> dict[str, Any]:
    notice = emergency_notice(state.get("red_flags") or [])
    content = f"{notice}\n\n---\n\n{body}" if notice else body
    return {"messages": [AIMessage(content=content)], "next_agent": routed}


def synthesis_node(state: MedAgentState) -> dict[str, Any]:
    """Compile this turn's specialist results into one report."""
    cards = []
    for name, heading, explanation_key in _CARDS:
        if name not in (state.get("consulted") or []):
            continue
        analysis = state[f"{name}_analysis"]
        if analysis.get("questions"):
            numbered = "\n".join(f"{n}. {q}" for n, q in enumerate(analysis["questions"], 1))
            cards.append("\n\n".join([heading, analysis["summary"], numbered, analysis["details"]]))
            continue
        parts = [heading, f"**Result:** {analysis['diagnosis']}", analysis["summary"]]
        if state.get(explanation_key):
            parts.append(state[explanation_key])
        parts.append(f"*About this result:* {analysis['details']}")
        cards.append("\n\n".join(parts))
    return _reply(state, "\n\n---\n\n".join(cards), ",".join(state.get("consulted") or []))


GENERAL_SYSTEM_PROMPT = """You are MediChat, a careful health information assistant for the general public.
- Give clear, general health information in plain language, in a few short paragraphs.
- You cannot examine the patient. Do not diagnose, and do not tell them to start, stop or change a medication.
- Do not state probabilities or percentages about this patient's own condition.
- Say when something should be checked by a clinician, and how soon.
- If the patient wants one of the app's model assessments, tell them what to provide: for heart, their age, sex, type of chest pain and measurements such as blood pressure and cholesterol; for breast, a mammogram image attached as "Mammogram"; for skin, a close-up colour photo attached as "Skin photo".
- You are NOT one of the app's assessment models. Never write an "assessment", a "Result:", a risk level (such as "increased risk" or "low risk") or a verdict about this patient, and never imitate the format of an earlier assessment report.
- If the message is not about health, say briefly that you can only help with health topics."""

REPORT_PLACEHOLDER = "(The app showed the patient a model assessment report here.)"
_LOOKS_LIKE_ASSESSMENT = re.compile(
    r"\*\*\s*result\s*:?\s*\*\*|^\s*#+\s.*\b(assessment|analysis|screening)\b"
    r"|\b(increased|elevated|high|low|lower|moderate)\s+risk\s*$",
    re.IGNORECASE | re.MULTILINE,
)
GENERAL_REFUSED_ASSESSMENT_REPLY = (
    "I can't assess your personal risk in a general answer. For a heart estimate, tell me "
    "your age, sex, whether you get chest pain (and whether effort brings it on and rest "
    "relieves it), plus measurements such as blood pressure and cholesterol. For a "
    "mammogram or a skin lesion, attach the image and choose its type. For anything that "
    "worries you, a healthcare professional can examine you properly."
)


def _history_for_general(state: MedAgentState) -> list[BaseMessage]:
    """Recent conversation with model reports replaced by a placeholder, so the
    LLM has nothing to copy an assessment from."""
    history = []
    for message in _recent_history(state):
        text = message.content if isinstance(message.content, str) else ""
        if isinstance(message, AIMessage) and ("**Result:**" in text or text.startswith("## ")):
            history.append(AIMessage(content=REPORT_PLACEHOLDER))
        else:
            history.append(message)
    return history


def general_node(state: MedAgentState) -> dict[str, Any]:
    """Answer a general health question when no specialist model applies."""
    context = state.get("patient_context")
    system = GENERAL_SYSTEM_PROMPT
    if context:
        system += f"\n\nIntake form the patient filled in (self-reported):\n{context}"
    if state.get("heart_rate"):
        system += f"\nResting heart rate entered: {state['heart_rate']} bpm"
    try:
        response = get_llm().invoke([
            SystemMessage(content=system),
            *_history_for_general(state),
            HumanMessage(content=state["raw_user_query"]),
        ])
        body = response.content.strip() if isinstance(response.content, str) else ""
        if _LOOKS_LIKE_ASSESSMENT.search(body):
            logger.warning("General answer imitated an assessment; replaced")
            body = GENERAL_REFUSED_ASSESSMENT_REPLY
    except Exception:
        logger.exception("General answer LLM failed")
        body = ""
    body = body or GENERAL_UNAVAILABLE_REPLY
    # If they asked about their own skin or breast concern without an image,
    # ask for it explicitly rather than leaving that to the LLM.
    intent = state.get("intent") or {}
    files = state.get("file_references") or {}
    if intent.get("skin") and not files.get("skin_photo_path"):
        body += "\n\n" + ASK_SKIN_PHOTO
    if intent.get("breast") and not files.get("breast_scan_path"):
        body += "\n\n" + ASK_MAMMOGRAM
    return _reply(state, body, "general")


def small_talk_node(state: MedAgentState) -> dict[str, Any]:
    query = state["raw_user_query"].lower()
    if re.search(r"\b(thanks|thank you|thx)\b", query):
        body = THANKS_REPLY
    elif re.search(r"\b(bye|goodbye)\b", query):
        body = GOODBYE_REPLY
    else:
        body = NO_ANALYSIS_REPLY
    return _reply(state, body, "small_talk")


# ── Graph ─────────────────────────────────────────────────────────────────────

def compile_medagent_graph(checkpointer: Any | None = None):
    """Build and compile the MedAgent graph (see module docstring for the flow)."""
    graph = StateGraph(state_schema=MedAgentState)

    graph.add_node("Extract", extraction_node)
    graph.add_node("Plan", plan_node)
    graph.add_node("HeartDiseaseNode", heart_disease_node)
    graph.add_node("BreastCancerNode", breast_cancer_node)
    graph.add_node("SkinDiseaseNode", skin_disease_node)
    graph.add_node("Synthesis", synthesis_node)
    graph.add_node("General", general_node)
    graph.add_node("SmallTalk", small_talk_node)

    graph.set_entry_point("Extract")
    graph.add_edge("Extract", "Plan")

    routes = {
        "heart_disease": "HeartDiseaseNode",
        "breast_cancer": "BreastCancerNode",
        "skin_disease": "SkinDiseaseNode",
        "synthesis": "Synthesis",
        "general": "General",
        "small_talk": "SmallTalk",
    }
    for source in ("Plan", "HeartDiseaseNode", "BreastCancerNode", "SkinDiseaseNode"):
        graph.add_conditional_edges(source, route_next, routes)
    for final in ("Synthesis", "General", "SmallTalk"):
        graph.add_edge(final, END)

    return graph.compile(checkpointer=checkpointer)
