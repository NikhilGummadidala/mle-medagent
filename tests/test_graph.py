"""Conversation graph: extraction grounding, routing, state and safety."""
from __future__ import annotations

import uuid

import pytest

import langgraph_multi_agent as g
from api import graph_runner
from safety import emergency_notice, scan_red_flags
from tests.conftest import png_bytes

HEART_ONLY = g.Intent(heart=True, breast=False, skin=False, health_question=True)
QUESTION = g.Intent(heart=False, breast=False, skin=False, health_question=True)
NOTHING = g.Intent(heart=False, breast=False, skin=False, health_question=False)


def run(message, **kwargs):
    kwargs.setdefault("conversation_id", uuid.uuid4().hex)
    return graph_runner.run_graph(user_id=1, message=message, **kwargs)


# ── extraction grounding ──────────────────────────────────────────────────────

def test_grounding_keeps_only_what_the_patient_said():
    text = "I'm a 58 year old man, BP 150/95, cholesterol 260. Chest pain when I climb stairs."
    extracted = g.ExtractedInputs(
        age=58, sex="male", resting_blood_pressure=150, cholesterol=260,
        chest_pain_type="typical angina", exercise_induced_angina=True,
        max_heart_rate=140,          # invented: 140 is not in the message
        oldpeak=0, ca=0,             # zero-filled, never mentioned
        thal="normal",               # never mentioned
        fasting_blood_sugar_above_120=False,
    )
    heart, breast = g.ground_extraction(extracted, text)
    assert heart == {
        "age": 58, "sex": 1, "resting_blood_pressure": 150, "cholesterol": 260,
        "chest_pain_type": 1, "exercise_induced_angina": 1,
    }
    assert breast == {}


def test_grounding_rejects_out_of_range_values():
    heart, _ = g.ground_extraction(g.ExtractedInputs(age=580, cholesterol=260), "age 580, cholesterol 260")
    assert heart == {"cholesterol": 260}


def test_extraction_failure_does_not_break_the_turn(fake_llm):
    fake_llm({g.ExtractedInputs: RuntimeError("ollama down"), g.Intent: QUESTION}, text="General answer.")
    result = run("My cholesterol is 240, is that bad?")
    assert result["reply"] == "General answer."


# ── routing ───────────────────────────────────────────────────────────────────

def test_greeting_gets_the_canned_reply_without_running_models(fake_llm):
    llm = fake_llm({g.Intent: NOTHING})
    result = run("hello")
    assert result["routed_to"] == "small_talk"
    assert result["reply"] == g.NO_ANALYSIS_REPLY
    assert not result["heart_disease"] and len(result["messages"]) == 1
    assert "text" not in llm.calls


def test_general_health_question_is_answered(fake_llm):
    fake_llm({g.Intent: QUESTION}, text="Iron-rich foods include lentils and spinach.")
    result = run("What foods are high in iron?", age=30, sex=0)
    assert result["routed_to"] == "general"
    assert "lentils" in result["reply"]
    assert not result["heart_disease"]      # intake data alone never triggers a heart score


def test_heart_question_with_only_intake_data_asks_for_more(fake_llm):
    llm = fake_llm({g.Intent: HEART_ONLY})
    result = run("Am I at risk of heart disease?", age=60, sex=1, resting_blood_pressure=150)
    assert result["heart_disease"]["diagnosis"] == "more information needed"
    assert "Do you get chest pain" in result["reply"] and "**Result:**" not in result["reply"]
    assert "text" not in llm.calls           # nothing to explain, so no LLM explanation


def test_heart_assessment_runs_once_and_is_reported(fake_llm):
    extracted = g.ExtractedInputs(
        age=63, sex="male", chest_pain_type="asymptomatic", resting_blood_pressure=145,
        cholesterol=233,
    )
    llm = fake_llm({g.ExtractedInputs: extracted, g.Intent: HEART_ONLY})
    result = run("63 year old male, no chest pain (asymptomatic), BP 145, cholesterol 233. Heart risk?")
    assert result["routed_to"] == "heart_disease"
    assert result["heart_disease"]["diagnosis"].endswith("likelihood of heart disease")
    assert "5 of 13" in result["reply"] and "Plain explanation." in result["reply"]
    assert llm.calls.count("Intent") == 1 and llm.calls.count("text") == 1
    assert [m["content"].startswith("**Heart Disease Analysis:**") for m in result["messages"]] == [True, False]


def test_llm_planner_failure_falls_back_to_keywords(fake_llm):
    fake_llm({g.Intent: RuntimeError("bad json")})
    assert run("hi")["routed_to"] == "small_talk"
    assert run("Is my heart ok?", age=50, sex=1)["routed_to"] == "heart_disease"


def test_explanation_with_invented_numbers_is_replaced(fake_llm):
    extracted = g.ExtractedInputs(age=63, sex="male", chest_pain_type="asymptomatic", cholesterol=233)
    fake_llm({g.ExtractedInputs: extracted, g.Intent: HEART_ONLY}, text="You have a 90% chance of disease.")
    result = run("63 male asymptomatic chest pain status, cholesterol 233")
    assert "90%" not in result["reply"]
    assert g.SAFE_ASSESSMENT_FALLBACK in result["reply"]


# ── images ────────────────────────────────────────────────────────────────────

def test_new_image_is_assessed_even_if_the_planner_misses_it(fake_llm, tmp_path, monkeypatch):
    fake_llm({g.Intent: NOTHING})
    seen = {}

    def fake_predict(path):
        seen["path"] = path
        return {"diagnosis": "melanocytic nevus (mole)", "confidence": 0.9,
                "summary": "Closest match: mole.", "details": "d", "needs_prompt_review": False}

    monkeypatch.setattr(g, "predict_skin_disease", fake_predict)
    photo = tmp_path / "p.png"
    photo.write_bytes(png_bytes())
    result = run("what is this", skin_photo_path=str(photo))
    assert seen["path"] == str(photo)
    assert result["skin_disease"]["diagnosis"] == "melanocytic nevus (mole)"
    assert "Skin Lesion Analysis" in result["reply"]


def test_missing_model_is_reported_without_asking_the_llm(fake_llm, tmp_path, monkeypatch):
    llm = fake_llm({g.Intent: NOTHING})

    def missing(path):
        raise FileNotFoundError(r"C:\secret\server\path\model.safetensors")

    monkeypatch.setattr(g, "predict_skin_disease", missing)
    result = run("check this", skin_photo_path=str(tmp_path / "x.png"))
    assert result["skin_disease"]["diagnosis"] == "model unavailable"
    assert "secret" not in result["reply"]          # no server paths in the reply
    assert "text" not in llm.calls                   # the LLM never stands in for the model


# ── state across turns ────────────────────────────────────────────────────────

def test_measurements_accumulate_across_turns(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.ExtractedInputs: g.ExtractedInputs(chest_pain_type="non-anginal pain"), g.Intent: HEART_ONLY})
    first = run("I get chest pain that isn't related to exertion", conversation_id=conversation, age=52, sex=0)
    assert first["heart_disease"]["diagnosis"] == "more information needed"   # asks BP/cholesterol first

    fake_llm({g.ExtractedInputs: g.ExtractedInputs(cholesterol=210), g.Intent: HEART_ONLY})
    second = run("My cholesterol is 210", conversation_id=conversation)
    assert second["heart_disease"]["diagnosis"].endswith("likelihood of heart disease")
    assert "4 of 13" in second["reply"]


def test_each_turn_reports_only_its_own_results(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.ExtractedInputs: g.ExtractedInputs(chest_pain_type="asymptomatic", cholesterol=250), g.Intent: HEART_ONLY})
    run("no chest pain, asymptomatic, cholesterol 250", conversation_id=conversation, age=60, sex=1)
    fake_llm({g.Intent: NOTHING})
    second = run("thanks", conversation_id=conversation)
    assert second["reply"] == g.THANKS_REPLY and not second["heart_disease"]
    history = graph_runner.get_history(1, conversation)
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "no chest pain, asymptomatic, cholesterol 250"


def test_conversations_are_separate_per_user(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: NOTHING})
    run("hello", conversation_id=conversation)
    assert graph_runner.get_history(2, conversation) == []


# ── safety ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("message, key", [
    ("I have crushing pain in my chest", "chest"),
    ("my dad can't breathe", "breath"),
    ("I want to kill myself", "selfharm"),
])
def test_red_flags_detected(message, key):
    assert [flag["key"] for flag in scan_red_flags(message)] == [key]


def test_informational_questions_are_not_flagged():
    assert scan_red_flags("What is a stroke?") == []
    assert scan_red_flags("What are symptoms of a heart attack?") == []


def test_emergency_notice_leads_the_reply(fake_llm):
    fake_llm({g.Intent: QUESTION}, text="Here is some information.")
    result = run("I have chest pain and my arm is numb")
    assert result["reply"].startswith("**🚨 This may be an emergency")
    assert "911" in result["reply"] and "Here is some information." in result["reply"]


def test_red_flags_reported_by_the_intake_form_are_honoured():
    context = "Emergency Screening: RED FLAGS REPORTED, patient was advised to call 911 and chose to continue — Heavy bleeding that won't stop"
    flags = scan_red_flags("what should I do", context)
    assert flags and "911" in emergency_notice(flags)


def test_sanitizer_blocks_numeric_claims_but_not_plain_wording():
    assert g.sanitize_assessment("There is a 73% chance.") == g.SAFE_ASSESSMENT_FALLBACK
    assert g.sanitize_assessment("It is highly likely benign.") == g.SAFE_ASSESSMENT_FALLBACK
    assert g.sanitize_assessment(None) == g.SAFE_ASSESSMENT_FALLBACK
    text = "This result suggests a lower risk; please discuss it with your doctor."
    assert g.sanitize_assessment(text) == text


# ── rule-based extraction (found necessary in live runs with llama3.2) ────────

def facts(pain, exertion="not stated", rest="not stated"):
    return g.ChestPainFacts(has_chest_pain=pain, brought_on_by_exertion=exertion, relieved_by_rest=rest)


@pytest.mark.parametrize("given, expected", [
    (facts("yes", "yes", "yes"), {"chest_pain_type": 1, "exercise_induced_angina": 1}),
    (facts("yes", "yes", "no"), {"chest_pain_type": 2, "exercise_induced_angina": 1}),
    (facts("yes", "no"), {"chest_pain_type": 3, "exercise_induced_angina": 0}),
    (facts("no"), {"chest_pain_type": 4, "exercise_induced_angina": 0}),
    (facts("yes"), {}),                      # pain mentioned, character unknown: do not guess
    (facts("not stated"), {}),
])
def test_chest_pain_type_follows_the_clinical_rule(given, expected):
    assert g.classify_chest_pain(given) == expected


def test_plain_wording_is_read_without_the_llm():
    assert g.rule_based_extraction("I'm a 61 year old man") == {"sex": 1, "age": 61}
    assert g.rule_based_extraction("58-year-old woman, I have no chest pain at all") == {
        "sex": 0, "chest_pain_type": 4, "age": 58}
    assert g.rule_based_extraction("I am 50 and my chest feels tight") == {"age": 50}
    assert g.rule_based_extraction("my husband is a man of 60 years and I am a woman") == {"age": 60}
    assert g.rule_based_extraction("I'm 72 bpm at rest") == {}
    assert g.rule_based_extraction("My cholesterol is 190 and my blood pressure is 118/76.") == {
        "resting_blood_pressure": 118, "cholesterol": 190}
    assert g.rule_based_extraction("born 12/05/1980, total cholesterol: 245") == {"cholesterol": 245}
    assert g.rule_based_extraction("BP was 150/95 last week and 130/85 today") == {}   # ambiguous


def test_chest_pain_facts_feed_the_heart_model(fake_llm):
    fake_llm({g.ExtractedInputs: g.ExtractedInputs(cholesterol=270),
              g.ChestPainFacts: facts("yes", "yes", "yes"), g.Intent: HEART_ONLY})
    result = run("I'm a 61 year old man, cholesterol 270, pressure in my chest on stairs that eases with rest")
    assert result["heart_disease"]["diagnosis"].endswith("likelihood of heart disease")
    assert "5 of 13" in result["reply"]


def test_thanks_after_a_photo_does_not_repeat_the_analysis(fake_llm, tmp_path, monkeypatch):
    conversation = uuid.uuid4().hex
    always_skin = g.Intent(heart=False, breast=False, skin=True, health_question=True)
    fake_llm({g.Intent: always_skin})
    monkeypatch.setattr(g, "predict_skin_disease", lambda path: {
        "diagnosis": "melanoma", "confidence": 0.9, "summary": "s", "details": "d",
        "needs_prompt_review": True})
    photo = tmp_path / "p.png"
    photo.write_bytes(png_bytes())
    run("look at this", conversation_id=conversation, skin_photo_path=str(photo))
    again = run("Thanks so much!", conversation_id=conversation)
    assert again["routed_to"] == "small_talk" and not again["skin_disease"]
    followup = run("what did my skin photo show?", conversation_id=conversation)
    assert followup["skin_disease"]["diagnosis"] == "melanoma"   # asked about it: re-assessed
