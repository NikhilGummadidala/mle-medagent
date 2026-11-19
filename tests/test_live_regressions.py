"""Regressions for faults first seen when running the app against the real
local LLM (llama3.2), which the stubbed tests could not have shown."""
from __future__ import annotations

import uuid

import pytest

import langgraph_multi_agent as g
from api import graph_runner
from safety import scan_red_flags

QUESTION = g.Intent(heart=False, breast=False, skin=False, health_question=True)


def run(message, **kwargs):
    kwargs.setdefault("conversation_id", uuid.uuid4().hex)
    return graph_runner.run_graph(user_id=1, message=message, **kwargs)


@pytest.mark.parametrize("message, expected", [
    ("I don't have chest pain. My cholesterol is 190", []),
    ("I have no chest pain", []),
    ("I do not have any pain in my chest", []),
    ("I'm not suicidal", []),
    ("I have not had a seizure", []),
    # A denial of one thing must never hide a different, real symptom.
    ("No trouble breathing, but my chest hurts a lot", ["chest"]),
    ("I never fainted but I have crushing pain in my chest", ["chest"]),
    ("I don't feel well, my chest is tight", ["chest"]),
    ("no fever but I passed out", ["faint"]),
])
def test_denied_symptoms_do_not_raise_the_emergency_notice(message, expected):
    assert [flag["key"] for flag in scan_red_flags(message)] == expected


def test_stated_measurements_reach_the_heart_model_even_if_the_planner_says_no(fake_llm):
    """The LLM classified this as a general question and then wrote its own
    'Heart Disease Assessment'. Measurements in the message now route to the model."""
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: QUESTION})
    run("Am I at risk of heart disease?", conversation_id=conversation, age=45, sex=0)
    fake_llm({
        g.ExtractedInputs: g.ExtractedInputs(cholesterol=190, resting_blood_pressure=118),
        g.Intent: QUESTION,
    }, text="Plain explanation.")
    result = run(
        "I don't have chest pain. My cholesterol is 190 and my blood pressure is 118/76.",
        conversation_id=conversation,
    )
    assert result["routed_to"] == "heart_disease"
    assert result["heart_disease"]["diagnosis"] == "lower likelihood of heart disease"
    assert "5 of 13" in result["reply"]
    assert "emergency" not in result["reply"].lower()


@pytest.mark.parametrize("imitation", [
    "## ❤️ Heart Disease Assessment\n\n**Result:** increased risk\n\nCholesterol: high (190)",
    "**Result**: low risk",
    "Based on what you said:\n\nincreased risk",
])
def test_general_answer_cannot_pose_as_a_model_assessment(fake_llm, imitation):
    fake_llm({g.Intent: QUESTION}, text=imitation)
    result = run("is my heart ok?")
    assert result["routed_to"] == "general"
    assert result["reply"] == g.GENERAL_REFUSED_ASSESSMENT_REPLY


def test_ordinary_general_answers_pass_through(fake_llm):
    text = "High blood pressure raises the risk of stroke over time.\n\n* Eat less salt\n* Stay active"
    fake_llm({g.Intent: QUESTION}, text=text)
    assert run("why does blood pressure matter?")["reply"] == text


def test_earlier_reports_are_hidden_from_the_general_llm(fake_llm, monkeypatch):
    conversation = uuid.uuid4().hex
    fake_llm({g.ExtractedInputs: g.ExtractedInputs(cholesterol=250, chest_pain_type="asymptomatic"),
              g.Intent: g.Intent(heart=True, breast=False, skin=False, health_question=True)})
    run("no chest pain, asymptomatic, cholesterol 250", conversation_id=conversation, age=60, sex=1)

    seen = []
    llm = fake_llm({g.Intent: QUESTION}, text="An answer.")
    original = llm.invoke

    def spy(messages):
        seen.extend(m.content for m in messages)
        return original(messages)

    monkeypatch.setattr(llm, "invoke", spy)
    run("what is cholesterol?", conversation_id=conversation)
    assert g.REPORT_PLACEHOLDER in seen
    assert not any("**Result:**" in content for content in seen)


def test_thanks_and_goodbye_get_fitting_replies(fake_llm):
    fake_llm()
    assert run("thanks")["reply"] == g.THANKS_REPLY
    assert run("bye")["reply"] == g.GOODBYE_REPLY
    assert run("hello")["reply"] == g.NO_ANALYSIS_REPLY
