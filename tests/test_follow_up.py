"""When there is not enough to go on, the assistant asks follow-up questions
before it gives an assessment, and understands the (often terse) answers."""
from __future__ import annotations

import uuid

import langgraph_multi_agent as g
from api import graph_runner
from models import heart_disease

HEART = g.Intent(heart=True, breast=False, skin=False, health_question=True)
QUESTION = g.Intent(heart=False, breast=False, skin=False, health_question=True)


def facts(pain, exertion="not stated", rest="not stated"):
    return g.ChestPainFacts(has_chest_pain=pain, brought_on_by_exertion=exertion, relieved_by_rest=rest)


def run(message, conversation, **kwargs):
    return graph_runner.run_graph(user_id=1, message=message, conversation_id=conversation, **kwargs)


def asked(result):
    return result["heart_disease"].get("questions", [])


# ── which questions ───────────────────────────────────────────────────────────

def test_questions_cover_exactly_what_is_missing():
    questions, stage = heart_disease.follow_up_questions({})
    assert stage == "required" and len(questions) == 3

    questions, stage = heart_disease.follow_up_questions({"age": 50, "sex": 1})
    assert stage == "required" and questions == [heart_disease.REQUIRED_QUESTIONS["chest_pain_type"]]

    questions, stage = heart_disease.follow_up_questions({"age": 50, "sex": 1, "chest_pain_type": 4})
    assert stage == "helpful" and len(questions) == 2

    assert heart_disease.follow_up_questions({"age": 50, "sex": 1, "chest_pain_type": 4}, True) == ([], None)
    assert heart_disease.follow_up_questions(
        {"age": 50, "sex": 1, "chest_pain_type": 4, "cholesterol": 220}) == ([], None)


# ── the conversation ──────────────────────────────────────────────────────────

def test_no_information_at_all_asks_everything_then_estimates(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: HEART})
    first = run("Am I at risk of heart disease?", conversation)
    assert first["heart_disease"]["diagnosis"] == "more information needed"
    assert len(asked(first)) == 3 and "How old are you?" in first["reply"]
    assert "**Result:**" not in first["reply"]

    # terse answers, planner unhelpful: the open questions still route the reply
    fake_llm({g.Intent: QUESTION, g.ChestPainFacts: facts("yes", "yes", "yes")})
    second = run("I'm 58, male. Yes, when I climb stairs, and it eases when I rest", conversation)
    assert second["heart_disease"]["question_stage"] == "helpful"
    assert "blood pressure" in second["reply"] and "skip" in second["reply"]

    fake_llm({g.Intent: QUESTION})
    third = run("BP is 145/90, don't know my cholesterol", conversation)
    assert third["heart_disease"]["diagnosis"].endswith("likelihood of heart disease")
    assert "5 of 13" in third["reply"] and "**Result:**" in third["reply"]


def test_bare_answers_are_understood_in_context(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: HEART})
    assert len(asked(run("is my heart ok?", conversation, sex=0))) == 2      # age + chest pain

    fake_llm({g.Intent: QUESTION, g.ChestPainFacts: facts("not stated")})
    second = run("47", conversation)                                          # just the age
    assert asked(second) == [heart_disease.REQUIRED_QUESTIONS["chest_pain_type"]]

    fake_llm({g.Intent: QUESTION, g.ChestPainFacts: facts("no")})
    third = run("no", conversation)                                           # no chest pain
    assert third["heart_disease"]["question_stage"] == "helpful"

    fake_llm({g.Intent: QUESTION})
    fourth = run("skip", conversation)
    assert fourth["heart_disease"]["diagnosis"] == "lower likelihood of heart disease"
    # age, sex, chest pain type, and "no exercise-induced angina" implied by no chest pain
    assert "4 of 13" in fourth["reply"]


def test_helpful_questions_are_asked_only_once(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: HEART})
    first = run("I have no chest pain. What's my heart risk?", conversation, age=40, sex=1)
    assert first["heart_disease"]["question_stage"] == "helpful"
    second = run("I'd rather not say, what's my heart risk?", conversation)
    assert second["heart_disease"]["diagnosis"].endswith("likelihood of heart disease")
    third = run("and what is my heart risk now?", conversation)
    assert not asked(third)


def test_nothing_is_asked_when_enough_was_given_up_front(fake_llm):
    fake_llm({g.Intent: HEART})
    result = run("60 year old man, no chest pain, cholesterol 250", uuid.uuid4().hex)
    assert not asked(result) and result["heart_disease"]["diagnosis"].endswith("likelihood of heart disease")


def test_declining_a_required_question_ends_the_questioning(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: HEART})
    run("check my heart", conversation, age=50, sex=1)
    fake_llm({g.Intent: QUESTION, g.ChestPainFacts: facts("not stated")})
    result = run("I don't know", conversation)
    assert result["heart_disease"]["diagnosis"] == "insufficient data"
    fake_llm({g.Intent: QUESTION}, text="Iron answer.")
    assert run("ok", conversation)["routed_to"] != "heart_disease"            # no longer waiting


def test_an_unrelated_question_is_answered_not_hijacked(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: HEART})
    run("check my heart", conversation, age=50, sex=1)
    fake_llm({g.Intent: QUESTION, g.ChestPainFacts: facts("not stated")}, text="Iron answer.")
    result = run(
        "Actually, separately from all that, could you tell me which everyday foods are good sources of iron?",
        conversation,
    )
    assert result["routed_to"] == "general" and result["reply"] == "Iron answer."


def test_thanks_does_not_consume_the_open_questions(fake_llm):
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: HEART})
    run("check my heart", conversation, age=50, sex=1)
    assert run("thanks", conversation)["routed_to"] == "small_talk"
    fake_llm({g.Intent: QUESTION, g.ChestPainFacts: facts("no")})
    assert run("no chest pain at all", conversation)["routed_to"] == "heart_disease"


def test_a_bare_number_is_an_answer_even_if_the_classifier_calls_it_small_talk(fake_llm):
    """Seen live: llama3.2 classified "58" as small talk and the greeting was sent."""
    conversation = uuid.uuid4().hex
    fake_llm({g.Intent: HEART})
    run("Am I at risk of heart disease?", conversation)
    nothing = g.Intent(heart=False, breast=False, skin=False, health_question=False)
    fake_llm({g.Intent: nothing, g.ChestPainFacts: facts("not stated")})
    result = run("58", conversation)
    assert result["routed_to"] == "heart_disease"
    assert result["heart_disease"]["features_provided"] == ["age"] and len(asked(result)) == 2


# ── images ────────────────────────────────────────────────────────────────────

def test_skin_or_breast_concern_without_an_image_asks_for_one(fake_llm):
    fake_llm({g.Intent: g.Intent(heart=False, breast=False, skin=True, health_question=True)}, text="Moles vary.")
    reply = run("I have a mole that changed colour, can you check it?", uuid.uuid4().hex)["reply"]
    assert reply.startswith("Moles vary.") and g.ASK_SKIN_PHOTO in reply

    fake_llm({g.Intent: g.Intent(heart=False, breast=True, skin=False, health_question=True)}, text="Lumps vary.")
    reply = run("can you check my mammogram?", uuid.uuid4().hex)["reply"]
    assert g.ASK_MAMMOGRAM in reply and g.ASK_SKIN_PHOTO not in reply
