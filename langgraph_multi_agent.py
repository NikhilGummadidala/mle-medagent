"""
Multi-agent medical diagnosis system using LangGraph.
Three specialized sub-agents: Heart Disease, Breast Cancer, Skin Disease.
LLM-based supervisor routes queries via message history using Ollama.
ML models provide clinical predictions; LLMs generate natural-language explanations.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, add_messages
from pydantic import BaseModel, Field

from ingestion import encode_image_to_base64, get_image_mime_type
from models.heart_disease import predict as predict_heart_disease
from models.breast_cancer import predict as predict_breast_cancer
from models.skin_disease import predict as predict_skin_disease

# ── Measurement Key Definitions ────────────────────────────────────────────────

USER_MEASUREMENT_KEYS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "ca",           # number of major vessels (0-3) colored by fluoroscopy
    "thal",         # thalassemia type (0=normal, 1=fixed defect, 2=reversible defect)
]

DEFAULT_MEASUREMENTS: dict[str, int | float] = {
    "age": 50,
    "sex": 0,
    "chest_pain_type": 0,
    "resting_blood_pressure": 120,
    "cholesterol": 200,
    "fasting_blood_sugar": 0,
    "resting_ecg": 0,
    "max_heart_rate": 140,
    "exercise_induced_angina": 0,
    "oldpeak": 0.0,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}

# Wisconsin Breast Cancer dataset feature keys (30 features)
BREAST_CANCER_FEATURE_KEYS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


# ── Unified State Schema ──────────────────────────────────────────────────────

class MedAgentState(TypedDict):
    # ── User Inputs (detailed, for ML models) ──
    raw_user_query: str                                 # Original free-text query
    user_measurements: dict[str, int | float | None]    # Heart disease vitals with keys from USER_MEASUREMENT_KEYS
    breast_cancer_features: dict[str, float | None]     # 30 Wisconsin features (radius_mean, texture_mean, etc.)
    file_references: dict[str, str | None]              # {"breast_scan_path": ..., "skin_photo_path": ...}

    # ── Routing & Orchestration ──
    next_agent: str                                     # "heart_disease" | "breast_cancer" | "skin_disease" | "synthesis" | "__end__"
    messages: Annotated[list[BaseMessage], add_messages]  # LangGraph conversation history (append reducer)

    # ── Agent Outputs (ML model structured data) ──
    heart_disease_analysis: dict[str, Any]               # {diagnosis, confidence, details, features_used}
    breast_cancer_analysis: dict[str, Any]               # {diagnosis, confidence, details}
    skin_disease_analysis: dict[str, Any]                # {diagnosis, confidence, details}

    # ── User-Facing Aliases (clean API surface) ──
    user_query: str                                      # Alias for raw_user_query
    heart_rate: int | None                               # Convenience: extracted max_heart_rate
    breast_scan_path: str | None                         # Extracted from file_references
    skin_image_path: str | None                          # Extracted from file_references

    # ── LLM-Generated Explanations ──
    heart_diagnosis: str | None                          # Natural-language heart analysis
    cancer_diagnosis: str | None                         # Natural-language breast analysis
    skin_diagnosis: str | None                           # Natural-language skin analysis


# ── Structured Output Schemas ─────────────────────────────────────────────────

class RoutingDecision(BaseModel):
    next_agent: Literal[
        "heart_disease", "breast_cancer", "skin_disease", "synthesis", "__end__"
    ] = Field(
        description="The specialist to route to, 'synthesis' to compile all diagnoses, "
        "or '__end__' to finish without a medical response"
    )


# ── Input Extraction Schema ───────────────────────────────────────────────────

class ExtractedInputs(BaseModel):
    """Structured output from the LLM-based input extractor."""
    raw_user_query: str = Field(description="The original user query, cleaned up")
    user_measurements: dict[str, int | float | None] = Field(
        default_factory=dict,
        description="Heart disease vitals extracted from text. Keys: age, sex, chest_pain_type, "
        "resting_blood_pressure, cholesterol, fasting_blood_sugar, resting_ecg, "
        "max_heart_rate, exercise_induced_angina, oldpeak, slope, ca, thal. "
        "Null for any not mentioned.",
    )
    breast_cancer_features: dict[str, float | None] = Field(
        default_factory=dict,
        description="Breast cancer features if provided. Keys: radius_mean, texture_mean, etc.",
    )
    breast_scan_path: str | None = Field(
        default=None,
        description="Path or URI to a breast scan image if mentioned.",
    )
    skin_photo_path: str | None = Field(
        default=None,
        description="Path or URI to a skin lesion photo if mentioned.",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def fill_missing_measurements(
    measurements: dict[str, int | float | None],
) -> dict[str, int | float]:
    """Replace None values with sensible defaults so models always receive a full feature vector."""
    filled = {}
    for key in USER_MEASUREMENT_KEYS:
        val = measurements.get(key)
        filled[key] = val if val is not None else DEFAULT_MEASUREMENTS[key]
    return filled

def get_available_measurements(state: MedAgentState) -> str:
    """Build a human-readable summary of which measurements were provided vs missing."""
    provided = []
    missing = []
    for key in USER_MEASUREMENT_KEYS:
        val = state["user_measurements"].get(key)
        if val is not None:
            provided.append(f"{key}={val}")
        else:
            missing.append(key)
    msg = f"Provided: {', '.join(provided) or 'none'}"
    if missing:
        msg += f" | Missing (defaults used): {', '.join(missing)}"
    return msg


# ── Input Extraction Node (LLM-based) ────────────────────────────────────────

EXTRACTOR_SYSTEM_PROMPT = """You are a medical data extraction assistant. Given a patient's message, extract any structured medical data mentioned.

Extract the following when mentioned:
1. Heart disease measurements: age, sex (0=female,1=male), chest_pain_type (0-3), resting_blood_pressure, cholesterol, fasting_blood_sugar (0/1), resting_ecg (0-2), max_heart_rate, exercise_induced_angina (0/1), oldpeak (float), slope (0-2), ca (0-3), thal (0-3)
2. Breast cancer features: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, and their _se and _worst variants (30 total)
3. File paths: any mentioned image file paths for breast scans or skin photos

Rules:
- Return null for any field not explicitly mentioned by the user
- Convert text descriptions to numerical values (e.g., "male" -> sex=1, "female" -> sex=0)
- If the user says "chest pain type 2", set chest_pain_type=2
- If blood pressure is "140/90", use 140 for resting_blood_pressure
- Only extract what is explicitly stated; do not infer or assume"""

EXTRACTOR_USER_PROMPT = """Extract structured medical data from this patient message:

"{query}" """


def input_extraction_node(state: MedAgentState) -> dict[str, Any]:
    """Use LLM to parse natural language into structured medical inputs."""
    llm = ChatOllama(model="llama3.2", temperature=0)
    structured_llm = llm.with_structured_output(ExtractedInputs)

    query = state["raw_user_query"]

    response = structured_llm.invoke([
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=EXTRACTOR_USER_PROMPT.format(query=query)),
    ])

    # Merge extracted measurements into existing state (don't overwrite non-null with null)
    merged_measurements = dict(state.get("user_measurements", {}))
    for k, v in response.user_measurements.items():
        if v is not None:
            merged_measurements[k] = v

    merged_breast_features = dict(state.get("breast_cancer_features", {}))
    for k, v in response.breast_cancer_features.items():
        if v is not None:
            merged_breast_features[k] = v

    merged_files = dict(state.get("file_references", {}))
    if response.breast_scan_path:
        merged_files["breast_scan_path"] = response.breast_scan_path
    if response.skin_photo_path:
        merged_files["skin_photo_path"] = response.skin_photo_path

    return {
        "user_measurements": merged_measurements,
        "breast_cancer_features": merged_breast_features,
        "file_references": merged_files,
        # User-facing aliases
        "user_query": query,
        "heart_rate": merged_measurements.get("max_heart_rate"),
        "breast_scan_path": merged_files.get("breast_scan_path"),
        "skin_image_path": merged_files.get("skin_photo_path"),
    }


# ── Supervisor Node (LLM-based Router) ────────────────────────────────────────

TEXT_MODEL = "llama3.2"
VISION_MODEL = "llama3.2-vision"

ROUTER_SYSTEM_PROMPT = """You are a medical triage supervisor. You route patient queries to the appropriate specialist agent and decide when all analyses are complete.

You have access to the current diagnostic state. Use it to make routing decisions.

Respond with exactly one word from the list below:
- "heart_disease" — if the query involves chest pain, cardiac issues, blood pressure, heart rate, or cardiovascular symptoms AND this specialist has NOT yet been consulted
- "breast_cancer" — if the query involves breast lumps, mammograms, breast tissue concerns, or breast cancer screening AND this specialist has NOT yet been consulted
- "skin_disease" — if the query involves skin lesions, rashes, moles, dermatological issues, or skin photos AND this specialist has NOT yet been consulted
- "synthesis" — if at least one specialist has already provided a diagnosis AND there are no remaining unanalyzed inputs that need a specialist
- "__end__" — if the query is a greeting, pleasantry, thanks, farewell, or any other general non-medical conversation with no medical data

Routing priority:
1. Route to the specialist whose domain matches the query data (if not yet analyzed)
2. Route to "synthesis" when all relevant specialists have been consulted
3. Route to "__end__" for non-medical conversation

Examples:
- "Hello" -> __end__
- "Thanks" -> __end__
- "My chest hurts, heart rate is 120" -> heart_disease
- "I have a rash on my arm" -> skin_disease
- "I found a breast lump" -> breast_cancer
- (after heart_disease and skin_disease have run) -> synthesis

Only respond with the single word — no explanation."""

def supervisor_node(state: MedAgentState) -> dict[str, Any]:
    llm = ChatOllama(model=TEXT_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(RoutingDecision)

    # Build context about current diagnostic state
    heart_done = bool(state.get("heart_disease_analysis"))
    breast_done = bool(state.get("breast_cancer_analysis"))
    skin_done = bool(state.get("skin_disease_analysis"))
    has_heart_rate = state.get("heart_rate") is not None
    has_breast_scan = bool(state.get("breast_scan_path"))
    has_skin_image = bool(state.get("skin_image_path"))

    context_preamble = (
        f"[DIAGNOSTIC STATE] "
        f"heart_rate={'provided' if has_heart_rate else 'absent'}, "
        f"breast_scan={'provided' if has_breast_scan else 'absent'}, "
        f"skin_image={'provided' if has_skin_image else 'absent'} | "
        f"heart_done={heart_done}, breast_done={breast_done}, skin_done={skin_done}"
    )

    response = structured_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=context_preamble),
        *state["messages"],
    ])

    decision = response.next_agent

    return {
        "next_agent": decision,
        "messages": [HumanMessage(content=f"[Router: routed to {decision}]")],
    }


# ── Skin Disease Node ─────────────────────────────────────────────────────────

def skin_disease_node(state: MedAgentState) -> dict[str, Any]:
    image_path = state.get("skin_image_path") or state["file_references"].get("skin_photo_path")

    if not image_path:
        analysis = {
            "diagnosis": "no image provided",
            "confidence": 0.0,
            "details": "No skin photo was provided. Please upload a skin lesion image.",
        }
    else:
        try:
            analysis = predict_skin_disease(image_path)
        except FileNotFoundError as e:
            analysis = {
                "diagnosis": "model unavailable",
                "confidence": 0.0,
                "details": str(e),
            }

    # Multimodal vision LLM analysis
    diagnosis_text = "No skin image provided for analysis."
    if image_path:
        try:
            vision_llm = ChatOllama(model=VISION_MODEL, temperature=0)
            image_b64 = encode_image_to_base64(image_path)
            mime = get_image_mime_type(image_path)

            vision_response = vision_llm.invoke([
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": (
                            f"This is a skin lesion/dermatological image. "
                            f"The ML model classified it as: {analysis['diagnosis']} "
                            f"with {analysis['confidence']:.2f} confidence. "
                            f"Describe what you observe in this skin image, explain "
                            f"the classification, and provide patient-friendly guidance. "
                            f"Be concise (2-3 sentences)."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime};base64,{image_b64}",
                    },
                ])
            ])
            diagnosis_text = vision_response.content
        except Exception as e:
            diagnosis_text = (
                f"ML model result: {analysis['diagnosis']} "
                f"(confidence: {analysis['confidence']:.2f}). "
                f"Vision analysis unavailable: {e}"
            )

    return {
        "skin_disease_analysis": analysis,
        "skin_diagnosis": diagnosis_text,
        "messages": [
            AIMessage(
                content=f"**Skin Disease Analysis:**\n"
                f"- Diagnosis: {analysis['diagnosis']}\n"
                f"- Confidence: {analysis['confidence']:.2f}\n"
                f"- Details: {analysis['details']}\n\n"
                f"**Assessment:** {diagnosis_text}"
            )
        ],
    }


# ── Heart Disease Node ────────────────────────────────────────────────────────

def heart_disease_node(state: MedAgentState) -> dict[str, Any]:
    try:
        analysis = predict_heart_disease(state["user_measurements"])
    except FileNotFoundError as e:
        analysis = {
            "diagnosis": "model unavailable",
            "confidence": 0.0,
            "details": str(e),
            "features_used": [],
        }

    availability = get_available_measurements(state)

    # LLM natural-language explanation
    llm = ChatOllama(model=TEXT_MODEL, temperature=0)
    llm_response = llm.invoke([
        SystemMessage(
            content="You are a cardiology assistant. Explain this heart disease risk "
            "assessment in clear, patient-friendly language. Include the confidence "
            "level and what the measurements suggest. Be concise (2-3 sentences)."
        ),
        HumanMessage(
            content=f"Prediction: {analysis['diagnosis']}\n"
            f"Confidence: {analysis['confidence']:.2f}\n"
            f"Measurements used: {availability}"
        ),
    ])

    return {
        "heart_disease_analysis": analysis,
        "heart_diagnosis": llm_response.content,
        "messages": [
            AIMessage(
                content=f"**Heart Disease Analysis:**\n"
                f"- Diagnosis: {analysis['diagnosis']}\n"
                f"- Confidence: {analysis['confidence']:.2f}\n"
                f"- Details: {analysis['details']}\n"
                f"- {availability}\n\n"
                f"**Assessment:** {llm_response.content}"
            )
        ],
    }


# ── Breast Cancer Node ────────────────────────────────────────────────────────

def breast_cancer_node(state: MedAgentState) -> dict[str, Any]:
    image_path = state.get("breast_scan_path") or state["file_references"].get("breast_scan_path")
    features = state.get("breast_cancer_features", {})
    has_features = features and any(v is not None for v in features.values())

    if image_path:
        try:
            analysis = predict_breast_cancer(image_path=image_path)
        except FileNotFoundError as e:
            analysis = {
                "diagnosis": "model unavailable",
                "confidence": 0.0,
                "details": str(e),
            }
    elif has_features:
        try:
            analysis = predict_breast_cancer(features=features)
        except (FileNotFoundError, ValueError) as e:
            analysis = {
                "diagnosis": "model unavailable",
                "confidence": 0.0,
                "details": str(e),
            }
    else:
        analysis = {
            "diagnosis": "no input provided",
            "confidence": 0.0,
            "details": "No breast scan image or tabular features were provided.",
        }

    # LLM explanation — multimodal vision if image available, text-only otherwise
    if image_path:
        try:
            vision_llm = ChatOllama(model=VISION_MODEL, temperature=0)
            image_b64 = encode_image_to_base64(image_path)
            mime = get_image_mime_type(image_path)

            vision_response = vision_llm.invoke([
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": (
                            f"This is a breast scan/mammogram image. "
                            f"The ML model prediction is: {analysis['diagnosis']} "
                            f"with {analysis['confidence']:.2f} confidence. "
                            f"Describe what you observe in this medical image and "
                            f"explain the findings in patient-friendly language. "
                            f"Be concise (2-3 sentences)."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime};base64,{image_b64}",
                    },
                ])
            ])
            diagnosis_text = vision_response.content
        except Exception as e:
            # Fallback to text-only if vision model fails
            llm = ChatOllama(model=TEXT_MODEL, temperature=0)
            text_response = llm.invoke([
                SystemMessage(
                    content="You are a breast cancer screening assistant. "
                    "Explain this prediction in patient-friendly language. Be concise."
                ),
                HumanMessage(
                    content=f"Prediction: {analysis['diagnosis']}\n"
                    f"Confidence: {analysis['confidence']:.2f}\n"
                    f"Image was provided but vision analysis failed: {e}"
                ),
            ])
            diagnosis_text = text_response.content
    else:
        llm = ChatOllama(model=TEXT_MODEL, temperature=0)
        text_response = llm.invoke([
            SystemMessage(
                content="You are a breast cancer screening assistant. "
                "Explain this prediction in patient-friendly language. Be concise."
            ),
            HumanMessage(
                content=f"Prediction: {analysis['diagnosis']}\n"
                f"Confidence: {analysis['confidence']:.2f}"
            ),
        ])
        diagnosis_text = text_response.content

    return {
        "breast_cancer_analysis": analysis,
        "cancer_diagnosis": diagnosis_text,
        "messages": [
            AIMessage(
                content=f"**Breast Cancer Analysis:**\n"
                f"- Diagnosis: {analysis['diagnosis']}\n"
                f"- Confidence: {analysis['confidence']:.2f}\n"
                f"- Details: {analysis['details']}\n\n"
                f"**Assessment:** {diagnosis_text}"
            )
        ],
    }


# ── Synthesis Agent Node ──────────────────────────────────────────────────────

CLINICAL_DISCLAIMER = (
    "**⚕️ CLINICAL DISCLAIMER:** This is an automated preliminary assessment, "
    "not a formal diagnosis. The analysis was generated by AI models and should "
    "not be used as the basis for medical decisions. Please consult a qualified "
    "healthcare professional for definitive evaluation and treatment decisions."
)


def synthesis_agent_node(state: MedAgentState) -> dict[str, Any]:
    """Compile all specialist analyses into a unified patient report with cards."""
    cards = []

    if state.get("heart_disease_analysis"):
        a = state["heart_disease_analysis"]
        diag = state.get("heart_diagnosis", "No LLM assessment available.")
        cards.append(
            f"## ❤️ Heart Disease Assessment\n"
            f"**Result:** {a['diagnosis']} | **Confidence:** {a['confidence']:.1%}\n\n"
            f"{diag}"
        )

    if state.get("breast_cancer_analysis"):
        a = state["breast_cancer_analysis"]
        diag = state.get("cancer_diagnosis", "No LLM assessment available.")
        cards.append(
            f"## 🎗️ Breast Cancer Screening\n"
            f"**Result:** {a['diagnosis']} | **Confidence:** {a['confidence']:.1%}\n\n"
            f"{diag}"
        )

    if state.get("skin_disease_analysis"):
        a = state["skin_disease_analysis"]
        diag = state.get("skin_diagnosis", "No LLM assessment available.")
        cards.append(
            f"## 🩺 Skin Disease Analysis\n"
            f"**Result:** {a['diagnosis']} | **Confidence:** {a['confidence']:.1%}\n\n"
            f"{diag}"
        )

    report = (
        "\n\n---\n\n".join(cards)
        if cards
        else "No specialist analyses were performed."
    )
    report += f"\n\n---\n\n{CLINICAL_DISCLAIMER}"

    return {
        "messages": [AIMessage(content=report)],
    }


# ── Conditional Edge Router ───────────────────────────────────────────────────

def route_next(state: MedAgentState) -> str:
    return state["next_agent"]


# ── Graph Builder ─────────────────────────────────────────────────────────────

def compile_medagent_graph(
    checkpointer: MemorySaver | None = None,
) -> StateGraph:
    """
    Build and compile the MedAgent LangGraph.

    Flow:
        InputExtraction -> Supervisor -> [conditional] ->
            heart_disease  -> Supervisor (loop)
            breast_cancer  -> Supervisor (loop)
            skin_disease   -> Supervisor (loop)
            synthesis      -> SynthesisNode -> __end__
            __end__        -> __end__

    Parameters
    ----------
    checkpointer : MemorySaver, optional
        Persistence backend for multi-turn conversations.
    """
    graph = StateGraph(state_schema=MedAgentState)

    graph.add_node("InputExtractionNode", input_extraction_node)
    graph.add_node("SupervisorNode", supervisor_node)
    graph.add_node("SkinDiseaseNode", skin_disease_node)
    graph.add_node("HeartDiseaseNode", heart_disease_node)
    graph.add_node("BreastCancerNode", breast_cancer_node)
    graph.add_node("SynthesisNode", synthesis_agent_node)

    graph.set_entry_point("InputExtractionNode")
    graph.add_edge("InputExtractionNode", "SupervisorNode")

    graph.add_conditional_edges(
        "SupervisorNode",
        route_next,
        {
            "skin_disease": "SkinDiseaseNode",
            "heart_disease": "HeartDiseaseNode",
            "breast_cancer": "BreastCancerNode",
            "synthesis": "SynthesisNode",
            "__end__": "__end__",
        },
    )

    # Specialists loop back to Supervisor for re-routing
    graph.add_edge("SkinDiseaseNode", "SupervisorNode")
    graph.add_edge("HeartDiseaseNode", "SupervisorNode")
    graph.add_edge("BreastCancerNode", "SupervisorNode")

    # Synthesis terminates the graph
    graph.add_edge("SynthesisNode", "__end__")

    return graph.compile(checkpointer=checkpointer)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    checkpointer = MemorySaver()
    graph = compile_medagent_graph(checkpointer=checkpointer)

    input_state: MedAgentState = {
        "raw_user_query": "Analyze this image of a skin lesion.",
        "user_measurements": {key: None for key in USER_MEASUREMENT_KEYS},
        "breast_cancer_features": {key: None for key in BREAST_CANCER_FEATURE_KEYS},
        "file_references": {
            "breast_scan_path": None,
            "skin_photo_path": "data/sample_skin_lesion.jpg",
        },
        "next_agent": "",
        "messages": [HumanMessage(content="Analyze this image of a skin lesion.")],
        "heart_disease_analysis": {},
        "breast_cancer_analysis": {},
        "skin_disease_analysis": {},
        # User-facing aliases
        "user_query": "Analyze this image of a skin lesion.",
        "heart_rate": None,
        "breast_scan_path": None,
        "skin_image_path": "data/sample_skin_lesion.jpg",
        # LLM explanations
        "heart_diagnosis": None,
        "cancer_diagnosis": None,
        "skin_diagnosis": None,
    }

    config = {"configurable": {"thread_id": "test-session"}}
    result = graph.invoke(input_state, config)

    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    for k, v in result.items():
        if k == "messages":
            print(f"\n{k}:")
            for m in v:
                print(f"  [{m.type}] {m.content[:120]}{'...' if len(m.content) > 120 else ''}")
        else:
            print(f"{k}: {v}")
