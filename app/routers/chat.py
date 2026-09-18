from fastapi import APIRouter, Depends, HTTPException
from groq import Groq

from app.chat_extraction import build_clarifying_message, extract_variables, missing_fields
from app.chat_session import get_session
from app.composer import compose_recommendations_with_evidence
from app.dependencies import get_groq_client
from app.explain_builder import build_explain
from app.schemas.chat import ChatRequest, ChatResponse, StructuredChatRequest
from app.schemas.recommendations import RecommendationsResponse

router = APIRouter()


def _summarize(recommendations: RecommendationsResponse) -> str:
    if not recommendations.recommendations:
        return (
            "I couldn't find a recommendation grounded in the current knowledge "
            "base for this situation."
        )
    lines = ["Here's what I'd recommend:"]
    for rec in recommendations.recommendations:
        lines.append(
            f"- {rec.action} (impacts {rec.metric_impacted}, {rec.time_horizon}-term, "
            f"confidence: {rec.confidence})"
        )
    return "\n".join(lines)


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    groq_client: Groq = Depends(get_groq_client),
) -> ChatResponse:
    session = get_session(request.session_id)
    session.history.append({"role": "user", "content": request.message})

    if request.structured_input:
        session.variables.update(request.structured_input)
    session.variables.update(extract_variables(request.message))

    missing = missing_fields(session.variables)
    if missing:
        reply = build_clarifying_message(missing)
        session.history.append({"role": "assistant", "content": reply})
        return ChatResponse(
            session_id=request.session_id,
            status="needs_clarification",
            reply=reply,
            missing_fields=missing,
            recommendations=None,
        )

    recommendations, enriched_chains = compose_recommendations_with_evidence(
        session.variables, client=groq_client
    )
    reply = _summarize(recommendations)
    session.history.append({"role": "assistant", "content": reply})

    return ChatResponse(
        session_id=request.session_id,
        status="complete",
        reply=reply,
        missing_fields=[],
        recommendations=recommendations,
        explain=build_explain(session.variables, enriched_chains, recommendations),
    )


@router.post("/chat/structured", response_model=ChatResponse)
def chat_structured(
    request: StructuredChatRequest,
    groq_client: Groq = Depends(get_groq_client),
) -> ChatResponse:
    session = get_session(request.session_id)
    session.variables.update(request.inputs)
    session.history.append({"role": "user", "content": f"[structured input] {request.inputs}"})

    missing = missing_fields(session.variables)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields for structured input: {missing}",
        )

    recommendations, enriched_chains = compose_recommendations_with_evidence(
        session.variables, client=groq_client
    )
    reply = _summarize(recommendations)
    session.history.append({"role": "assistant", "content": reply})

    return ChatResponse(
        session_id=request.session_id,
        status="complete",
        reply=reply,
        missing_fields=[],
        recommendations=recommendations,
        explain=build_explain(session.variables, enriched_chains, recommendations),
    )
