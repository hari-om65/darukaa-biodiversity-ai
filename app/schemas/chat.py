from typing import Any, Literal

from pydantic import BaseModel

from app.schemas.recommendations import RecommendationsResponse


class ChatRequest(BaseModel):
    session_id: str
    message: str
    structured_input: dict[str, Any] | None = None


class StructuredChatRequest(BaseModel):
    session_id: str
    inputs: dict[str, Any]


class ChatResponse(BaseModel):
    session_id: str
    status: Literal["needs_clarification", "complete"]
    reply: str
    missing_fields: list[str] = []
    recommendations: RecommendationsResponse | None = None
