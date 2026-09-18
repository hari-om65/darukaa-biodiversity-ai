from typing import Literal

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    action: str
    mechanism: str
    metric_impacted: str
    time_horizon: Literal["short", "medium", "long"]
    confidence: Literal["low", "medium", "high"]
    sources: list[str] = Field(min_length=1)


class RecommendationsResponse(BaseModel):
    recommendations: list[Recommendation]
