from typing import Any

from pydantic import BaseModel


class ThresholdFired(BaseModel):
    variable: str
    operator: str
    value: Any
    flag: str
    node: str


class RetrievedChunk(BaseModel):
    source: str
    year: int
    topic_tags: str
    score: float
    distance: float
    text: str


class ChainEvidence(BaseModel):
    path: list[str]
    chunks: list[RetrievedChunk]


class RecommendationMapping(BaseModel):
    action: str
    supporting_chains: list[list[str]]
    sources: list[str]


class Explain(BaseModel):
    thresholds_fired: list[ThresholdFired] = []
    graph_paths: list[list[str]] = []
    evidence: list[ChainEvidence] = []
    recommendation_mapping: list[RecommendationMapping] = []
