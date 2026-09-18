from typing import Any, Literal

from pydantic import BaseModel

from app.schemas.explain import Explain


class WhatIfRequest(BaseModel):
    session_id: str
    variable: str
    value: Any


class ChainResult(BaseModel):
    path: list[str]
    affected_metric: str
    relation: Literal["positive", "negative"]
    strength: Literal["low", "medium", "high"]
    projected_horizon: Literal["short", "medium", "long"]


class WhatIfResponse(BaseModel):
    session_id: str
    variable: str
    old_value: Any | None
    new_value: Any
    before_chains: list[ChainResult]
    after_chains: list[ChainResult]
    removed_chains: list[ChainResult]
    added_chains: list[ChainResult]
    explain: Explain = Explain()
