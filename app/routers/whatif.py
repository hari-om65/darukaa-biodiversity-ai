"""/whatif: simulate a proposed variable change against a session's known
variables and show how the reasoning chains shift, without committing the
change to the session.
"""

from typing import Any

from fastapi import APIRouter

from app.chat_session import get_session
from app.schemas.whatif import ChainResult, WhatIfRequest, WhatIfResponse
from reasoning.engine import analyze, project_time_horizon

router = APIRouter()


def _chain_key(chain: dict[str, Any]) -> tuple[str, ...]:
    return tuple(chain["path"])


def _to_chain_result(chain: dict[str, Any]) -> ChainResult:
    return ChainResult(
        path=chain["path"],
        affected_metric=chain["affected_metric"],
        relation=chain["relation"],
        strength=chain["strength"],
        projected_horizon=project_time_horizon(chain),
    )


@router.post("/whatif", response_model=WhatIfResponse)
def whatif(request: WhatIfRequest) -> WhatIfResponse:
    session = get_session(request.session_id)

    before_inputs = dict(session.variables)
    old_value = before_inputs.get(request.variable)

    after_inputs = dict(before_inputs)
    after_inputs[request.variable] = request.value

    before_chains = analyze(before_inputs)
    after_chains = analyze(after_inputs)

    before_by_key = {_chain_key(c): c for c in before_chains}
    after_by_key = {_chain_key(c): c for c in after_chains}

    removed = [c for key, c in before_by_key.items() if key not in after_by_key]
    added = [c for key, c in after_by_key.items() if key not in before_by_key]

    return WhatIfResponse(
        session_id=request.session_id,
        variable=request.variable,
        old_value=old_value,
        new_value=request.value,
        before_chains=[_to_chain_result(c) for c in before_chains],
        after_chains=[_to_chain_result(c) for c in after_chains],
        removed_chains=[_to_chain_result(c) for c in removed],
        added_chains=[_to_chain_result(c) for c in added],
    )
