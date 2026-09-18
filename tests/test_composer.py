import os

import pytest

from app.composer import _collect_valid_sources, compose_recommendations, gather_chain_evidence
from app.schemas.recommendations import RecommendationsResponse
from knowledge.ingestion.ingest import ingest_all

# A wheat monoculture in a semi-arid region: degraded soil, low rainfall.
WHEAT_INPUTS = {
    "soil_organic_carbon": 0.3,
    "rainfall": "low",
    "crop": "monoculture",
    "crop_type": "wheat",
    "region": "semi-arid",
}


@pytest.fixture(scope="module", autouse=True)
def _ensure_knowledge_base_ingested():
    # gather_chain_evidence() calls retrieve() against the persistent
    # knowledge/chroma_db store (no client override), so it must actually be
    # populated. upsert() is idempotent, so re-running this is safe.
    ingest_all()


class _FakeParsedResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _FakeMessages:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.call_count = 0

    def parse(self, **kwargs):
        outcome = self._outcomes[self.call_count]
        self.call_count += 1
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeParsedResponse(outcome)


class _FakeClient:
    def __init__(self, outcomes):
        self.messages = _FakeMessages(outcomes)


def _recommendation(sources: list[str]) -> RecommendationsResponse:
    return RecommendationsResponse(
        recommendations=[
            {
                "action": "Adopt reduced tillage and cover cropping on the wheat plots",
                "mechanism": (
                    "Raises soil organic carbon, which the retrieved evidence links to "
                    "higher microbial diversity and better water retention."
                ),
                "metric_impacted": "microbial_diversity",
                "time_horizon": "medium",
                "confidence": "medium",
                "sources": sources,
            }
        ]
    )


def test_gather_chain_evidence_returns_evidence_for_wheat_scenario():
    enriched = gather_chain_evidence(WHEAT_INPUTS)

    assert len(enriched) >= 3
    assert all("evidence" in chain for chain in enriched)
    assert any(chain["evidence"] for chain in enriched)


def test_compose_recommendations_retries_once_on_ungrounded_sources():
    valid_sources = _collect_valid_sources(gather_chain_evidence(WHEAT_INPUTS))
    real_source = next(iter(valid_sources))

    fake_client = _FakeClient(
        outcomes=[
            _recommendation(["nonexistent_source.md"]),  # invented citation -> rejected
            _recommendation([real_source]),  # grounded in real retrieved evidence
        ]
    )

    result = compose_recommendations(WHEAT_INPUTS, client=fake_client)

    assert isinstance(result, RecommendationsResponse)
    assert fake_client.messages.call_count == 2
    assert result.recommendations[0].sources == [real_source]


def test_compose_recommendations_raises_after_second_failure():
    fake_client = _FakeClient(
        outcomes=[
            _recommendation(["nonexistent_source.md"]),
            _recommendation(["still_nonexistent.md"]),
        ]
    )

    with pytest.raises(RuntimeError):
        compose_recommendations(WHEAT_INPUTS, client=fake_client)

    assert fake_client.messages.call_count == 2


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY to call the Anthropic API",
)
def test_compose_recommendations_live_semi_arid_wheat():
    result = compose_recommendations(WHEAT_INPUTS)
    valid_sources = _collect_valid_sources(gather_chain_evidence(WHEAT_INPUTS))

    assert isinstance(result, RecommendationsResponse)
    assert len(result.recommendations) >= 1

    for rec in result.recommendations:
        assert rec.time_horizon in {"short", "medium", "long"}
        assert rec.confidence in {"low", "medium", "high"}
        assert rec.sources
        assert set(rec.sources) <= valid_sources
