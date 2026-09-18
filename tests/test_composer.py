import os

import pytest

from app.composer import (
    _collect_valid_sources,
    _validate_sources_are_grounded,
    build_user_prompt,
    compose_recommendations,
    compose_recommendations_with_evidence,
    gather_chain_evidence,
)
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


def test_gather_chain_evidence_returns_empty_when_nothing_triggers():
    assert gather_chain_evidence({}) == []


def test_gather_chain_evidence_falls_back_to_unscoped_search_when_tag_scoped_is_empty(
    monkeypatch,
):
    calls = []

    def fake_retrieve(query, tags=None, k=4, client=None):
        calls.append(tags)
        if tags:
            return []  # simulate an empty tag-scoped result
        return [
            {
                "id": "fallback-0",
                "text": "fallback chunk",
                "metadata": {"source": "fallback.md", "year": 2020, "topic_tags": "soil"},
                "distance": 0.1,
            }
        ]

    monkeypatch.setattr("app.composer.retrieve", fake_retrieve)

    enriched = gather_chain_evidence({"soil_organic_carbon": 0.3})

    assert enriched
    assert all(chain["evidence"] for chain in enriched)
    assert any(tags for tags in calls)  # a tag-scoped call happened first
    assert any(tags is None for tags in calls)  # and the unscoped fallback ran


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


class _ExplodingMessages:
    def parse(self, **kwargs):
        raise AssertionError("the API should not be called when no chains are triggered")


class _ExplodingClient:
    def __init__(self):
        self.messages = _ExplodingMessages()


def test_compose_recommendations_skips_api_call_when_no_chains_triggered():
    result = compose_recommendations({}, client=_ExplodingClient())

    assert result == RecommendationsResponse(recommendations=[])


def test_compose_recommendations_with_evidence_returns_chains_used_for_prompt():
    valid_sources = _collect_valid_sources(gather_chain_evidence(WHEAT_INPUTS))
    real_source = next(iter(valid_sources))
    fake_client = _FakeClient(outcomes=[_recommendation([real_source])])

    result, enriched_chains = compose_recommendations_with_evidence(WHEAT_INPUTS, client=fake_client)

    assert isinstance(result, RecommendationsResponse)
    assert enriched_chains
    assert all("evidence" in chain for chain in enriched_chains)
    assert {chain["path"][0] for chain in enriched_chains} == {
        "soil_organic_carbon",
        "rainfall",
        "monoculture",
    }


def test_validate_sources_are_grounded_raises_on_unknown_source():
    result = _recommendation(["unknown.md"])

    with pytest.raises(ValueError, match="unknown.md"):
        _validate_sources_are_grounded(result, valid_sources={"known.md"})


def test_validate_sources_are_grounded_passes_when_sources_are_known():
    result = _recommendation(["known.md"])

    _validate_sources_are_grounded(result, valid_sources={"known.md"})  # must not raise


def test_build_user_prompt_includes_inputs_chains_and_evidence():
    chains = [
        {
            "path": ["soil_organic_carbon", "microbial_diversity"],
            "affected_metric": "microbial_diversity",
            "relation": "positive",
            "strength": "high",
            "evidence": [
                {
                    "text": "Soil organic carbon supports microbial diversity.",
                    "metadata": {
                        "source": "soil_organic_carbon.md",
                        "year": 2020,
                        "topic_tags": "soil,biodiversity",
                    },
                }
            ],
        },
        {
            "path": ["rainfall", "species_survival"],
            "affected_metric": "species_survival",
            "relation": "positive",
            "strength": "high",
            "evidence": [],
        },
    ]

    prompt = build_user_prompt({"soil_organic_carbon": 0.3}, chains)

    assert "## Site inputs" in prompt
    assert '"soil_organic_carbon": 0.3' in prompt
    assert "soil_organic_carbon -> microbial_diversity" in prompt
    assert "source: soil_organic_carbon.md | year: 2020 | tags: soil,biodiversity" in prompt
    assert "Soil organic carbon supports microbial diversity." in prompt
    assert "rainfall -> species_survival" in prompt
    assert "evidence: none retrieved for this chain" in prompt


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
