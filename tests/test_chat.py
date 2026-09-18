import pytest
from fastapi.testclient import TestClient

from app.chat_session import reset_sessions
from app.dependencies import get_anthropic_client
from app.main import app
from app.schemas.recommendations import RecommendationsResponse
from knowledge.ingestion.ingest import ingest_all, retrieve


class _FakeParsedResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _FakeMessages:
    def __init__(self, output):
        self._output = output
        self.call_count = 0

    def parse(self, **kwargs):
        self.call_count += 1
        return _FakeParsedResponse(self._output)


class _FakeClient:
    def __init__(self, output):
        self.messages = _FakeMessages(output)


def _fake_recommendations(sources: list[str]) -> RecommendationsResponse:
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


@pytest.fixture(scope="module", autouse=True)
def _ensure_knowledge_base_ingested():
    ingest_all()


@pytest.fixture(autouse=True)
def _clean_sessions():
    reset_sessions()
    yield
    reset_sessions()


def _override_with_fake_client(sources: list[str]) -> None:
    app.dependency_overrides[get_anthropic_client] = lambda: _FakeClient(
        _fake_recommendations(sources)
    )


def test_three_turn_conversation_reaches_recommendation():
    evidence = retrieve("soil organic carbon", tags=["soil"], k=1)
    real_source = evidence[0]["metadata"]["source"] if evidence else "soil_organic_carbon.md"
    _override_with_fake_client([real_source])
    client = TestClient(app)

    try:
        session_id = "farmer-session-1"

        # Turn 1: vague complaint - nothing extractable, all fields missing.
        r1 = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "My wheat field doesn't seem healthy this year, yields are down.",
            },
        )
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["status"] == "needs_clarification"
        assert set(body1["missing_fields"]) == {
            "soil_organic_carbon",
            "rainfall",
            "crop",
            "region",
        }
        assert body1["recommendations"] is None
        assert body1["reply"]

        # Turn 2: partial data - clarifying question should only cover what's
        # still missing, not re-ask about soil carbon or rainfall.
        r2 = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "Soil organic carbon is 0.3 and rainfall has been low.",
            },
        )
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["status"] == "needs_clarification"
        assert set(body2["missing_fields"]) == {"crop", "region"}

        # Turn 3: remaining data supplied - pipeline runs end to end.
        r3 = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "It's a monoculture wheat crop in a semi-arid region.",
            },
        )
        assert r3.status_code == 200
        body3 = r3.json()
        assert body3["status"] == "complete"
        assert body3["missing_fields"] == []
        assert body3["recommendations"] is not None
        assert len(body3["recommendations"]["recommendations"]) == 1
        assert body3["recommendations"]["recommendations"][0]["sources"] == [real_source]
        assert "recommend" in body3["reply"].lower()
    finally:
        app.dependency_overrides.pop(get_anthropic_client, None)


def test_chat_structured_skips_clarification():
    _override_with_fake_client(["soil_organic_carbon.md"])
    client = TestClient(app)

    try:
        response = client.post(
            "/chat/structured",
            json={
                "session_id": "farmer-session-2",
                "inputs": {
                    "soil_organic_carbon": 0.3,
                    "rainfall": "low",
                    "crop": "monoculture",
                    "region": "semi-arid",
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "complete"
        assert body["missing_fields"] == []
        assert body["recommendations"] is not None
    finally:
        app.dependency_overrides.pop(get_anthropic_client, None)


def test_chat_structured_rejects_incomplete_input():
    client = TestClient(app)
    response = client.post(
        "/chat/structured",
        json={"session_id": "farmer-session-3", "inputs": {"soil_organic_carbon": 0.3}},
    )
    assert response.status_code == 422
