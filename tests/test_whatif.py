import pytest
from fastapi.testclient import TestClient

from app.chat_session import get_session, reset_sessions
from app.main import app
from reasoning.engine import project_time_horizon


@pytest.fixture(autouse=True)
def _clean_sessions():
    reset_sessions()
    yield
    reset_sessions()


def test_whatif_raising_soil_organic_carbon_removes_critical_low_chains():
    session_id = "whatif-session-1"
    session = get_session(session_id)
    session.variables.update(
        {
            "soil_organic_carbon": 0.3,
            "rainfall": "low",
            "crop": "monoculture",
            "region": "semi-arid",
        }
    )

    client = TestClient(app)
    response = client.post(
        "/whatif",
        json={"session_id": session_id, "variable": "soil_organic_carbon", "value": 1.5},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["old_value"] == 0.3
    assert body["new_value"] == 1.5

    before_starts = {c["path"][0] for c in body["before_chains"]}
    after_starts = {c["path"][0] for c in body["after_chains"]}
    assert "soil_organic_carbon" in before_starts
    assert "soil_organic_carbon" not in after_starts

    # Every soil-organic-carbon chain present before disappears after raising it;
    # nothing new is triggered by the raise itself.
    assert len(body["removed_chains"]) > 0
    assert all(c["path"][0] == "soil_organic_carbon" for c in body["removed_chains"])
    assert body["added_chains"] == []

    # The rainfall- and monoculture-triggered chains are untouched.
    remaining_starts = {c["path"][0] for c in body["after_chains"]}
    assert remaining_starts == {"rainfall", "monoculture"}

    # /whatif simulates - it must not mutate the session's stored variables.
    assert session.variables["soil_organic_carbon"] == 0.3


def test_whatif_chains_include_projected_horizon():
    session_id = "whatif-session-2"
    session = get_session(session_id)
    session.variables.update({"crop": "monoculture"})

    client = TestClient(app)
    response = client.post(
        "/whatif",
        json={"session_id": session_id, "variable": "crop", "value": "monoculture"},
    )

    body = response.json()
    assert body["after_chains"]
    for chain in body["after_chains"]:
        assert chain["projected_horizon"] in {"short", "medium", "long"}


def test_project_time_horizon_heuristic():
    # 1 hop, high strength -> short
    assert (
        project_time_horizon({"path": ["a", "b"], "strength": "high"}) == "short"
    )
    # 1 hop, low strength -> long
    assert project_time_horizon({"path": ["a", "b"], "strength": "low"}) == "long"
    # 2 hops, high strength -> medium
    assert (
        project_time_horizon({"path": ["a", "b", "c"], "strength": "high"}) == "medium"
    )
    # 2 hops, low strength -> long
    assert (
        project_time_horizon({"path": ["a", "b", "c"], "strength": "low"}) == "long"
    )
