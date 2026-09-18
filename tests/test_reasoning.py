import pytest

from reasoning.engine import (
    GRAPH,
    _compose_relation,
    _compose_strength,
    _walk_downstream,
    analyze,
    evaluate_thresholds,
    load_graph,
    load_thresholds,
    project_time_horizon,
)


def test_load_graph_has_expected_edges():
    graph = load_graph()

    assert graph.number_of_edges() == 7
    assert graph["soil_organic_carbon"]["microbial_diversity"] == {
        "relation": "positive",
        "strength": "high",
    }
    assert graph["habitat_fragmentation"]["species_richness"] == {
        "relation": "negative",
        "strength": "high",
    }
    assert graph is not GRAPH  # a fresh load, not the cached module-level graph


def test_load_thresholds_has_expected_entries():
    thresholds = load_thresholds()

    assert len(thresholds) == 3
    by_flag = {t["flag"]: t for t in thresholds}
    assert by_flag["critical_low"] == {
        "variable": "soil_organic_carbon",
        "operator": "<",
        "value": 0.5,
        "flag": "critical_low",
        "node": "soil_organic_carbon",
    }
    # crop's threshold node deliberately differs from its variable name -
    # the triggering value ("monoculture") is the graph node, not "crop".
    assert by_flag["flagged"]["variable"] == "crop"
    assert by_flag["flagged"]["node"] == "monoculture"


def test_evaluate_thresholds_all_operators():
    thresholds = [
        {"variable": "a", "operator": "<", "value": 5, "flag": "lt", "node": "a"},
        {"variable": "b", "operator": "<=", "value": 5, "flag": "lte", "node": "b"},
        {"variable": "c", "operator": ">", "value": 5, "flag": "gt", "node": "c"},
        {"variable": "d", "operator": ">=", "value": 5, "flag": "gte", "node": "d"},
        {"variable": "e", "operator": "==", "value": 5, "flag": "eq", "node": "e"},
        {"variable": "f", "operator": "!=", "value": 5, "flag": "ne", "node": "f"},
    ]
    inputs = {"a": 4, "b": 5, "c": 6, "d": 5, "e": 5, "f": 6}

    triggered_flags = {t["flag"] for t in evaluate_thresholds(inputs, thresholds)}

    assert triggered_flags == {"lt", "lte", "gt", "gte", "eq", "ne"}


def test_evaluate_thresholds_skips_variables_not_present_in_inputs():
    thresholds = [{"variable": "missing", "operator": "<", "value": 5, "flag": "x", "node": "missing"}]

    assert evaluate_thresholds({}, thresholds) == []
    assert evaluate_thresholds({"other": 1}, thresholds) == []


def test_evaluate_thresholds_condition_not_met_does_not_trigger():
    thresholds = [{"variable": "a", "operator": "<", "value": 5, "flag": "x", "node": "a"}]

    assert evaluate_thresholds({"a": 10}, thresholds) == []


def test_compose_relation_double_negative_is_positive():
    edges = [{"relation": "negative"}, {"relation": "negative"}]
    assert _compose_relation(edges) == "positive"


def test_compose_relation_mixed_signs_is_negative():
    edges = [{"relation": "positive"}, {"relation": "negative"}]
    assert _compose_relation(edges) == "negative"


def test_compose_relation_single_edge():
    assert _compose_relation([{"relation": "positive"}]) == "positive"
    assert _compose_relation([{"relation": "negative"}]) == "negative"


def test_compose_strength_picks_weakest_link_regardless_of_order():
    assert _compose_strength([{"strength": "high"}, {"strength": "low"}]) == "low"
    assert _compose_strength([{"strength": "low"}, {"strength": "high"}]) == "low"
    assert _compose_strength([{"strength": "high"}, {"strength": "medium"}]) == "medium"
    assert _compose_strength([{"strength": "high"}]) == "high"


def test_walk_downstream_returns_empty_for_unknown_start():
    assert _walk_downstream(GRAPH, "not_a_real_node") == []


def test_analyze_returns_empty_when_nothing_triggers():
    inputs = {"soil_organic_carbon": 0.9, "rainfall": "high", "crop": "polyculture"}
    assert analyze(inputs) == []


def test_analyze_empty_inputs_triggers_nothing():
    assert analyze({}) == []


def test_analyze_crop_threshold_walks_from_monoculture_node_not_crop():
    # In isolation, only the crop threshold fires; every resulting chain must
    # start at the graph node "monoculture" (the threshold's node override),
    # never at "crop" (which isn't a node in the graph at all).
    results = analyze({"crop": "monoculture"})

    assert results
    assert {r["path"][0] for r in results} == {"monoculture"}


@pytest.mark.parametrize(
    ("path_len", "strength", "expected"),
    [
        (2, "high", "short"),
        (2, "medium", "medium"),
        (2, "low", "long"),
        (3, "high", "medium"),
        (3, "medium", "long"),
        (3, "low", "long"),
    ],
)
def test_project_time_horizon_full_matrix(path_len, strength, expected):
    chain = {"path": list(range(path_len)), "strength": strength}
    assert project_time_horizon(chain) == expected


def test_analyze_finds_chains_across_soil_water_and_land_use():
    inputs = {
        "soil_organic_carbon": 0.3,
        "rainfall": "low",
        "crop": "monoculture",
        "region": "semi-arid",
    }

    results = analyze(inputs)

    assert len(results) >= 3

    for result in results:
        assert set(result.keys()) == {"path", "affected_metric", "relation", "strength"}
        assert result["relation"] in {"positive", "negative"}
        assert result["strength"] in {"low", "medium", "high"}

    chain_starts = {result["path"][0] for result in results}
    assert {"soil_organic_carbon", "rainfall", "monoculture"} <= chain_starts

    affected_metrics = {result["affected_metric"] for result in results}
    assert "species_richness" in affected_metrics  # land-use chain (monoculture -> ... )
    assert "species_survival" in affected_metrics  # water chain (rainfall -> ...)
    assert "microbial_diversity" in affected_metrics  # soil chain (soil_organic_carbon -> ...)
