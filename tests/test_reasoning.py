from reasoning.engine import analyze


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
