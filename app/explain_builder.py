"""Builds the /chat and /whatif "explain" payload: which thresholds fired,
which graph paths were walked, which evidence chunks were retrieved per
chain, and how those chunks map to each returned recommendation.
"""

from typing import Any

from app.schemas.explain import ChainEvidence, Explain, RecommendationMapping, RetrievedChunk, ThresholdFired
from app.schemas.recommendations import RecommendationsResponse
from reasoning.engine import evaluate_thresholds


def build_explain(
    inputs: dict[str, Any],
    chains: list[dict[str, Any]],
    recommendations: RecommendationsResponse | None = None,
) -> Explain:
    """`chains` may be plain reasoning chains (from analyze(), as in /whatif)
    or evidence-enriched chains (from composer.gather_chain_evidence(), as in
    /chat) - chains without an "evidence" key simply produce no evidence/
    recommendation_mapping entries. `recommendations` is omitted for /whatif,
    which doesn't call the composer.
    """
    thresholds_fired = [ThresholdFired(**t) for t in evaluate_thresholds(inputs)]
    graph_paths = [chain["path"] for chain in chains]

    evidence = [
        ChainEvidence(
            path=chain["path"],
            chunks=[
                RetrievedChunk(
                    source=ev["metadata"]["source"],
                    year=ev["metadata"]["year"],
                    topic_tags=ev["metadata"]["topic_tags"],
                    # Chroma returns a distance (lower = more similar); this
                    # is a monotonic 1/(1+distance) transform into (0, 1] so
                    # "higher = more relevant" without assuming a particular
                    # distance metric's bounds.
                    score=1 / (1 + ev["distance"]),
                    distance=ev["distance"],
                    text=ev["text"],
                )
                for ev in chain.get("evidence", [])
            ],
        )
        for chain in chains
        if chain.get("evidence")
    ]

    recommendation_mapping: list[RecommendationMapping] = []
    if recommendations is not None:
        source_to_chains: dict[str, list[list[str]]] = {}
        for chain in chains:
            for ev in chain.get("evidence", []):
                source_to_chains.setdefault(ev["metadata"]["source"], []).append(chain["path"])

        for rec in recommendations.recommendations:
            supporting_chains: list[list[str]] = []
            seen: set[tuple[str, ...]] = set()
            for source in rec.sources:
                for path in source_to_chains.get(source, []):
                    key = tuple(path)
                    if key not in seen:
                        seen.add(key)
                        supporting_chains.append(path)
            recommendation_mapping.append(
                RecommendationMapping(
                    action=rec.action,
                    supporting_chains=supporting_chains,
                    sources=rec.sources,
                )
            )

    return Explain(
        thresholds_fired=thresholds_fired,
        graph_paths=graph_paths,
        evidence=evidence,
        recommendation_mapping=recommendation_mapping,
    )
