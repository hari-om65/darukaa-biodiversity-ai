"""Composes evidence-grounded recommendations.

For each causal reasoning chain produced by reasoning.engine.analyze(), retrieves
scoped evidence from the knowledge base via knowledge.ingestion.ingest.retrieve(),
then makes one call to the Anthropic API asking for structured, JSON-only
recommendations that must be grounded in that retrieved evidence. The response is
validated with Pydantic and, on parse/validation failure, retried once.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from pydantic import ValidationError

from app.schemas.recommendations import RecommendationsResponse
from knowledge.ingestion.ingest import retrieve
from reasoning.engine import analyze

# Override with ANTHROPIC_MODEL to pin a different model without a code change.
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 4096
EVIDENCE_K = 3

# Maps causal-graph node names (reasoning/graph.yaml) to the knowledge-base
# topic tags relevant to retrieving evidence about that node.
NODE_TOPIC_TAGS: dict[str, list[str]] = {
    "soil_organic_carbon": ["soil"],
    "microbial_diversity": ["soil", "biodiversity"],
    "water_retention": ["water", "soil"],
    "rainfall": ["water", "climate"],
    "species_survival": ["biodiversity", "water"],
    "monoculture": ["land_use", "human_impact"],
    "habitat_fragmentation": ["land_use", "biodiversity"],
    "species_richness": ["biodiversity", "land_use"],
    "pollinator_support": ["biodiversity"],
}

SYSTEM_PROMPT = """You are a biodiversity and land-management advisor for Darukaa.

You are given: the observed input values for a site, a set of causal reasoning \
chains derived from a domain causal graph, and evidence excerpts retrieved from \
a curated knowledge base for each chain.

Rules:
- Output ONLY the requested JSON. No prose, no markdown fences, no commentary.
- Every recommendation's "mechanism" must be grounded in the retrieved evidence \
provided to you. Do not invent statistics, percentages, or findings that are not \
present in the evidence excerpts.
- Do not give generic advice ("improve soil health", "plant more trees") that is \
not tied to a specific reasoning chain and its evidence.
- Every recommendation's "sources" list must name only source filenames that \
appear in the evidence provided to you for that chain.
- If the evidence for a chain is too thin to support a specific, non-generic \
recommendation, omit that chain rather than fabricating one.
"""


def gather_chain_evidence(inputs: dict[str, Any], k: int = EVIDENCE_K) -> list[dict[str, Any]]:
    """Run the causal reasoning engine and attach retrieved evidence to each chain."""
    chains = analyze(inputs)
    enriched = []
    for chain in chains:
        tags: list[str] = []
        for node in chain["path"]:
            for tag in NODE_TOPIC_TAGS.get(node, []):
                if tag not in tags:
                    tags.append(tag)

        query = " -> ".join(chain["path"])
        evidence = retrieve(query, tags=tags or None, k=k)
        if not evidence and tags:
            # Tag-scoped search came up empty; fall back to an unscoped search
            # rather than silently sending the model zero evidence for a real chain.
            evidence = retrieve(query, k=k)

        enriched.append({**chain, "evidence": evidence})
    return enriched


def build_user_prompt(inputs: dict[str, Any], enriched_chains: list[dict[str, Any]]) -> str:
    lines = [
        "## Site inputs",
        json.dumps(inputs, indent=2),
        "",
        "## Reasoning chains and evidence",
    ]
    for i, chain in enumerate(enriched_chains, start=1):
        lines.append(f"\n### Chain {i}")
        lines.append(f"path: {' -> '.join(chain['path'])}")
        lines.append(f"affected_metric: {chain['affected_metric']}")
        lines.append(f"relation: {chain['relation']}")
        lines.append(f"strength: {chain['strength']}")
        if chain["evidence"]:
            lines.append("evidence:")
            for ev in chain["evidence"]:
                meta = ev["metadata"]
                lines.append(
                    f"- [source: {meta['source']} | year: {meta['year']} | "
                    f"tags: {meta['topic_tags']}] {ev['text']}"
                )
        else:
            lines.append("evidence: none retrieved for this chain")

    lines.append(
        "\nUsing only the chains and evidence above, produce recommendations as instructed."
    )
    return "\n".join(lines)


def _collect_valid_sources(enriched_chains: list[dict[str, Any]]) -> set[str]:
    sources: set[str] = set()
    for chain in enriched_chains:
        for ev in chain["evidence"]:
            sources.add(ev["metadata"]["source"])
    return sources


def _validate_sources_are_grounded(result: RecommendationsResponse, valid_sources: set[str]) -> None:
    """Raise ValueError if any recommendation cites a source that was not
    actually retrieved as evidence for this request (an invented citation)."""
    for rec in result.recommendations:
        unknown = [s for s in rec.sources if s not in valid_sources]
        if unknown:
            raise ValueError(f"Recommendation '{rec.action}' cites unknown sources: {unknown}")


def compose_recommendations_with_evidence(
    inputs: dict[str, Any],
    client: anthropic.Anthropic | None = None,
) -> tuple[RecommendationsResponse, list[dict[str, Any]]]:
    """Build one evidence-grounded prompt from `inputs` and return validated
    recommendations, along with the enriched chains (each reasoning chain plus
    the evidence retrieved for it) used to build that prompt.

    Callers that need to explain *how* a recommendation was reached (which
    thresholds fired, which chunks were retrieved, ...) should use this
    instead of compose_recommendations() to avoid re-running retrieval.

    Retries once (a single additional API call, with the failure reason
    appended to the prompt) if the response fails JSON parsing, Pydantic
    validation, or evidence-grounding validation.
    """
    client = client or anthropic.Anthropic()

    enriched_chains = gather_chain_evidence(inputs)
    if not enriched_chains:
        return RecommendationsResponse(recommendations=[]), enriched_chains

    valid_sources = _collect_valid_sources(enriched_chains)
    base_prompt = build_user_prompt(inputs, enriched_chains)

    last_error: Exception | None = None
    for attempt in range(2):
        prompt = base_prompt
        if attempt > 0 and last_error is not None:
            prompt += (
                f"\n\nYour previous response was invalid: {last_error}. "
                "Return ONLY corrected JSON matching the schema, citing only "
                "the source filenames listed in the evidence above."
            )

        try:
            response = client.messages.parse(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=RecommendationsResponse,
            )
            result = response.parsed_output
            _validate_sources_are_grounded(result, valid_sources)
            return result, enriched_chains
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc

    raise RuntimeError(f"Failed to obtain valid recommendations after retry: {last_error}")


def compose_recommendations(
    inputs: dict[str, Any],
    client: anthropic.Anthropic | None = None,
) -> RecommendationsResponse:
    """Build one evidence-grounded prompt from `inputs` and return validated
    recommendations. See compose_recommendations_with_evidence() for a variant
    that also returns the chains/evidence used to build the prompt.
    """
    result, _ = compose_recommendations_with_evidence(inputs, client=client)
    return result
