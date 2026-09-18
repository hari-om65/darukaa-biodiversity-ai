"""Causal graph reasoning engine.

Loads reasoning/graph.yaml into a NetworkX digraph and reasoning/thresholds.yaml
into a list of trigger conditions. `analyze(inputs)` evaluates the thresholds
against a set of input values and walks up to two hops downstream through the
causal graph from each triggered variable, returning the impact chains found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import yaml

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "graph.yaml"
THRESHOLDS_PATH = BASE_DIR / "thresholds.yaml"

MAX_HOPS = 2

_OPERATORS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

_RELATION_SIGN = {"positive": 1, "negative": -1}
_STRENGTH_RANK = {"low": 0, "medium": 1, "high": 2}


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    graph = nx.DiGraph()
    for edge in data.get("edges", []):
        graph.add_edge(edge["from"], edge["to"], relation=edge["relation"], strength=edge["strength"])
    return graph


def load_thresholds(path: Path = THRESHOLDS_PATH) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("thresholds", [])


GRAPH = load_graph()
THRESHOLDS = load_thresholds()


def evaluate_thresholds(
    inputs: dict[str, Any], thresholds: list[dict[str, Any]] = THRESHOLDS
) -> list[dict[str, Any]]:
    """Return the threshold entries whose condition is met by `inputs`."""
    triggered = []
    for threshold in thresholds:
        variable = threshold["variable"]
        if variable not in inputs:
            continue
        op = _OPERATORS[threshold["operator"]]
        if op(inputs[variable], threshold["value"]):
            triggered.append(threshold)
    return triggered


def _compose_relation(edges: list[dict[str, Any]]) -> str:
    sign = 1
    for edge in edges:
        sign *= _RELATION_SIGN.get(edge["relation"], 1)
    return "positive" if sign >= 0 else "negative"


def _compose_strength(edges: list[dict[str, Any]]) -> str:
    weakest = min(edges, key=lambda edge: _STRENGTH_RANK.get(edge["strength"], 0))
    return weakest["strength"]


def _walk_downstream(graph: nx.DiGraph, start: str, max_hops: int = MAX_HOPS) -> list[dict[str, Any]]:
    """Walk up to `max_hops` downstream from `start`.

    Returns one entry per chain segment reached (both 1-hop and 2-hop),
    each as {path, affected_metric, relation, strength}, where relation and
    strength are composed across the full chain from `start`.
    """
    chains: list[dict[str, Any]] = []
    if start not in graph:
        return chains

    def dfs(node: str, path: list[str], edges: list[dict[str, Any]], depth: int) -> None:
        if depth >= max_hops:
            return
        for neighbor in graph.successors(node):
            edge_data = graph.edges[node, neighbor]
            new_path = path + [neighbor]
            new_edges = edges + [edge_data]
            chains.append(
                {
                    "path": new_path,
                    "affected_metric": neighbor,
                    "relation": _compose_relation(new_edges),
                    "strength": _compose_strength(new_edges),
                }
            )
            dfs(neighbor, new_path, new_edges, depth + 1)

    dfs(start, [start], [], 0)
    return chains


def analyze(
    inputs: dict[str, Any],
    graph: nx.DiGraph = GRAPH,
    thresholds: list[dict[str, Any]] = THRESHOLDS,
) -> list[dict[str, Any]]:
    """Evaluate `inputs` against the configured thresholds and, for each
    triggered one, walk up to MAX_HOPS downstream through the causal graph.

    Returns a flat list of {path, affected_metric, relation, strength} dicts,
    one per downstream chain segment discovered across all triggered variables.
    """
    results: list[dict[str, Any]] = []
    for threshold in evaluate_thresholds(inputs, thresholds):
        node = threshold.get("node", threshold["variable"])
        results.extend(_walk_downstream(graph, node, MAX_HOPS))
    return results


def project_time_horizon(chain: dict[str, Any]) -> str:
    """Rough, heuristic-only projection of when a chain's effect might become
    observable - "short", "medium", or "long" term. This is NOT a calibrated
    forecast (no timeseries data backs it); it's a proxy derived purely from
    reasoning/graph.yaml's strength values and the chain's hop count, meant to
    give a directional sense of urgency alongside a chain's relation/strength.

    Heuristic:
        score = strength_rank(chain["strength"]) - (hops - 1)

    where strength_rank is low=0/medium=1/high=2 and hops = len(path) - 1.
    Each extra hop beyond the first docks one strength level, reflecting that
    an effect cascading through an intermediate variable plausibly takes
    longer to manifest than a direct one. The chain's strength is already the
    weakest link along its path (see _compose_strength), so a chain's horizon
    is bounded by both its least confident edge and its path length:

    - score >= 2 -> "short"  (only a 1-hop, high-strength chain qualifies -
      a strong, direct link should show up fastest, e.g. within a season/year)
    - score <= 0 -> "long"   (a low-strength chain at any hop count, or any
      2-hop chain no stronger than medium - weak signals and multi-step
      cascades both plausibly take years, e.g. soil carbon accumulation)
    - otherwise  -> "medium" (everything in between, roughly 1-3 years)
    """
    hops = len(chain["path"]) - 1
    score = _STRENGTH_RANK.get(chain["strength"], 0) - (hops - 1)

    if score >= 2:
        return "short"
    if score <= 0:
        return "long"
    return "medium"
