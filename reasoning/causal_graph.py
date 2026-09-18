"""Causal graph engine for reasoning over biodiversity cause-effect relationships."""

import networkx as nx


class CausalGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_relationship(self, cause: str, effect: str, weight: float = 1.0) -> None:
        self.graph.add_edge(cause, effect, weight=weight)

    def get_effects(self, cause: str) -> list[str]:
        if cause not in self.graph:
            return []
        return list(self.graph.successors(cause))
