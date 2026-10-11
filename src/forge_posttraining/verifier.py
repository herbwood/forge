"""Generic verifier interface; no verifier is used by the inference controller."""

from __future__ import annotations
from typing import Any, Callable, Protocol
from .ontology import OntologyGraph, PrefixState, Verification

VERIFIED, MISSING, CONTRADICTION = "verified", "missing", "contradiction"


class Verifier(Protocol):
    """Strict training-time checks; gold values are never generation instructions."""

    CONTRADICTION: str
    _FINAL_LINE: Any

    def build_graph(self, payload: dict) -> OntologyGraph: ...
    def verify_text(self, text: str, graph: OntologyGraph) -> Verification: ...
    def node_status(self, text: str, graph: OntologyGraph, node_id: str) -> tuple[str, list]: ...
    def reasoning_end(self, text: str) -> int: ...
    def sentence_cuts(self, text: str, start: int, end: int) -> list[int]: ...
    def select_reset_state(self, responses: list[str], graph: OntologyGraph, token_len: list[Callable]): ...
    def evaluate_suffix(
        self, prefix: str, suffix: str, graph: OntologyGraph, state: PrefixState, payload: dict
    ): ...
    def depth_advance(self, original: PrefixState, closed: frozenset[str], graph: OntologyGraph) -> int: ...
    def answer_correct(self, response: str, payload: dict) -> float: ...
    def format_reward(self, response: str) -> float: ...


def verify_graph(
    text: str, graph: OntologyGraph, checker: Callable[[str, Any], tuple[str, list]]
) -> Verification:
    """Apply a domain-specific strict fact checker, then prerequisite closure."""
    status, evidence = {}, {}
    for key in graph.required:
        status[key], evidence[key] = checker(text, graph.nodes[key])
        if status[key] not in (VERIFIED, MISSING, CONTRADICTION):
            raise ValueError("Unknown verification status")
    closed = graph.closure({key for key, value in status.items() if value == VERIFIED})
    return Verification(status, evidence, closed, graph.frontier(closed), graph.rank(closed))
