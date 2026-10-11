"""Prerequisite DAGs and strictly verified reasoning states."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


class OntologyGraphError(ValueError):
    """An external ontology is malformed or inconsistent."""


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    label: str
    feature: Optional[str]
    slot: Optional[str]
    gold: Any
    prereqs: tuple[str, ...]
    depth: int


@dataclass
class OntologyGraph:
    nodes: dict[str, Node]
    terminal: str
    required: frozenset[str]
    facts: dict[str, dict[str, Any]]
    operations: list[dict[str, Any]]
    op_candidates: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def closure(self, verified: set[str]) -> frozenset[str]:
        closed: set[str] = set()
        for nid in sorted(self.required, key=lambda n: self.nodes[n].depth):
            if nid in verified and all((p in closed for p in self.nodes[nid].prereqs)):
                closed.add(nid)
        return frozenset(closed)

    def frontier(self, closed: frozenset[str]) -> frozenset[str]:
        return frozenset(
            (
                n
                for n in self.required
                if n not in closed and all((p in closed for p in self.nodes[n].prereqs))
            )
        )

    def rank(self, closed: frozenset[str]) -> tuple[int, int]:
        return (max((self.nodes[n].depth for n in closed), default=0), len(closed))


@dataclass
class PrefixState:
    cut: int
    closed: frozenset[str]
    frontier: frozenset[str]
    rank: tuple[int, int]
    token_len: int


@dataclass
class Verification:
    status: dict[str, str]
    evidence: dict[str, list]
    closed: frozenset[str]
    frontier: frozenset[str]
    rank: tuple[int, int]


VerifiedState = PrefixState


def graph_from_dict(data: dict[str, Any]) -> OntologyGraph:
    """Build a generic supplied DAG; terminal dependencies define required nodes."""
    raw = {n["id"]: n for n in data["nodes"]}
    if len(raw) != len(data["nodes"]) or data["terminal"] not in raw:
        raise OntologyGraphError("Duplicate node or missing terminal")
    depths: dict[str, int] = {}

    def depth(node_id: str, stack: tuple[str, ...] = ()) -> int:
        if node_id not in raw or node_id in stack:
            raise OntologyGraphError("Missing prerequisite or cycle")
        if node_id not in depths:
            depths[node_id] = 1 + max(
                (depth(p, stack + (node_id,)) for p in raw[node_id].get("prereqs", [])), default=0
            )
        return depths[node_id]

    nodes = {
        key: Node(
            key,
            n.get("kind", "fact"),
            n["label"],
            n.get("feature"),
            n.get("slot"),
            n.get("gold"),
            tuple(sorted(n.get("prereqs", []))),
            depth(key),
        )
        for key, n in raw.items()
    }
    required: set[str] = set()
    todo = [data["terminal"]]
    while todo:
        key = todo.pop()
        if key not in required:
            required.add(key)
            todo.extend(nodes[key].prereqs)
    return OntologyGraph(
        nodes, data["terminal"], frozenset(required), data.get("facts", {}), data.get("operations", [])
    )
