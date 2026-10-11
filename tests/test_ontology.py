"""Strict prerequisites, terminality and contradiction checks on invented CAD facts."""

import pytest
from forge_posttraining.adapters import forge as verifier
from forge_posttraining.ontology import OntologyGraphError, graph_from_dict
from forge_posttraining.verifier import verify_graph
from conftest import CLOSED, COMPARE, DIAMETERS


def test_terminal_and_frontier(graph):
    before = verifier.verify_text(DIAMETERS, graph)
    assert graph.terminal not in before.closed
    assert {graph.nodes[n].label for n in before.frontier} == {"compare_diameter"}
    after = verifier.verify_text(CLOSED, graph)
    assert graph.terminal in after.closed and not after.frontier


def test_transition_and_earliest_cut(graph, payload):
    state = verifier.best_prefix_state(DIAMETERS, graph)
    accepted = verifier.evaluate_suffix(DIAMETERS, COMPARE + " Extra words.", graph, state, payload)
    assert accepted.E == 1 and accepted.first_transition_cut == len(COMPARE)
    assert verifier.evaluate_suffix(DIAMETERS, " More thinking.", graph, state, payload).E == 0


def test_contradictions_reject_credit(graph, payload):
    state = verifier.best_prefix_state(DIAMETERS, graph)
    invalid = verifier.evaluate_suffix(DIAMETERS, COMPARE + " F01 has diameter 13 mm.", graph, state, payload)
    assert invalid.E == 0 and invalid.e_reason == "state_contradicted"
    assert graph.terminal not in verifier.verify_text(CLOSED + " F01 has diameter 13 mm.", graph).closed


def test_generic_dag_and_domain_checker():
    graph = graph_from_dict(
        {
            "terminal": "done",
            "nodes": [
                {"id": "fact", "label": "relation"},
                {"id": "done", "label": "terminal", "prereqs": ["fact"]},
            ],
        }
    )
    result = verify_graph("invented", graph, lambda text, node: ("verified", []))
    assert result.closed == frozenset({"fact", "done"})
    assert graph.closure({"done"}) == frozenset()


@pytest.mark.parametrize(
    "data",
    [
        {"terminal": "x", "nodes": []},
        {"terminal": "x", "nodes": [{"id": "x", "label": "r", "prereqs": ["x"]}]},
        {"terminal": "x", "nodes": [{"id": "x", "label": "r", "prereqs": ["missing"]}]},
    ],
)
def test_malformed_dag_fails_closed(data):
    with pytest.raises(OntologyGraphError):
        graph_from_dict(data)
