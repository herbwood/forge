"""Signal-free, sampled terminal closure acceptance and minimal selection."""

from dataclasses import replace
import pytest
from forge_posttraining.adapters import forge as verifier
from forge_posttraining import vtc
from forge_posttraining.schemas import SafetyError
from conftest import CLOSED, CLOSURE, DIAMETERS
from test_orsd import controller


def evaluate(text, tokenizer, payload, graph):
    return vtc.evaluate_closure(
        CLOSED,
        verifier.verify_text(CLOSED, graph).closed,
        tokenizer.encode(text),
        tokenizer,
        True,
        payload,
        graph,
        verifier,
    )


def test_terminal_trigger(graph):
    assert not vtc.terminal_trigger(verifier.verify_text(DIAMETERS, graph).closed, graph)
    assert vtc.terminal_trigger(verifier.verify_text(CLOSED, graph).closed, graph)


def test_only_z_one(tokenizer, payload, graph):
    assert evaluate(CLOSURE, tokenizer, payload, graph)["Z"] == 1
    assert evaluate("\n</think>\nFINAL: Invented wrong part", tokenizer, payload, graph)["Z"] == 0
    assert evaluate(" The answer is a rotor.", tokenizer, payload, graph)["Z"] == 0
    assert evaluate(" F01 has diameter 13 mm." + CLOSURE, tokenizer, payload, graph)["Z"] == 0


def test_shortest_closure_and_minimal_cut(tokenizer, payload, graph):
    short = evaluate(CLOSURE, tokenizer, payload, graph)
    long = evaluate(" More redundant reasoning." + CLOSURE, tokenizer, payload, graph)
    assert vtc.select_closure([long, short]) == 1
    extra = evaluate(CLOSURE + "\nUnneeded continuation.", tokenizer, payload, graph)
    assert extra["closure_tokens"] == short["closure_tokens"] < extra["tokens"]


def test_terminal_only_retry_and_one_closure(tokenizer, groups):
    wrong = replace(groups[0], rollout_ids=(tuple(tokenizer.encode(CLOSED)),) * 8)

    def callback(request, child):
        assert request.logit_bias is None
        assert "Next reasoning operation" not in tokenizer.decode(request.context_ids)
        return CLOSURE if request.n == 16 and child == 8 else " No answer."

    ctl, sampler = controller(tokenizer, callback)
    batch = ctl.augment([wrong, groups[1]])
    assert len(batch.targets) == 1 and batch.targets[0].channel == "close"
    assert batch.targets[0].child_index == 9
    assert [r.n for r in sampler.requests] == [8, 16]


def test_reserved_control_rejected():
    with pytest.raises(SafetyError):
        vtc.require_signal_free("Next reasoning operation: compare_diameter.")
