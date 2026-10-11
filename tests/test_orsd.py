"""All-wrong routing, model attribution, progressive search and row isolation."""

from dataclasses import replace
import pytest
from forge_posttraining.adapters import forge as verifier
from forge_posttraining.orsd import ORSDConfig, ORSDController, all_wrong, leakage_audit, signal_text
from forge_posttraining.schemas import SafetyError
from conftest import CLOSED, CLOSURE, COMPARE, COMPONENT, OWNER, ScriptedSampler


def controller(tokenizer, callback, **kwargs):
    sampler = ScriptedSampler(tokenizer, callback)
    # Toy word pieces differ from the paper tokenizer; production defaults are tested separately.
    config = ORSDConfig(relation_budgets={k: 128 for k in ORSDConfig().relation_budgets}, **kwargs)
    return ORSDController(tokenizer, sampler, verifier, "synthetic-current-actor", config), sampler


def progressive(tokenizer, request, child):
    if request.logit_bias is None:
        return CLOSURE
    text = tokenizer.decode(request.context_ids)
    return next(
        value
        for relation, value in [
            ("compare_diameter", COMPARE),
            ("owner_bom_item", OWNER),
            ("identifies_component", COMPONENT),
        ]
        if relation in text
    )


def test_trigger():
    assert all_wrong((0,) * 8)
    assert not all_wrong((0,) * 7 + (1,))
    assert not all_wrong((1,) * 8)
    with pytest.raises(SafetyError):
        all_wrong((0,) * 7)


def test_operation_only_signal(payload, graph):
    text = signal_text("synthetic prefix.", "compare_diameter")
    assert "compare_diameter" in text and not leakage_audit(text, payload, graph)
    assert all(str(value) not in text for value in ("42.25", "29.75", "F01", payload["canonical_answer"]))
    with pytest.raises(SafetyError):
        signal_text("", "filter_quantity_gt")
    with pytest.raises(SafetyError):
        signal_text("", "unrecognized_relation")


def test_progressive_targets_signal_free(tokenizer, groups):
    ctl, sampler = controller(tokenizer, lambda request, child: progressive(tokenizer, request, child))
    batch = ctl.augment(groups)
    route = [t for t in batch.targets if t.channel == "route"]
    close = [t for t in batch.targets if t.channel == "close"]
    assert [t.relation for t in route] == ["compare_diameter", "owner_bom_item", "identifies_component"]
    assert len(close) == 1 and close[0].evidence == 1
    assert batch.retained_direct == [groups[1]]
    for target in batch.targets:
        assert target.evidence == 1 and target.prompt_weight == target.advantage == 0
        assert not target.include_in_policy_loss
        assert "Next reasoning operation" not in tokenizer.decode(target.context_ids + target.target_ids)
        assert target.model_revision == "synthetic-current-actor"
        request = next(r for r in sampler.requests if r.request_id == target.request_id)
        sampled = progressive(tokenizer, request, target.child_index - 1)
        assert tokenizer.encode(sampled)[: len(target.target_ids)] == list(target.target_ids)
    assert close[0].context_ids == route[-1].context_ids + route[-1].target_ids
    closure_requests = [r for r in sampler.requests if r.logit_bias is None]
    assert len(closure_requests) == 1 and closure_requests[0].max_tokens == 58


def test_retry_retains_only_children_nine_to_sixteen(tokenizer, groups):
    def callback(request, child):
        return COMPARE if request.n == 16 and child == 8 else " No verified information."

    ctl, sampler = controller(tokenizer, callback, max_depth_advance=1, enable_vtc=False)
    batch = ctl.augment(groups)
    assert [r.n for r in sampler.requests] == [8, 16]
    assert len(batch.targets) == 1 and batch.targets[0].child_index == 9


def test_invalid_children_never_become_targets(tokenizer, groups):
    ctl, sampler = controller(
        tokenizer, lambda request, child: " F02 has the larger diameter.", enable_vtc=False
    )
    batch = ctl.augment(groups)
    assert not batch.targets and [r.n for r in sampler.requests] == [8, 16]


def test_retry_does_not_reuse_first_eight_children(tokenizer, groups):
    ctl, _ = controller(
        tokenizer,
        lambda request, child: COMPARE if request.n == 16 and child < 8 else " No verified information.",
        enable_vtc=False,
    )
    assert not ctl.augment(groups).targets


def test_teacher_control_copy_cannot_be_student_target(tokenizer, groups):
    ctl, _ = controller(
        tokenizer,
        lambda request, child: " Next reasoning operation: compare_diameter. Continue reasoning." + COMPARE,
        enable_vtc=False,
    )
    assert not ctl.augment(groups).targets


def test_routing_target_is_earliest_sampled_transition(tokenizer, groups):
    ctl, _ = controller(
        tokenizer,
        lambda request, child: COMPARE + " Redundant later sentence.",
        max_depth_advance=1,
        enable_vtc=False,
    )
    targets = ctl.augment(groups).targets
    assert len(targets) == 1 and targets[0].target_text == COMPARE


def test_mask_violations_fail_closed(tokenizer, groups):
    ctl, _ = controller(tokenizer, lambda request, child: COMPARE + "</think>")
    with pytest.raises(SafetyError, match="Blocked termination"):
        ctl.augment(groups)


def test_current_actor_provenance_required(tokenizer, groups):
    ctl, sampler = controller(tokenizer, lambda request, child: COMPARE)
    original = sampler.sample
    sampler.sample = lambda requests: [
        [replace(s, actor_revision="stale") for s in group] for group in original(requests)
    ]
    with pytest.raises(SafetyError, match="current actor"):
        ctl.augment(groups)


def test_no_sd_only_update(tokenizer, groups):
    ctl, _ = controller(tokenizer, lambda request, child: " No verified information.")
    with pytest.raises(SafetyError, match="No retained direct group"):
        ctl.augment(groups[:1])


def test_progressive_terminal_context_preferred(tokenizer, groups):
    wrong = replace(groups[0], rollout_ids=groups[0].rollout_ids[:7] + (tuple(tokenizer.encode(CLOSED)),))
    ctl, _ = controller(tokenizer, lambda request, child: progressive(tokenizer, request, child))
    batch = ctl.augment([wrong, groups[1]])
    close = [t for t in batch.targets if t.channel == "close"]
    assert len(close) == 1 and close[0].level > 0


def test_paper_budgets():
    config = ORSDConfig()
    assert config.relation_budgets == {
        "owner_bom_item": 44,
        "has_diameter": 127,
        "compare_diameter": 30,
        "compare_quantity": 39,
        "has_quantity": 21,
        "identifies_component": 17,
    }
    assert config.sampling.first_n == 8 and config.sampling.retry_n == 16
    assert (config.sampling.temperature, config.sampling.top_p, config.sampling.top_k) == (1.0, 0.95, -1)
