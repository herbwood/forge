"""Independent channel weighting, shared normalization and SD policy isolation."""

import pytest
import torch
from forge_posttraining.losses import (
    direct_reward,
    distributed_sd_loss,
    selective_distillation_loss,
    target_weights,
)
from forge_posttraining.schemas import DistillationTarget, SafetyError
from integrations.easyr1.trainer_hooks import direct_advantages_only, student_rows


def target(prompt, channel, length=2):
    return DistillationTarget(
        prompt, (), tuple(range(length)), channel, "synthetic", "actor", "request", 1, 1, 1.0, "", "invented"
    )


def test_closure_does_not_reweight_routing():
    routes = [target("p", "route"), target("p", "route")]
    assert target_weights(routes) == [0.5, 0.5]
    assert target_weights(routes + [target("p", "close")]) == [0.5, 0.5, 1.0]


def test_multiple_prompts_sum_before_shared_denominator():
    targets = [target("p", "route"), target("p", "route"), target("p", "close"), target("q", "close")]
    ces = [
        torch.tensor([1.0, 3.0]),
        torch.tensor([2.0, 4.0]),
        torch.tensor([5.0, 6.0]),
        torch.tensor([7.0, 8.0]),
    ]
    assert selective_distillation_loss(ces, targets, 10.0).item() == pytest.approx((5 + 11 + 15) / 10)
    assert selective_distillation_loss(ces[:2], targets[:2], 10.0).item() == 0.5
    assert selective_distillation_loss(ces[2:3], targets[2:3], 10.0).item() == pytest.approx(1.1)


def test_no_denominator_fallback_or_duplicate_closure():
    with pytest.raises(SafetyError):
        selective_distillation_loss([torch.ones(2)], [target("p", "close")], 0.0)
    with pytest.raises(SafetyError):
        target_weights([target("p", "close"), target("p", "close")])


def test_distributed_gradient_scaling():
    logprobs = torch.tensor([[-2.0, -3.0, -99.0]], requires_grad=True)
    value = distributed_sd_loss(logprobs, torch.tensor([[0.5, 1.0, 0.0]]), torch.tensor(8.0), 2)
    assert value.item() == 1.0
    value.backward()
    assert torch.equal(logprobs.grad, torch.tensor([[-0.125, -0.25, 0.0]]))


def test_outcome_gates_process_bonus():
    assert direct_reward(0, 1, 1, 0) == 0.1
    assert direct_reward(1, 1, 1, 0) == 1.05
    assert direct_reward(1, 1, 1, 1) == 1.0


def test_student_masks_and_advantages():
    original = {
        "p": dict(
            prompts=torch.tensor([0, 0, 8, 9]),
            prompt_attention_mask=torch.tensor([0, 0, 1, 1]),
            position_ids=torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5]]).repeat(3, 1),
            multi_modal_data={"synthetic": True},
        )
    }
    rows = student_rows([target("p", "route"), target("p", "close")], original, 4, 0)
    for row in rows:
        assert row["forge_prompt_weight"] == 0 and row["advantages"].count_nonzero() == 0
        assert row["forge_sd_weight"].tolist() == [1.0, 1.0, 0.0, 0.0]
        assert row["response_mask"].tolist() == [1, 1, 0, 0]
        assert row["multi_modal_data"] is original["p"]["multi_modal_data"]
    scores = torch.ones(3, 4)
    advantages, returns = direct_advantages_only(
        ["normal", "route", "close"], lambda indices: (scores[indices] * 2, scores[indices] * 3), scores
    )
    assert advantages[0].sum() == 8 and advantages[1:].count_nonzero() == 0
    assert returns[0].sum() == 12 and returns[1:].count_nonzero() == 0
