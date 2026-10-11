"""Minimal position-preserving student rows and actor loss integration hooks."""

from __future__ import annotations
from typing import Sequence
import torch
from forge_posttraining.schemas import DistillationTarget, SafetyError
from forge_posttraining.losses import target_weights, distributed_sd_loss


def extend_prompt(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    position_ids: torch.Tensor,
    prefix_ids: Sequence[int],
    pad_token_id: int,
):
    """Preserve all original multimodal rotary positions while moving prefix to prompt."""
    width = prompt_ids.size(-1)
    keep = prompt_mask.bool()
    body = prompt_ids[keep]
    prefix = torch.as_tensor(prefix_ids, dtype=prompt_ids.dtype, device=prompt_ids.device)
    ids = torch.cat([body, prefix])
    if ids.numel() > width:
        raise SafetyError("Student context exceeds prompt width")
    pos = torch.cat(
        [position_ids[..., :width][..., keep], position_ids[..., width : width + len(prefix_ids)]], dim=-1
    )
    if pos.shape[-1] != ids.numel():
        raise SafetyError("Original response positions missing for prefix")
    pad = width - ids.numel()
    new_ids = torch.cat([torch.full((pad,), pad_token_id, dtype=ids.dtype, device=ids.device), ids])
    mask = torch.cat(
        [
            torch.zeros(pad, dtype=prompt_mask.dtype, device=prompt_mask.device),
            torch.ones(ids.numel(), dtype=prompt_mask.dtype, device=prompt_mask.device),
        ]
    )
    positions = torch.cat(
        [torch.zeros(pos.shape[:-1] + (pad,), dtype=pos.dtype, device=pos.device), pos], dim=-1
    )
    return new_ids, mask, positions


def student_rows(
    targets: Sequence[DistillationTarget], original_rows: dict, response_width: int, pad_token_id: int
) -> list[dict]:
    """Build suffix-only SD tensors; teacher search rows are never training examples."""
    output = []
    for target, weight in zip(targets, target_weights(targets)):
        if not target.target_ids or len(target.target_ids) > response_width or target.evidence != 1:
            raise SafetyError("Student target lacks verified sampled tokens")
        if target.prompt_weight != 0 or target.advantage != 0 or target.include_in_policy_loss:
            raise SafetyError("SD row entered a direct policy objective")
        original = original_rows[target.prompt_id]
        ids, mask, pos = extend_prompt(
            original["prompts"],
            original["prompt_attention_mask"],
            original["position_ids"],
            target.context_ids,
            pad_token_id,
        )
        response = torch.full((response_width,), pad_token_id, dtype=ids.dtype, device=ids.device)
        response[: len(target.target_ids)] = torch.tensor(
            target.target_ids, dtype=ids.dtype, device=ids.device
        )
        rmask = torch.zeros(response_width, dtype=mask.dtype, device=mask.device)
        rmask[: len(target.target_ids)] = 1
        rpos = pos[..., -1:] + torch.arange(1, response_width + 1, device=pos.device).view(
            *([1] * (pos.ndim - 1)), -1
        )
        sdw = torch.zeros(response_width, dtype=torch.float32, device=ids.device)
        sdw[: len(target.target_ids)] = weight
        output.append(
            dict(
                prompts=ids,
                responses=response,
                input_ids=torch.cat([ids, response]),
                attention_mask=torch.cat([mask, rmask]),
                response_mask=rmask,
                position_ids=torch.cat([pos, rpos], -1),
                forge_prompt_weight=torch.tensor(0.0, device=ids.device),
                forge_sd_weight=sdw,
                token_level_scores=torch.zeros(response_width, device=ids.device),
                advantages=torch.zeros(response_width, device=ids.device),
                returns=torch.zeros(response_width, device=ids.device),
                channel=target.channel,
                original_prompt_id=target.prompt_id,
                multi_modal_data=original["multi_modal_data"],
                actor_revision=target.model_revision,
            )
        )
    return output


def actor_objective(
    normalized_direct_loss,
    log_probs,
    sd_weights,
    global_direct_denominator,
    world_size: int,
    lambda_sd: float = 1.0,
):
    """Add SD to the already correctly normalized direct GRPO/KL objective."""
    return normalized_direct_loss + distributed_sd_loss(
        log_probs, sd_weights, global_direct_denominator, world_size, lambda_sd
    )


def direct_advantages_only(row_kinds: Sequence[str], compute_direct: callable, token_scores: torch.Tensor):
    """Delegate advantages for retained direct rows, leave all SD entries exactly zero."""
    indices = [i for i, kind in enumerate(row_kinds) if kind == "normal"]
    advantages = torch.zeros_like(token_scores)
    returns = torch.zeros_like(token_scores)
    if indices:
        direct_adv, direct_returns = compute_direct(indices)
        advantages[indices] = direct_adv
        returns[indices] = direct_returns
    return advantages, returns


def reward_hook(reward_inputs: list[dict], **kwargs):
    """Active CAD parsing and exact direct objective; use as EasyR1 reward_function."""
    from forge_posttraining.adapters._answer import (
        parse_ground_truth,
        answer_correct,
        format_reward,
        reasoning_part,
    )
    from forge_posttraining.adapters._text_grammar import _path_diag
    from forge_posttraining.losses import direct_reward

    output = []
    for item in reward_inputs:
        text = str(item["response"])
        payload = parse_ground_truth(item["ground_truth"])
        a = float(answer_correct(text, payload))
        f = float(format_reward(text))
        diagnostics = _path_diag(reasoning_part(text), payload)
        path = diagnostics["path"]
        contradiction = diagnostics["contradiction"]
        quality = max(0.0, min(1.0, path)) * (1 - max(0.0, min(1.0, contradiction)))
        bonus = 0.05 * a * f * quality
        output.append(
            dict(
                **diagnostics,
                overall=direct_reward(a, f, path, contradiction),
                answer=a,
                format=f,
                vanilla=0.9 * a + 0.1 * f,
                process_quality=quality,
                process_bonus=bonus,
                wrong_process_bonus=bonus * (1 - a),
            )
        )
    return output
