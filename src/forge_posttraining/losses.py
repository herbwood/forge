"""Final independent routing/closure numerators with shared actor normalization."""

from __future__ import annotations
from collections import Counter
from typing import Sequence
import math
import torch
from .schemas import DistillationTarget, SafetyError


def target_weights(targets: Sequence[DistillationTarget]) -> list[float]:
    """Routing chunk weight 1/J_route; one closure/prompt has token weight 1."""
    routing = Counter(t.prompt_id for t in targets if t.channel == "route")
    closures = Counter(t.prompt_id for t in targets if t.channel == "close")
    if any(n > 1 for n in closures.values()):
        raise SafetyError("More than one closure target per prompt")
    return [1.0 / routing[t.prompt_id] if t.channel == "route" else 1.0 for t in targets]


def sd_numerator(token_ce: Sequence[torch.Tensor], targets: Sequence[DistillationTarget]) -> torch.Tensor:
    """Sum raw target-token CE, average routing chunks per prompt, then sum prompts."""
    if len(token_ce) != len(targets):
        raise ValueError("CE rows and targets are not aligned")
    if not token_ce:
        return torch.tensor(0.0)
    weights = target_weights(targets)
    total = token_ce[0].sum() * 0
    for ce, target, weight in zip(token_ce, targets, weights):
        if ce.ndim != 1 or ce.numel() != len(target.target_ids) or not torch.isfinite(ce).all():
            raise SafetyError("Invalid target-token CE span")
        total = total + weight * ce.sum()
    return total


def selective_distillation_loss(
    token_ce: Sequence[torch.Tensor],
    targets: Sequence[DistillationTarget],
    actor_denominator: torch.Tensor | float,
    lambda_sd: float = 1.0,
) -> torch.Tensor:
    """lambda_SD * N_SD / D_actor. No target-length or combined-J division."""
    d = float(actor_denominator)
    if not math.isfinite(d) or d <= 0 or not math.isfinite(lambda_sd) or lambda_sd <= 0:
        raise SafetyError("Positive finite direct actor denominator and SD coefficient required")
    return lambda_sd * sd_numerator(token_ce, targets) / actor_denominator


def distributed_sd_loss(
    log_probs: torch.Tensor,
    token_weights: torch.Tensor,
    global_direct_denominator: torch.Tensor,
    world_size: int,
    lambda_sd: float = 1.0,
) -> torch.Tensor:
    """Local CE numerator scaled for DDP gradient averaging, matching the actor hook."""
    if log_probs.shape != token_weights.shape or world_size < 1:
        raise SafetyError("Invalid distributed SD tensor contract")
    if not torch.isfinite(token_weights).all() or (token_weights < 0).any():
        raise SafetyError("SD weights must be finite and nonnegative")
    if not torch.isfinite(global_direct_denominator) or global_direct_denominator <= 0:
        raise SafetyError("No direct actor denominator; SD-only fallback is forbidden")
    selected = token_weights > 0
    numerator = (-log_probs[selected].float() * token_weights[selected]).sum()
    return lambda_sd * numerator * world_size / global_direct_denominator


def direct_reward(answer: float, fmt: float, path: float, contradiction: float) -> float:
    """Exact outcome-gated frozen direct reward; path/contradiction clipped to [0,1]."""
    quality = max(0.0, min(1.0, path)) * (1 - max(0.0, min(1.0, contradiction)))
    return max(0.0, min(1.05, 0.9 * answer + 0.1 * fmt + 0.05 * answer * fmt * quality))
