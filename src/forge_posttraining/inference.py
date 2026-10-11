"""Causal offline-equivalent replay. No ontology, verifier or gold inputs."""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Callable, Sequence
from .utils import TokenAligner


@dataclass(frozen=True)
class ReplayResult:
    token_ids: tuple[int, ...]
    stopped: bool
    stop_token_count: int | None
    probabilities: tuple[float, ...]
    forced_token_ids: tuple[int, ...] = ()


def sentence_boundaries(
    tokenizer, token_ids: Sequence[int], boundary_regex: str = r"(?<=[.!?])\s+|\n+"
) -> list[int]:
    """Frozen rule R, exact token-aligned causal boundaries before termination."""
    aligner = TokenAligner(tokenizer, list(token_ids), True)
    text = aligner.text
    end = text.find("</think>")
    if end < 0:
        final = re.search(r"(?im)^\s*FINAL\s*:", text)
        end = final.start() if final else len(text)
    reasoning = text[:end]
    out = []
    seen = set()
    close = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close) != 1:
        raise ValueError("STOP must be exactly one closing-think token for this actor")
    for match in re.finditer(boundary_regex, reasoning):
        cut = match.start()
        if cut >= len(text) or not reasoning[:cut].strip():
            continue
        count = aligner.tokens_for_prefix(reasoning[:cut])
        if count is None or count in seen or close[0] in token_ids[:count]:
            continue
        seen.add(count)
        out.append(count)
    return out


def replay(
    tokenizer,
    actor_reasoning_ids: Sequence[int],
    score_prefix: Callable[[tuple[int, ...]], float],
    resume_greedy: Callable[[tuple[int, ...], int], Sequence[int]],
    max_response_tokens: int = 4096,
    threshold: float = 0.5,
    boundary_regex: str = r"(?<=[.!?])\s+|\n+",
) -> ReplayResult:
    """First STOP discards all future original tokens; force only </think>, then resume."""
    original = tuple(actor_reasoning_ids)
    probabilities = []
    if len(original) > max_response_tokens:
        raise ValueError("Original response exceeds shared token budget")
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid controller threshold")
    for count in sentence_boundaries(tokenizer, original, boundary_regex):
        # The scoring callback receives only this prefix, never future generated tokens.
        probability = float(score_prefix(original[:count]))
        probabilities.append(probability)
        if not 0 <= probability <= 1:
            raise ValueError("Invalid terminality probability")
        if probability < threshold:
            continue
        forced = tuple(tokenizer.encode("</think>", add_special_tokens=False))
        if len(forced) != 1:
            raise ValueError("STOP encoding must contain exactly one token")
        prefix = original[:count] + forced
        remaining = max_response_tokens - len(prefix)
        if remaining < 0:
            raise ValueError("No budget for STOP token")
        suffix = tuple(resume_greedy(prefix, remaining)) if remaining else ()
        if len(suffix) > remaining:
            raise ValueError("Actor continuation exceeded remaining shared budget")
        return ReplayResult(prefix + suffix, True, count, tuple(probabilities), forced)
    return ReplayResult(original, False, None, tuple(probabilities))
