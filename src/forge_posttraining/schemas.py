"""Framework-neutral contracts for model-generated search and student targets."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence


class SafetyError(RuntimeError):
    """A frozen method invariant failed; training must not silently fall back."""


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    first_n: int = 8
    retry_n: int = 16


@dataclass(frozen=True)
class GenerationRequest:
    """No answer/ontology payload. Context is a model prefix plus teacher signal."""

    request_id: str
    prompt_id: str
    prompt_ids: tuple[int, ...]
    context_ids: tuple[int, ...]
    multimodal: Any
    n: int
    max_tokens: int
    sampling: SamplingConfig
    actor_revision: str
    logit_bias: dict[int, float] | None = None


@dataclass(frozen=True)
class Sample:
    """A child sampled by the current actor with attributable input/provenance."""

    token_ids: tuple[int, ...]
    request_id: str
    prompt_ids: tuple[int, ...]
    context_ids: tuple[int, ...]
    actor_revision: str


class Sampler(Protocol):
    def sample(self, requests: Sequence[GenerationRequest]) -> Sequence[Sequence[Sample]]:
        """Generate n children/request. Child i keeps the same seed in n=8/n=16."""
        ...


@dataclass(frozen=True)
class DirectGroup:
    prompt_id: str
    prompt_ids: tuple[int, ...]
    rollout_ids: tuple[tuple[int, ...], ...]
    answer_correct: tuple[float, ...]
    payload: dict[str, Any]
    multimodal: Any = None


@dataclass(frozen=True)
class DistillationTarget:
    prompt_id: str
    context_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    channel: Literal["route", "close"]
    relation: str
    model_revision: str
    request_id: str
    child_index: int
    level: int
    evidence: float
    context_text: str
    target_text: str
    signal_ids: tuple[int, ...] = ()
    prompt_weight: float = 0.0
    advantage: float = 0.0
    include_in_policy_loss: bool = False


@dataclass
class AugmentedBatch:
    retained_direct: list[DirectGroup] = field(default_factory=list)
    targets: list[DistillationTarget] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    search_requests: list[GenerationRequest] = field(default_factory=list)
