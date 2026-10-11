"""Adapt the external generation window without vendoring distributed runtime code."""

from __future__ import annotations
from typing import Callable, Protocol, Sequence
from forge_posttraining.schemas import GenerationRequest, Sample, AugmentedBatch, SafetyError


class EasyR1Sampler:
    """Framework callbacks preserve original prompts, images, positions and actor provenance."""

    def __init__(self, build_requests: Callable, generate_sequences: Callable, decode_children: Callable):
        self.build_requests = build_requests
        self.generate_sequences = generate_sequences
        self.decode_children = decode_children

    def sample(self, requests: Sequence[GenerationRequest]) -> Sequence[Sequence[Sample]]:
        if not requests:
            return []
        if len({r.n for r in requests}) != 1:
            raise SafetyError("Batch must use one n setting")
        maximum = max(r.max_tokens for r in requests)
        # Builder pads request count for data-parallel workers; decoder discards padding.
        proto = self.build_requests(requests, maximum)
        generated = self.generate_sequences(proto)
        return self.decode_children(generated, requests)


class TrainingRuntime(Protocol):
    """Implemented by a configured external EasyR1/verl trainer factory."""

    tokenizer: object
    sampler: EasyR1Sampler
    verifier: object
    actor_revision: str

    def rollout_batches(self):
        """Yield direct n=8 groups for one configured epoch; no teacher rows."""
        ...

    def update_direct_and_sd(self, batch: AugmentedBatch, lambda_sd: float):
        """One optimizer/scheduler step, direct GRPO/KL + separately normalized SD."""
        ...

    def save(self):
        """Save the trained adapter to the caller's output directory."""
        ...
