"""Frozen P3 features and a single standardized linear binary terminality head."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class GCTHConfig:
    threshold: float = 0.5
    classifier: str = "linear"
    state_layer: str = "final_norm"
    question_pooling: str = "mean"
    sentence_boundary_regex: str = r"(?<=[.!?])\s+|\n+"
    seed: int = 20261009
    l2_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    standardization_epsilon: float = 1e-6
    max_iter: int = 500
    history_size: int = 20


class TerminalityHead(nn.Module):
    """p_term = sigmoid(w^T standardize([q;s]) + b), STOP at >= 0.5."""

    def __init__(self, hidden_size: int, threshold: float = 0.5):
        super().__init__()
        if hidden_size <= 0 or not 0 <= threshold <= 1:
            raise ValueError("Invalid GCTH dimensions or threshold")
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.linear = nn.Linear(2 * hidden_size, 1)
        self.register_buffer("feature_mean", torch.zeros(2 * hidden_size))
        self.register_buffer("feature_std", torch.ones(2 * hidden_size))

    def forward(self, question: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if question.shape != state.shape or question.shape[-1] != self.hidden_size:
            raise ValueError("q/s feature shape mismatch")
        features = torch.cat([question, state], dim=-1)
        return torch.sigmoid(self.linear((features - self.feature_mean) / self.feature_std)).squeeze(-1)

    def should_stop(self, question: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self(question, state) >= self.threshold

    def save(self, path: str | Path, config: dict, split_metadata: dict) -> None:
        """Portable JSON weights/statistics; no actor checkpoint is included."""
        data = dict(
            hidden_size=self.hidden_size,
            threshold=self.threshold,
            weights=self.linear.weight.detach().cpu().reshape(-1).tolist(),
            bias=float(self.linear.bias.detach().cpu()[0]),
            feature_mean=self.feature_mean.cpu().tolist(),
            feature_std=self.feature_std.cpu().tolist(),
            config=config,
            split_metadata=split_metadata,
        )
        with Path(path).open("x") as stream:
            json.dump(data, stream, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TerminalityHead":
        data = json.loads(Path(path).read_text())
        head = cls(data["hidden_size"], data["threshold"])
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor(data["weights"]).reshape(1, -1))
            head.linear.bias.fill_(data["bias"])
            head.feature_mean.copy_(torch.tensor(data["feature_mean"]))
            head.feature_std.copy_(torch.tensor(data["feature_std"]))
        if not torch.isfinite(head.feature_std).all() or (head.feature_std <= 0).any():
            raise ValueError("Invalid feature standardization")
        return head.eval()


def pooled_features(
    hidden: torch.Tensor, question_positions: Sequence[int], state_position: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final language-model norm vectors: mean question tokens and boundary's last token."""
    if (
        hidden.ndim != 2
        or not question_positions
        or min(question_positions) < 0
        or max(question_positions) >= len(hidden)
    ):
        raise ValueError("Invalid final-norm question span")
    if not 0 <= state_position < len(hidden):
        raise ValueError("Invalid state boundary")
    return hidden[list(question_positions)].float().mean(0), hidden[state_position].float()


def identity_splits(cad_ids: Sequence[str]) -> dict[str, list[int]]:
    """Train-only CAD identities: 20% controller-heldout, 20% of remainder inner."""
    h = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 100
    split = {"fit": [], "inner": [], "heldout": []}
    for i, cad in enumerate(cad_ids):
        if not cad:
            raise ValueError("CAD identity is required")
        split["heldout" if h(cad) < 20 else "inner" if h("inner:" + cad) < 20 else "fit"].append(i)
    return split


def balanced_indices(indices: Sequence[int], labels: np.ndarray, seed: int) -> list[int]:
    """Exact deterministic equal-class subsampling used for controller fitting."""
    pos = [i for i in indices if labels[i] == 1]
    neg = [i for i in indices if labels[i] == 0]
    if not pos or not neg:
        raise ValueError("Fitting split needs terminal and non-terminal labels")
    rng = random.Random(seed)
    k = min(len(pos), len(neg))
    return sorted(rng.sample(pos, k) + rng.sample(neg, k))


def fit_logreg(
    features: np.ndarray, labels: np.ndarray, l2: float, config: GCTHConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE + 0.5*lambda*||w||²; no bias penalty; LBFGS strong-Wolfe."""
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, b],
        lr=1,
        max_iter=config.max_iter,
        history_size=config.history_size,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + 0.5 * l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def balanced_accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    """Mean terminal/non-terminal recall on the inner CAD split only."""
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("Inner selection split requires both classes")
    return float(((prediction[labels == 1] == 1).mean() + (prediction[labels == 0] == 0).mean()) / 2)


def train_linear_head(
    question: np.ndarray,
    state: np.ndarray,
    labels: np.ndarray,
    cad_ids: Sequence[str],
    config: GCTHConfig | None = None,
) -> tuple[TerminalityHead, dict]:
    """Train the frozen P3 architecture; select regularization on inner identities, then refit."""
    config = config or GCTHConfig()
    if (
        config.classifier != "linear"
        or config.state_layer != "final_norm"
        or config.question_pooling != "mean"
    ):
        raise ValueError("Only the frozen P3 linear design is supported")
    if (
        question.shape != state.shape
        or question.ndim != 2
        or len(labels) != len(question)
        or len(cad_ids) != len(labels)
    ):
        raise ValueError("Training feature/label alignment failed")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(question).all() or not np.isfinite(state).all():
        raise ValueError("Invalid verified training labels/features")
    torch.manual_seed(config.seed)
    features = np.concatenate([question, state], 1).astype(np.float32)
    splits = identity_splits(cad_ids)
    fit = balanced_indices(splits["fit"], labels, config.seed)
    inner = splits["inner"]
    if not inner or not splits["heldout"]:
        raise ValueError("Train pool must populate disjoint fit, inner and controller-heldout CAD splits")
    mean = features[fit].mean(0)
    std = features[fit].std(0) + config.standardization_epsilon
    scores = {}
    for regularization in config.l2_grid:
        w, b = fit_logreg((features[fit] - mean) / std, labels[fit], regularization, config)
        pred = torch.sigmoid(torch.tensor((features[inner] - mean) / std) @ w + b).numpy() >= config.threshold
        scores[regularization] = balanced_accuracy(pred, labels[inner])
    chosen = max(config.l2_grid, key=lambda value: (scores[value], -abs(np.log10(value) + 2)))
    train = balanced_indices(splits["fit"] + inner, labels, config.seed)
    mean = features[train].mean(0)
    std = features[train].std(0) + config.standardization_epsilon
    w, b = fit_logreg((features[train] - mean) / std, labels[train], chosen, config)
    head = TerminalityHead(question.shape[1], config.threshold)
    with torch.no_grad():
        head.linear.weight.copy_(w.reshape(1, -1))
        head.linear.bias.copy_(b)
        head.feature_mean.copy_(torch.tensor(mean))
        head.feature_std.copy_(torch.tensor(std))
    metadata = dict(
        seed=config.seed,
        l2=chosen,
        inner_balanced_accuracy={str(k): v for k, v in scores.items()},
        identity_groups={name: sorted({cad_ids[i] for i in indices}) for name, indices in splits.items()},
        state_counts={name: len(indices) for name, indices in splits.items()},
        balanced_fit_states=len(fit),
        balanced_controller_train_states=len(train),
        train_only=True,
        parameters=sum(p.numel() for p in head.parameters()),
    )
    return head.eval(), metadata
