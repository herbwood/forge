"""Frozen linear features and causal first-STOP replay, using no real actor."""

import numpy as np
import torch
from forge_posttraining.inference import replay, sentence_boundaries
from forge_posttraining.terminality import (
    GCTHConfig,
    TerminalityHead,
    identity_splits,
    pooled_features,
    train_linear_head,
)
from forge_posttraining.model_io import FrozenActor
from forge_posttraining.utils import TokenAligner


def test_linear_shape_and_threshold():
    head = TerminalityHead(2560)
    assert sum(p.numel() for p in head.parameters()) == 5121
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.zero_()
    q, s = torch.zeros(2560), torch.zeros(2560)
    assert head(q, s) == 0.5 and head.should_stop(q, s)
    with torch.no_grad():
        head.linear.bias.fill_(-0.001)
    assert not head.should_stop(q, s)


def test_question_mean_and_boundary_state():
    hidden = torch.arange(24).reshape(6, 4)
    q, s = pooled_features(hidden, [1, 2], 4)
    assert torch.equal(q, hidden[1:3].float().mean(0))
    assert torch.equal(s, hidden[4].float())


def test_continue_unchanged(tokenizer):
    original = tuple(tokenizer.encode("First sentence. Second sentence.\n</think>\nFINAL: quantity=11"))
    result = replay(
        tokenizer,
        original,
        lambda prefix: 0.49,
        lambda prefix, budget: (_ for _ in ()).throw(AssertionError("Unexpected resume")),
    )
    assert not result.stopped and result.token_ids == original and not result.forced_token_ids


def test_stop_only_and_causal_prefix(tokenizer):
    original = tuple(
        tokenizer.encode("First sentence. Future secretless sentence.\n</think>\nFINAL: quantity=3")
    )
    scored, resumed = [], []

    def score(prefix):
        scored.append(prefix)
        assert "Future" not in tokenizer.decode(prefix)
        return 0.5

    def resume(prefix, budget):
        resumed.append((prefix, budget))
        assert tokenizer.decode(prefix) == "First sentence.</think>"
        return tokenizer.encode("\nFINAL: quantity=11")

    result = replay(tokenizer, original, score, resume, 100)
    assert result.stopped and len(scored) == 1
    assert result.forced_token_ids == tuple(tokenizer.encode("</think>"))
    assert "Future" not in tokenizer.decode(result.token_ids)
    assert resumed[0][1] == 100 - len(resumed[0][0])
    assert tokenizer.decode(result.token_ids).endswith("quantity=11")


def test_every_boundary_before_closing_think(tokenizer):
    ids = tokenizer.encode("One. Two! Three?\n</think>\nFINAL: quantity=11")
    assert [tokenizer.decode(ids[:n]) for n in sentence_boundaries(tokenizer, ids)] == [
        "One.",
        "One. Two!",
        "One. Two! Three?",
    ]


def test_boundary_alignment_accepts_only_whitespace_overshoot():
    class MergedTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(("One.\n", "Future")[i] for i in ids)

    aligner = TokenAligner(MergedTokenizer(), [0, 1], True)
    assert aligner.tokens_for_prefix("One.") == 1
    assert aligner.tokens_for_prefix("One.\nFu") is None


def test_head_artifact_round_trip(tmp_path):
    head = TerminalityHead(2)
    with torch.no_grad():
        head.feature_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        head.feature_std.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]))
    path = tmp_path / "synthetic_head.json"
    head.save(path, {"classifier": "linear"}, {"train_only": True})
    loaded = TerminalityHead.load(path)
    q, s = torch.tensor([1.0, 7.0]), torch.tensor([3.0, 9.0])
    assert torch.equal(head(q, s), loaded(q, s))
    assert sum(p.numel() for p in loaded.parameters()) == 5


def test_identity_split_and_heldout_isolation():
    cad_ids = [f"synthetic-cad-{i}" for i in range(30) for _ in range(2)]
    labels = np.array([0, 1] * 30, np.float32)
    q = np.repeat(labels[:, None], 2, axis=1)
    s = q + 0.1
    splits = identity_splits(cad_ids)
    identities = [{cad_ids[i] for i in indices} for indices in splits.values()]
    assert all(not identities[i] & identities[j] for i in range(3) for j in range(i + 1, 3))
    config = GCTHConfig(l2_grid=(0.01,), max_iter=30)
    head, metadata = train_linear_head(q, s, labels, cad_ids, config)
    q[splits["heldout"]] = 1e4
    s[splits["heldout"]] = -1e4
    other, _ = train_linear_head(q, s, labels, cad_ids, config)
    assert torch.equal(head.linear.weight, other.linear.weight)
    assert torch.equal(head.feature_mean, other.feature_mean)
    assert metadata["train_only"] and metadata["parameters"] == 5


def test_mock_actor_is_frozen_and_features_are_prefix_only(tokenizer):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"is_encoder_decoder": False})()
            self.language_model = torch.nn.Module()
            self.language_model.norm = torch.nn.Identity()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.seen = []

        def forward(self, input_ids, attention_mask, use_cache):
            self.seen.append(input_ids.clone())
            return self.language_model.norm(input_ids.float().unsqueeze(-1).repeat(1, 1, 2))

    model = Model()
    actor = FrozenActor(model, type("Processor", (), {"tokenizer": tokenizer})())
    inputs = {"input_ids": torch.tensor([[2, 4, 6]])}
    q, s = actor.features(inputs, [0, 1], (8,))
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert q.tolist() == [3.0, 3.0] and s.tolist() == [8.0, 8.0]
    assert model.seen[0].tolist() == [[2, 4, 6, 8]]
    actor.close()
