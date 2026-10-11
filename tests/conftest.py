"""Invented fixtures: no external data, tokenizer, actor, or checkpoint downloads."""

import json
import re
from pathlib import Path

import pytest

from forge_posttraining.adapters import forge
from forge_posttraining.schemas import DirectGroup, Sample

ROOT = Path(__file__).resolve().parents[1]
DIAMETERS = "F01 has diameter 42.25 mm. F02 has diameter 29.75 mm."
COMPARE = " F01 has the larger diameter."
OWNER = " The owner marker associated with F01 corresponds to BOM item 7."
COMPONENT = " BOM item 7 identifies synthetic rotor."
CLOSED = DIAMETERS + COMPARE + OWNER + COMPONENT
CLOSURE = "\n</think>\nFINAL: synthetic rotor"


class ToyTokenizer:
    """Lossless word pieces, including distinct frozen search control tokens."""

    controls = ("</think>", " FINAL", "FINAL", "<|im_end|>", "<|endoftext|>", "<|image_pad|>")

    def __init__(self):
        self.pieces = list(self.controls)

    def encode(self, text, add_special_tokens=False):
        pattern = "|".join(re.escape(x) for x in self.controls) + r"|\s+|[^\s<>]+" + r"|."
        ids = []
        for piece in re.findall(pattern, text):
            if piece not in self.pieces:
                self.pieces.append(piece)
            ids.append(self.pieces.index(piece))
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids)


class ScriptedSampler:
    """Represent sampled children while recording real request/context provenance."""

    def __init__(self, tokenizer, callback):
        self.tokenizer, self.callback, self.requests = tokenizer, callback, []

    def sample(self, requests):
        output = []
        for request in requests:
            self.requests.append(request)
            output.append(
                [
                    Sample(
                        tuple(self.tokenizer.encode(self.callback(request, i))),
                        request.request_id,
                        request.prompt_ids,
                        request.context_ids,
                        request.actor_revision,
                    )
                    for i in range(request.n)
                ]
            )
        return output


@pytest.fixture
def tokenizer():
    return ToyTokenizer()


@pytest.fixture
def payload():
    return json.loads((ROOT / "examples/toy_ontology.json").read_text())["synthetic-comparison"]


@pytest.fixture
def graph(payload):
    return forge.build_graph(payload)


@pytest.fixture
def groups(tokenizer, payload):
    def group(prompt_id, text, answers):
        return DirectGroup(
            prompt_id,
            tuple(tokenizer.encode("synthetic question")),
            tuple(tuple(tokenizer.encode(text)) for _ in range(8)),
            answers,
            payload,
        )

    return [
        group("synthetic-wrong", DIAMETERS, (0,) * 8),
        group("synthetic-retained", CLOSED + CLOSURE, (1,) * 8),
    ]
