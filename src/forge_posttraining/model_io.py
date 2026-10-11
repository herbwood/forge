# Image-processing fragment: Copyright 2024 Bytedance Ltd. and/or its affiliates.
# Licensed under Apache-2.0; see licenses/Apache-2.0.txt. Modified to parameterize use.
"""Hugging Face reference actor, prompt processing and frozen final-norm features."""

from __future__ import annotations
import math
from io import BytesIO
import torch
from PIL import Image
from .prompts import task_prompt
from .terminality import pooled_features


def process_image(image, min_pixels: int | None = 262144, max_pixels: int | None = 1048576):
    """EasyR1 image-area processing, preserving its operation order and integer rounding.

    Adapted from EasyR1/verl/utils/dataset.py, Apache-2.0; see third-party notices.
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image.load()
    if max_pixels is not None and image.width * image.height > max_pixels:
        scale = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    if min_pixels is not None and image.width * image.height < min_pixels:
        scale = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class FrozenActor:
    """A frozen causal VLM; the same actor supplies hidden vectors and continuations."""

    def __init__(self, model, processor, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("A causal decoder-only actor is required")
        modules = dict(self.model.named_modules())
        names = [n for n in modules if n.endswith("language_model.norm")]
        if len(names) != 1:
            raise ValueError("Exactly one final language_model.norm module is required")
        self.capture = {}

        def hook(module, inputs, output):
            self.capture["norm"] = output[0] if isinstance(output, tuple) else output

        self.handle = modules[names[0]].register_forward_hook(hook)

    @classmethod
    def load(
        cls, model_name: str, adapter: str | None = None, device: str = "cpu", cache_dir: str | None = None
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.float32, cache_dir=cache_dir
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return cls(model, processor, device)

    def close(self):
        self.handle.remove()

    def inputs(self, question: str, image_path: str) -> tuple[dict, list[int]]:
        problem = question if "<image>" in question else "<image>" + question
        content = []
        for index, part in enumerate(task_prompt(problem).split("<image>")):
            if index:
                content.append({"type": "image"})
            if part:
                content.append({"type": "text", "text": part})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        values = self.processor(
            [process_image(image_path)], [prompt], add_special_tokens=False, return_tensors="pt"
        )
        q = question.replace("<image>", "").strip()
        start = prompt.find(q)
        if start < 0 or prompt.count(q) != 1:
            raise ValueError("Question span must be unique in the rendered prompt")
        enc = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        expanded = values["input_ids"][0].tolist()
        pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        collapsed = [
            token
            for i, token in enumerate(expanded)
            if not (token == pad and i > 0 and expanded[i - 1] == pad)
        ]
        if collapsed != enc["input_ids"]:
            raise ValueError("Expanded image tokens do not match collapsed question tokenization")
        if enc["input_ids"].count(pad) != 1:
            raise ValueError("Reference question pooling supports exactly one image span")
        count = expanded.count(pad)
        pad_position = enc["input_ids"].index(pad)
        positions = [
            i + (count - 1 if i > pad_position else 0)
            for i, (a, b) in enumerate(enc["offset_mapping"])
            if a < start + len(q) and b > start
        ]
        if not positions or any(expanded[p] == pad for p in positions):
            raise ValueError("Invalid question-token positions")
        return dict(values), positions

    def context_inputs(self, inputs: dict, prefix: tuple[int, ...]) -> dict:
        values = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if prefix:
            values["input_ids"] = torch.cat(
                [
                    values["input_ids"],
                    torch.tensor([prefix], device=self.device, dtype=values["input_ids"].dtype),
                ],
                dim=-1,
            )
        values["attention_mask"] = torch.ones_like(values["input_ids"])
        return values

    @torch.no_grad()
    def features(
        self, inputs: dict, positions: list[int], prefix: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not prefix:
            raise ValueError("A state must have at least one generated token")
        self.capture.clear()
        self.model(**self.context_inputs(inputs, prefix), use_cache=False)
        if "norm" not in self.capture:
            raise ValueError("Final norm was not captured")
        hidden = self.capture["norm"][0].float()
        q, s = pooled_features(hidden, positions, len(inputs["input_ids"][0]) + len(prefix) - 1)
        return q.detach(), s.detach()

    @torch.no_grad()
    def generate(self, inputs: dict, prefix: tuple[int, ...], max_new_tokens: int) -> tuple[int, ...]:
        if max_new_tokens <= 0:
            return ()
        values = self.context_inputs(inputs, prefix)
        width = values["input_ids"].shape[-1]
        generated = self.model.generate(
            **values, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
        return tuple(generated[0, width:].tolist())
