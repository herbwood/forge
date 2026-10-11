#!/usr/bin/env python3
"""Actor-only generation or causal GCTH STOP-only replay from the same frozen actor."""

import argparse
from pathlib import Path
import yaml
from forge_posttraining.model_io import FrozenActor
from forge_posttraining.terminality import TerminalityHead, GCTHConfig
from forge_posttraining.inference import replay
from forge_posttraining.utils import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="JSONL: id, question, image_path; no ontology/gold fields needed"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="Dynamic LoRA adapter")
    parser.add_argument("--head", help="Trained GCTH JSON head; omit for actor-only inference")
    parser.add_argument("--config", default="configs/gcth_linear.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Refusing to overwrite existing predictions")
    if args.max_response_tokens <= 0:
        parser.error("Response budget must be positive")
    config = GCTHConfig(**yaml.safe_load(Path(args.config).read_text()))
    actor = FrozenActor.load(args.model, args.adapter, args.device, args.cache_dir)
    head = TerminalityHead.load(args.head).to(args.device) if args.head else None
    if head and head.threshold != config.threshold:
        parser.error("Head and inference thresholds differ")
    output = []
    try:
        for row in read_jsonl(args.input):
            inputs, qpositions = actor.inputs(row["question"], row["image_path"])
            original = actor.generate(inputs, (), args.max_response_tokens)
            if head:

                def score(prefix):
                    q, s = actor.features(inputs, qpositions, prefix)
                    return float(head(q, s))

                result = replay(
                    actor.tokenizer,
                    original,
                    score,
                    lambda prefix, budget: actor.generate(inputs, prefix, budget),
                    args.max_response_tokens,
                    config.threshold,
                    config.sentence_boundary_regex,
                )
                ids = result.token_ids
                stopped = result.stopped
                stop_count = result.stop_token_count
            else:
                ids = original
                stopped = False
                stop_count = None
            raw = actor.tokenizer.decode(list(ids), skip_special_tokens=True)
            content = raw.partition("</think>")[2] if "</think>" in raw else raw
            output.append(
                dict(
                    id=row["id"],
                    raw_response=raw,
                    content=content,
                    response_ids=list(ids),
                    stopped=stopped,
                    stop_token_count=stop_count,
                )
            )
    finally:
        actor.close()
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
