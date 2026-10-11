#!/usr/bin/env python3
"""Freeze a supplied actor, verify training states, extract q/s and fit the linear GCTH head."""

import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import yaml
from forge_posttraining.adapters import forge as verifier
from forge_posttraining.model_io import FrozenActor
from forge_posttraining.terminality import GCTHConfig, train_linear_head
from forge_posttraining.utils import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--states", required=True, help="Verified train-only states with exact sampled prefix_ids"
    )
    parser.add_argument("--model", required=True, help="Frozen actor base or SFT model")
    parser.add_argument("--adapter", help="Dynamic LoRA adapter for the same actor")
    parser.add_argument("--output", required=True, help="New JSON head artifact")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", help="Caller-selected model cache")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Refusing to overwrite an existing head")
    config = GCTHConfig(**yaml.safe_load(Path(args.config).read_text()))
    rows = read_jsonl(args.states)
    if not rows:
        parser.error("No training states supplied")
    for row in rows:
        if row.get("split") != "train":
            parser.error("Controller training accepts only train states; no benchmark validation/test rows")
        if not row.get("prefix_ids") or not row.get("cad_id"):
            parser.error("Each state needs exact sampled prefix_ids and cad_id")
    actor = FrozenActor.load(args.model, args.adapter, args.device, args.cache_dir)
    questions = []
    states = []
    labels = []
    cad = []
    try:
        for row in rows:
            graph = verifier.build_graph(row["payload"])
            text = actor.tokenizer.decode(row["prefix_ids"], skip_special_tokens=True)
            checked = verifier.verify_text(text, graph)
            label = int(graph.terminal in checked.closed)
            if set(row["closed"]) != set(checked.closed) or row["label"] != label:
                raise ValueError("Supplied state/label did not strictly reverify")
            inputs, qpositions = actor.inputs(row["question"], row["image_path"])
            q, s = actor.features(inputs, qpositions, tuple(row["prefix_ids"]))
            questions.append(q.cpu().numpy())
            states.append(s.cpu().numpy())
            labels.append(label)
            cad.append(row["cad_id"])
    finally:
        actor.close()
    head, metadata = train_linear_head(
        np.stack(questions), np.stack(states), np.array(labels, np.float32), cad, config
    )
    metadata["actor_model"] = args.model
    metadata["actor_adapter"] = args.adapter
    head.save(args.output, asdict(config), metadata)
    print(f"Saved linear head with {metadata['parameters']} parameters")


if __name__ == "__main__":
    main()
