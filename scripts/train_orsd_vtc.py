#!/usr/bin/env python3
"""Train via an external EasyR1/verl runtime factory using the public ORSD/VTC controller."""

import argparse
import importlib
import json
from pathlib import Path
import yaml
from forge_posttraining.orsd import ORSDConfig, ORSDController
from forge_posttraining.schemas import SamplingConfig, SafetyError


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config["training"]["epochs"] != 1 or config["training"]["updates"] != 138:
        raise ValueError("The released configuration requires one epoch and 138 updates")
    rollout = config["rollout"]
    closure = config["closure"]
    prompt = config["prompt"]
    core = ORSDConfig(
        sampling=SamplingConfig(
            rollout["temperature"], rollout["top_p"], rollout["top_k"], rollout["n"], rollout["retry_n"]
        ),
        relation_budgets=config["routing"]["relation_budgets"],
        unsupported_relations=tuple(config["routing"]["unsupported_relations"]),
        max_depth_advance=config["routing"]["max_depth_advance"],
        response_length=prompt["train_max_response_tokens"],
        max_prompt_tokens=prompt["max_prompt_tokens"],
        closure_k=closure["K_closure"],
        lambda_sd=config["training"]["lambda_sd"],
    )
    return config, core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Paper-default YAML config")
    parser.add_argument("--runtime", help="External configured trainer factory as importable_module:factory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate the config without loading a model or training"
    )
    args = parser.parse_args()
    config, core = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"config_valid": True, "external_runtime_required": True, "updates": 138}))
        return
    if not args.runtime:
        parser.error(
            "--runtime is required for actual distributed training; this package does not vendor EasyR1"
        )
    module, name = args.runtime.split(":", 1)
    runtime = getattr(importlib.import_module(module), name)(config)
    updates = 0
    for groups in runtime.rollout_batches():
        if updates >= config["training"]["updates"]:
            break
        controller = ORSDController(
            runtime.tokenizer, runtime.sampler, runtime.verifier, runtime.actor_revision, core
        )
        batch = controller.augment(groups)
        runtime.update_direct_and_sd(batch, lambda_sd=core.lambda_sd)
        updates += 1
    if updates != config["training"]["updates"]:
        raise SafetyError("External runtime supplied a different number of updates")
    runtime.save()
    print(json.dumps({"completed_updates": updates}))


if __name__ == "__main__":
    main()
