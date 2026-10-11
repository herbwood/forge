# External EasyR1/verl integration

This directory provides the minimal contracts and hooks required to connect the released method to an external EasyR1/verl runtime. The distributed framework itself is not vendored here.

## Runtime factory

Provide `your_bridge:make_runtime(config)` in your external environment. It returns the `TrainingRuntime` protocol in `controller.py`:

| Member | Required behavior |
|---|---|
| `tokenizer` | Exact actor tokenizer, including single-token search controls |
| `sampler` | Public `Sampler`; current actor, batched children, exact provenance |
| `verifier` | Strict `Verifier` implementation; the FORGE adapter module is available |
| `actor_revision` | Current actor/adapter identity, refreshed after updates |
| `rollout_batches()` | Eight-child direct groups with binary answer correctness and external verification payloads |
| `update_direct_and_sd(batch, lambda_sd)` | Build position-preserving SD rows; apply retained direct GRPO/KL and independently normalized SD |
| `save()` | Save caller-selected external actor/adapter outputs |

The entry point consumes 138 updates and calls `save` only after that count. Configure actor/data/hardware/checkpoint destinations in the external runtime. `example_config.yaml` declares hook expectations; `runtime_factory: null` must be supplied externally. The released method YAML carries the model, LoRA, and search defaults.

## Sampler adapter

`EasyR1Sampler` accepts three callbacks: `build_requests(requests, maximum_budget)`, the external generation worker's `generate_sequences`, and `decode_children(generated, requests)`. The builder must attach multimodal data, exact prompt+context tokens, per-request logit bias, n and T/top-p/top-k parameters. Requests share n and generate under the largest requested budget, matching the source generation window; the controller then crops each child to its own relation limit. The builder handles framework padding/data-parallel layout. Decoder callbacks discard padding and return `Sample` lists in original request/child order with attributable current revision.

For retry n=16, children 1–8 must retain their n=8 indexing/seeds; the controller trains only newly considered children 9–16. Never attach gold answers, ontology values or verifier payloads to model generation inputs. Teacher contexts contain only the operation-name signal. Closure contexts have no signal and no search mask override. Teacher generation requests/children are not PPO/GRPO rows.

## Trainer hooks

- `reward_hook` applies the active strict FORGE grammar and exact direct reward.
- `student_rows` left-pads prompts, moves the accepted model prefix into the prompt, preserves original multimodal rotary positions, masks CE to sampled suffix tokens, and sets prompt weight, scores, advantages and returns to zero.
- `direct_advantages_only` delegates advantage calculation only for retained rows marked `normal`.
- `actor_objective` adds the SD numerator with DDP world-size scaling and shared global direct actor denominator to your already normalized direct objective.

The external actor must exclude SD rows from policy ratios, direct loss, KL and direct denominator, not merely zero their advantages. Preserve original per-prompt routing weights across minibatches/microbatches. `original_rows` maps original prompt IDs to prompt tensors, prompt mask, full original prompt+response position IDs and multimodal data. Give enough original response positions to cover the accepted prefix; preserve every multimodal rotary axis. Check the actor's actual denominator/reduction convention against this contract.

## Failure handling

Wrong provenance, unsupported tokenizer controls, token-mask violations, unverified targets, unknown relations, oversized prompts and absent direct normalization must fail closed. Explicitly unsupported routing relations can be skipped. Do not train fabricated targets, broaden the verifier, introduce teacher policy advantages or silently permit SD-only updates. End-to-end distributed training with your bridge remains a caller validation step.
