# Training workflow

Run commands from the repository root. Each script supports `--help`. Model/data paths below refer to caller-provided inputs; no weights or benchmark examples ship with the release.

```bash
pip install -e '.[vlm]'
mkdir -p outputs
python scripts/train_orsd_vtc.py --config configs/forge_orsd_vtc_qwen3vl4b.yaml --dry-run
```

## Actor training through EasyR1

Install and configure the external EasyR1/verl runtime, provide your SFT-Standard actor and training dataset, and implement the [runtime factory contract](../integrations/easyr1/README.md). The shipped entry point drives exactly one epoch/138 updates through that factory:

```bash
python scripts/train_orsd_vtc.py \
  --config configs/forge_orsd_vtc_qwen3vl4b.yaml --runtime your_bridge:make_runtime
```

The factory receives the full YAML configuration and returns a runtime exposing tokenizer, sampler, strict verifier, current actor revision, rollout batches, update hook and save hook. `your_bridge` is an externally supplied module. The release does not include a distributed trainer, GPU scheduler or checkpoint manager.

For each batch, produce eight direct children per prompt, run the direct reward/answer check, then apply the public controller. Discard all-wrong groups from GRPO. Add only verified signal-free student rows. Ensure the direct advantage calculation, policy ratios, KL and direct denominator exclude SD rows. Compute routing weights over all routing targets of an original prompt before microbatching. Supply the all-reduced direct actor denominator to the SD actor hook. Reject batches without a positive direct denominator; do not replace it with an SD token count. Refresh actor revision after each update and use it for search provenance. Sampling, token masks and multimodal positions are part of the integration contract.

## Constructing train-only states

Collect actor-generated training trajectories and supplied ontologies. Use the same actor tokenizer, not fabricated token IDs:

```bash
python scripts/build_ontology_states.py \
  --qa user_data/train_trajectories.jsonl --ontology user_data/train_ontologies.json \
  --tokenizer user_models/sft_actor --output outputs/verified_train_states.jsonl
```

The builder keeps strict closed-set growth, cuts at verified sentence boundaries, and stops at the first terminal state. It rejects rows outside the training split. Text-only synthetic examples demonstrate the schema but cannot substitute for model-sampled token trajectories.

## Fitting GCTH

```bash
python scripts/train_terminality_head.py \
  --config configs/gcth_linear.yaml --states outputs/verified_train_states.jsonl \
  --model user_models/sft_actor --adapter user_models/final_adapter \
  --device cuda --cache-dir user_models/cache --output outputs/gcth.json
```

The head's actor/model must match inference. All VLM parameters are frozen; only a single linear head is fitted. Each prefix is decoded and reverified before final-norm feature extraction. CAD hash splits separate fit, inner selection and controller-heldout identities within the training pool. Each fitting split and inner selection split needs both terminal and non-terminal labels; heldout identities must be present. A one-drawing toy cannot meet these training requirements.

Fitting balances labels deterministically, standardizes features using fitting-only statistics, selects L2 on inner balanced accuracy, and refits on balanced fit+inner states. The JSON output saves head weights/bias, feature statistics, exact controller config and split metadata. Model/adapter provenance comes from the caller's arguments; review any locally generated artifact before publishing it. No private feature files or trained weights are included here.

## Verification

```bash
pip install -e '.[dev]'
pytest
ruff check src scripts integrations tests
```

These offline tests use only synthetic fixtures and mocked model sampling/features. They verify method invariants and do not rerun the final training experiments.
