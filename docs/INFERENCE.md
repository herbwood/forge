# Actor and GCTH inference

Actor-only inference uses the supplied actor's greedy output unchanged. GCTH adds a STOP decision at every completed reasoning sentence boundary; it does not generate answers. Neither inference mode consumes an ontology or verifier.

```bash
python scripts/run_gcth_inference.py \
  --input user_data/questions.jsonl --model user_models/sft_actor \
  --adapter user_models/final_adapter --device cuda \
  --cache-dir user_models/cache --output outputs/actor_predictions.jsonl

python scripts/run_gcth_inference.py \
  --input user_data/questions.jsonl --model user_models/sft_actor \
  --adapter user_models/final_adapter --head outputs/gcth.json \
  --config configs/gcth_linear.yaml --device cuda \
  --cache-dir user_models/cache --output outputs/gcth_predictions.jsonl
```

Create `outputs` first. Both the head and actor are externally supplied. The default shared response limit is 4096; `--max-response-tokens` changes it. `--head` activates the controller; omitting it selects actor-only output.

## Causal offline-equivalent replay

```text
original = same_actor.generate_greedily(prompt, total_budget)
for boundary in completed_sentence_boundaries(original):
    prefix = original[:boundary]
    q, s = frozen_same_actor.final_norm_features(prompt, prefix)
    probability = head(q, s)
    if probability >= 0.5:
        prefix = prefix + [single_closing_think_token]
        remaining = total_budget - len(prefix)
        return prefix + same_actor.generate_greedily(prompt, prefix, remaining)
return original
```

Boundary discovery uses `(?<=[.!?])\s+|\n+`, stops before an existing closing-think/FINAL marker, and requires exact token-prefix alignment with only whitespace overshoot allowed. Duplicate token boundaries are skipped. The scoring callback receives only the causal prefix; it never receives future original tokens. The reference actor forwards prompt+prefix with caching disabled for feature extraction, using the final `language_model.norm` output. The causal decoder prevents future influence on q and s.

On CONTINUE, output tokens remain untouched. On the first STOP, all later original tokens are discarded and exactly one `</think>` token is forced. The same frozen actor resumes greedily under the remaining budget. No FINAL, answer, answer type, relation or EOS is forced. A naturally generated answer suffix can contain those actor-generated tokens.

The reference implementation repeats prefix forwards and therefore prioritizes a clear causal contract over throughput. It uses FP32 Hugging Face execution with a dynamic PEFT adapter. Matching the paper's external generation engine requires checking prompt/token, image preprocessing, actor/adapter and numerical parity; release CPU tests do not establish GPU generation parity.
