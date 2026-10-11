# Invented examples

`toy_qa.jsonl` and `toy_ontology.json` describe an invented two-feature comparison. All names, quantities, dimensions and IDs were created for this release. They are not drawn from benchmark records.

From the repository root:

```bash
mkdir -p outputs
python scripts/build_ontology_states.py \
  --qa examples/toy_qa.jsonl --ontology examples/toy_ontology.json \
  --output outputs/toy_states.jsonl
```

This yields five strict closed-set growth states, the last terminal. The example uses text reasoning for schema illustration; no sampled token IDs or image are included. `image_path` indicates a caller-provided image. Actual head training requires sampled actor tokens, matching images/weights, and enough training CAD identities with both labels for the disjoint controller splits. Tests use their own small lossless tokenizer and scripted sampler to exercise model-output provenance without downloads.
