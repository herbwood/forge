# Caller-supplied data

All included examples are invented. Dataset records, images, trajectories, weights and checkpoints are external inputs. Output files are created exclusively: scripts refuse to overwrite an existing output.

## Training trajectories

One JSON object per line:

```json
{"id":"synthetic-example","cad_id":"synthetic-drawing","split":"train","question":"What is the synthetic diameter?","image_path":"user_data/drawing.png","ontology_id":"synthetic-graph","response_ids":[101,102,103]}
```

The illustrated token IDs are schema placeholders, not an executable trajectory. `response_ids` must be exact sampled response tokens from the supplied actor tokenizer. Pass that tokenizer to the state-building script. `cad_id` groups all trajectories/states from the same CAD identity. Only explicitly marked training rows may construct controller-training states. Optional `origin` records provenance. Optional `state_group` names a source pool: the builder deduplicates `(id, closed state, state_group)`, defaulting to `shared`. Use one shared group for repeated hard-task trajectories and separate group names when retaining distinct source generations, matching the final training-pool construction. A text `reasoning` field can replace token IDs for a schema demonstration, but its resulting states cannot train the head.

## FORGE ontology adapter

The ontology JSON maps `ontology_id` to a payload. See [toy_ontology.json](../examples/toy_ontology.json) for a complete runnable text example. Required fields:

| Field | Meaning |
|---|---|
| `canonical_answer` | Training-time ground truth for verification |
| `answer_type` | Strict ontology terminal type: component, feature or quantity |
| `reasoning_family` | Task family used by the FORGE grammar/evaluation |
| `reward_graph.facts_by_feature` | Feature keys and supplied diameter, BOM, component, quantity facts |
| `reward_graph.paths` | Lists of edges with `src`, `dst`, `relation`; entities have `kind` and `value` |
| `reward_graph.operations` | Comparison/filter descriptors and their prerequisites |

Supported routing relations are `has_diameter`, `owner_bom_item`, `identifies_component`, `has_quantity`, `compare_diameter`, and `compare_quantity`. The frozen graph adapter also understands `filter_quantity_gt`; teacher search skips it by the explicit unsupported-relation policy. Comparison mode is `max`. Unknown or inconsistent ontology schemas fail closed. The graph constructs the required terminal dependency closure from the supplied answer type and graph, preserving the frozen adapter's strict behavior. No task IDs or answers are hard-coded in the generic implementation.

## Generic ontology/verifier contract

`ontology.graph_from_dict` accepts `terminal`, a list of `nodes`, and optional `facts`/`operations`. Each node supplies `id`, `label`, and optional `kind`, `feature`, `slot`, `gold`, `prereqs`. Dependencies must form a DAG; terminal ancestors define required nodes, and depth is computed. A domain-specific checker returns `verified`, `missing` or `contradiction` with evidence. `verifier.verify_graph` applies prerequisite closure. ORSD's full `Verifier` protocol additionally supplies strict prefix/suffix cuts, answer/format checks and contradiction checks; the FORGE adapter implements that protocol as module functions.

## Verified controller states

The state builder emits `id`, `cad_id`, `split`, `question`, `image_path`, `payload`, `prefix_text`, `closed`, `label`, `origin`, and, when aligned actor tokens exist, `prefix_ids`. `closed` contains strict prerequisite-closed node IDs. `label` is 1 exactly when the requested terminal node is closed. It emits closed-set growth through the first terminal state, never later answer text. The head trainer re-decodes and strictly re-verifies each supplied state/label before extracting features.

Training-state files contain verifier ground truth and must never serve as inference input features. Hold out CAD identities for controller selection within the training pool; benchmark validation/test examples are not head-training inputs.

## Runtime requests and targets

`GenerationRequest` carries prompt/context token IDs, multimodal inputs, sampling configuration and current actor revision. It has no ontology or gold payload. Returned `Sample` objects must match request ID, exact prompt/context and revision. `DirectGroup` stores eight direct token sequences, binary answer correctness and an external verifier payload. Targets retain request/child provenance and E=1 or Z=1 evidence. Only their exact sampled suffix tokens receive CE; teacher signal tokens are absent from student contexts.

## Inference and evaluation

Inference JSONL needs only `id`, `question`, `image_path`. One image span is supported by the reference feature extractor. Predictions contain `id`, `raw_response`, `content`, `response_ids`, `stopped`, and `stop_token_count`.

Evaluation gold requires `id`, `canonical_answer`, `answer_type` or a supported `reasoning_family`; `display_answer` is optional. Evaluation supports feature, view, quantity, diameter and component; this is broader than the strict ontology's supported terminal types. Predictions must cover exactly the same unique IDs and provide `content` or `raw_response`. The scorer preserves the official type-aware parser and normalization; diameter absolute tolerance defaults to `1e-6`. Metrics are overall and family accuracy plus parse failures.
