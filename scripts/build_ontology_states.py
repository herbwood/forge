#!/usr/bin/env python3
"""Build verified train-only states from caller-supplied ontology and actor trajectories."""

import argparse
import json
from forge_posttraining.adapters import forge as verifier
from forge_posttraining.utils import TokenAligner, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True, help="User-supplied training trajectory JSONL")
    parser.add_argument("--ontology", required=True, help="Ontology payloads keyed by ontology_id")
    parser.add_argument("--output", required=True, help="New state JSONL; existing files are not overwritten")
    parser.add_argument("--tokenizer", help="Actor tokenizer name/path; required for sampled-token alignment")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ontologies = json.loads(open(args.ontology, encoding="utf-8").read())
    output = []
    seen = set()
    for row in read_jsonl(args.qa):
        if row.get("split") != "train":
            parser.error("State construction accepts only explicitly marked train rows")
        if not row.get("cad_id"):
            parser.error("Each row requires cad_id")
        payload = ontologies[row["ontology_id"]]
        graph = verifier.build_graph(payload)
        if "response_ids" in row:
            if tokenizer is None:
                parser.error("Sampled response_ids require --tokenizer")
            aligner = TokenAligner(tokenizer, row["response_ids"], True)
            text = aligner.text
        else:
            aligner = None
            text = row["reasoning"]
        previous = frozenset()
        for cut in verifier.sentence_cuts(text, 0, verifier.reasoning_end(text)):
            closed = verifier.verify_text(text[:cut], graph).closed
            if closed <= previous or not (closed - previous):
                continue
            tokens = aligner.tokens_for_prefix(text[:cut]) if aligner else None
            if aligner and tokens is None:
                previous = closed
                continue
            prefix_ids = list(row["response_ids"][:tokens]) if aligner else None
            prefix = aligner.decode(tokens) if aligner else text[:cut]
            # Shared pools deduplicate the same QA/closed state across trajectories.
            # Distinct caller-named source pools can retain their separate generations.
            key = (row["id"], closed, row.get("state_group", "shared"))
            if key in seen:
                previous = closed
                if graph.terminal in closed:
                    break
                continue
            seen.add(key)
            current = dict(
                id=row["id"],
                cad_id=row["cad_id"],
                split="train",
                question=row["question"],
                image_path=row["image_path"],
                payload=payload,
                prefix_text=prefix,
                closed=sorted(closed),
                label=int(graph.terminal in closed),
                origin=row.get("origin", "supplied-actor-trajectory"),
                state_group=row.get("state_group", "shared"),
            )
            if prefix_ids is not None:
                current["prefix_ids"] = prefix_ids
            output.append(current)
            previous = closed
            if graph.terminal in closed:
                break
    write_jsonl(args.output, output)
    print(json.dumps({"verified_training_states": len(output)}))


if __name__ == "__main__":
    main()
