#!/usr/bin/env python3
"""Score external predictions using the frozen official answer extraction/normalization."""

import argparse
import json
from collections import defaultdict
from forge_posttraining.evaluation import extract_prediction, score_prediction, answer_type_for_family
from forge_posttraining.utils import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="User-supplied QA JSONL with canonical/display answers")
    parser.add_argument(
        "--predictions", required=True, help="JSONL with unique id and content or raw_response"
    )
    parser.add_argument("--output", required=True, help="New per-example scored JSONL")
    parser.add_argument("--diameter-abs-tol", type=float, default=1e-6)
    args = parser.parse_args()
    gold = read_jsonl(args.gold)
    predictions = read_jsonl(args.predictions)
    expected = {r["id"] for r in gold}
    indexed = {r["id"]: r for r in predictions}
    if len(expected) != len(gold) or len(indexed) != len(predictions) or set(indexed) != expected:
        parser.error("Require complete identical ID sets, no missing or duplicate predictions")
    output = []
    families = defaultdict(list)
    for row in gold:
        prediction = indexed[row["id"]]
        text = prediction.get("content", prediction.get("raw_response", ""))
        typ = row.get("answer_type") or answer_type_for_family(row["reasoning_family"])
        value, method = extract_prediction(text, typ)
        correct, normalized, gold_normalized = score_prediction(value, row, typ, args.diameter_abs_tol)
        result = dict(
            id=row["id"],
            prediction=value,
            parse_method=method,
            correct=correct,
            normalized_prediction=normalized,
            normalized_gold=gold_normalized,
        )
        output.append(result)
        families[row.get("reasoning_family", "unspecified")].append(result)
    write_jsonl(args.output, output)
    metrics = lambda rows: dict(
        rows=len(rows),
        accuracy=sum(r["correct"] for r in rows) / len(rows) if rows else None,
        parse_failures=sum(r["prediction"] is None for r in rows),
    )
    print(
        json.dumps(
            {"overall": metrics(output), "by_family": {k: metrics(v) for k, v in families.items()}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
