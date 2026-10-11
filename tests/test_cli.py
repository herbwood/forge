"""Offline CLI demonstrations; model-dependent commands are checked through help."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def run(script, *args):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    return subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_help_and_missing_inputs(script):
    help_result = run(script, "--help")
    assert help_result.returncode == 0 and "usage:" in help_result.stdout
    missing = run(script)
    assert missing.returncode == 2 and "required" in missing.stderr


def test_dry_run_validates_paper_config():
    result = run(
        ROOT / "scripts/train_orsd_vtc.py", "--config", "configs/forge_orsd_vtc_qwen3vl4b.yaml", "--dry-run"
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "config_valid": True,
        "external_runtime_required": True,
        "updates": 138,
    }


def test_synthetic_state_build_and_no_overwrite(tmp_path):
    output = tmp_path / "states.jsonl"
    args = ("--qa", "examples/toy_qa.jsonl", "--ontology", "examples/toy_ontology.json", "--output", output)
    result = run(ROOT / "scripts/build_ontology_states.py", *args)
    assert result.returncode == 0
    states = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["label"] for row in states] == [0, 0, 0, 0, 1]
    assert all(row["split"] == "train" and "prefix_ids" not in row for row in states)
    original = output.read_bytes()
    assert run(ROOT / "scripts/build_ontology_states.py", *args).returncode != 0
    assert output.read_bytes() == original


def test_synthetic_official_evaluation(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"id": "toy-qa-alpha", "content": "FINAL: synthetic rotor"}) + "\n"
    )
    result = run(
        ROOT / "scripts/evaluate_predictions.py",
        "--gold",
        "examples/toy_qa.jsonl",
        "--predictions",
        predictions,
        "--output",
        tmp_path / "scored.jsonl",
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["overall"] == {"rows": 1, "accuracy": 1.0, "parse_failures": 0}


def test_controller_states_deduplicate_shared_pool(tmp_path):
    row = json.loads((ROOT / "examples/toy_qa.jsonl").read_text())
    other = dict(row, state_group="synthetic-distinct-source")
    qa = tmp_path / "trajectories.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in (row, row, other)) + "\n")
    output = tmp_path / "states.jsonl"
    result = run(
        ROOT / "scripts/build_ontology_states.py",
        "--qa",
        qa,
        "--ontology",
        "examples/toy_ontology.json",
        "--output",
        output,
    )
    assert result.returncode == 0
    states = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(states) == 10
    assert sum(r["label"] for r in states) == 2


def test_inference_imports_no_ontology():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import forge_posttraining.inference, forge_posttraining.model_io, sys; "
            "assert not any(x in sys.modules for x in ['forge_posttraining.ontology', "
            "'forge_posttraining.verifier', 'forge_posttraining.adapters.forge'])",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
