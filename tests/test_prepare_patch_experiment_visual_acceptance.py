from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.prepare_patch_experiment_visual_acceptance import build_record, format_md


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_patch_experiment_visual_acceptance.py"


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def comparison() -> dict:
    return {
        "runs": [
            {"name": "v2", "patches": 188536, "high_entropy": 8410},
            {"name": "v5", "patches": 189898, "high_entropy": 8615},
        ]
    }


def args(tmp_path: Path, comparison_path: Path, force: bool = False):
    return argparse.Namespace(
        comparison_json=comparison_path,
        output=tmp_path / "visual.json",
        output_md=tmp_path / "visual.md",
        review_index_url="http://127.0.0.1:8765/docs/patch_experiment_review_index.html",
        selected_candidate="v2_bucket_attach",
        force=force,
    )


def test_build_record_defaults_to_pending(tmp_path: Path) -> None:
    comparison_path = write_json(tmp_path / "comparison.json", comparison())

    record = build_record(args(tmp_path, comparison_path))

    assert record["schema"] == "patch-experiment-visual-acceptance/v1"
    assert record["status"] == "pending"
    assert record["selected_candidate"] == "v2_bucket_attach"
    assert record["candidate_policy"] == "geometry_input_only"
    assert all(row["status"] == "pending" for row in record["checks"])


def test_existing_accepted_checks_make_record_accepted(tmp_path: Path) -> None:
    comparison_path = write_json(tmp_path / "comparison.json", comparison())
    existing = build_record(args(tmp_path, comparison_path))
    for row in existing["checks"]:
        row["status"] = "accepted"
    write_json(tmp_path / "visual.json", existing)

    record = build_record(args(tmp_path, comparison_path))

    assert record["status"] == "accepted"


def test_format_md_mentions_gate(tmp_path: Path) -> None:
    comparison_path = write_json(tmp_path / "comparison.json", comparison())

    text = format_md(build_record(args(tmp_path, comparison_path)))

    assert "Patch Experiment Visual Acceptance" in text
    assert "patch_experiment_review_index.html" in text
    assert "gate_patch_experiment_promotion.py" in text


def test_cli_writes_pending_record(tmp_path: Path) -> None:
    comparison_path = write_json(tmp_path / "comparison.json", comparison())
    output = tmp_path / "visual.json"
    output_md = tmp_path / "visual.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--comparison-json",
            str(comparison_path),
            "--output",
            str(output),
            "--output-md",
            str(output_md),
            "--force",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pending"
    assert output_md.read_text(encoding="utf-8").startswith("# Patch Experiment Visual Acceptance")
