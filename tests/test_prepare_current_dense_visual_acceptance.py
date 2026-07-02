from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.prepare_current_dense_visual_acceptance import build_record, format_md


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_current_dense_visual_acceptance.py"


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def qa() -> dict:
    return {
        "object_refinement": {
            "metrics": {
                "delta_v8_minus_v7": {
                    "accepted_candidate_rows": 1139,
                    "output_object_count": -1139,
                    "mixed_object_voxel_ratio_020": -0.0004,
                }
            }
        },
        "surface_guard": {
            "label_point_counts": {"delta_v17_minus_v9": {"floor": 0, "wall": 0}},
        },
    }


def args(tmp_path: Path, qa_path: Path, force: bool = False):
    return argparse.Namespace(
        qa_json=qa_path,
        output=tmp_path / "visual.json",
        output_md=tmp_path / "visual.md",
        review_index_url="http://127.0.0.1:8765/docs/current_dense_review_index.html",
        force=force,
    )


def test_build_record_defaults_to_pending(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa())

    record = build_record(args(tmp_path, qa_path))

    assert record["schema"] == "current-dense-visual-acceptance/v1"
    assert record["status"] == "pending"
    assert record["accepted_candidate"] == "v8_object_refinement"
    assert all(row["status"] == "pending" for row in record["checks"])
    assert all(row.get("artifact_ids") for row in record["checks"])


def test_existing_accepted_checks_make_record_accepted(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa())
    existing = build_record(args(tmp_path, qa_path))
    for row in existing["checks"]:
        row["status"] = "accepted"
    write_json(tmp_path / "visual.json", existing)

    record = build_record(args(tmp_path, qa_path))

    assert record["status"] == "accepted"


def test_format_md_mentions_promotion_block(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa())

    text = format_md(build_record(args(tmp_path, qa_path)))

    assert "Current Dense Visual Acceptance" in text
    assert "current_dense_review_index.html" in text
    assert "v8_object_refinement" in text
    assert "Promotion remains blocked" in text
    assert "update_current_dense_visual_acceptance.py" in text
    assert "gate_current_dense_mainline_promotion.py" in text


def test_cli_writes_pending_record(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa())
    output = tmp_path / "visual.json"
    output_md = tmp_path / "visual.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--qa-json",
            str(qa_path),
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
    assert output_md.read_text(encoding="utf-8").startswith("# Current Dense Visual Acceptance")
