from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_object_semantic_evidence_fusion import validate


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_object_semantic_evidence_fusion.py"


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def input_row() -> dict:
    return {
        "object_id": 1,
        "geometry_type": "horizontal",
        "bbox_3d": {"min": [0, 0, 0], "max": [1, 1, 0.1]},
        "voxel_count": 100,
        "semantic_votes": {"floor": 10},
    }


def output_row(**overrides) -> dict:
    row = {
        **input_row(),
        "semantic_label": "floor",
        "semantic_id": 3,
        "semantic_fusion_status": "evidence_fusion_applied",
        "semantic_fusion_confidence": 1.0,
        "semantic_evidence_scores": {"floor": 10},
        "semantic_vetoed_scores": {},
    }
    row.update(overrides)
    return row


def test_validator_accepts_valid_fusion_output(tmp_path: Path) -> None:
    before = write_jsonl(tmp_path / "before.jsonl", [input_row()])
    after = write_jsonl(tmp_path / "after.jsonl", [output_row()])
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"schema": "object-semantic-evidence-fusion/v1", "object_count": 1}), encoding="utf-8")

    result = validate(before, after, report)

    assert result["passed"] is True
    assert result["errors"] == []


def test_validator_rejects_object_ownership_change(tmp_path: Path) -> None:
    before = write_jsonl(tmp_path / "before.jsonl", [input_row()])
    after = write_jsonl(tmp_path / "after.jsonl", [output_row(geometry_type="vertical")])

    result = validate(before, after)

    assert result["passed"] is False
    assert any("ownership_field_changed=geometry_type" in err for err in result["errors"])


def test_validator_rejects_scene_only_promotion(tmp_path: Path) -> None:
    src = input_row()
    src.pop("semantic_votes")
    src["scene_prior"] = {"scene_expected_label_weights": {"floor": 10}}
    out = output_row(semantic_votes={}, scene_prior=src["scene_prior"], semantic_evidence_scores={"floor": 3.5})
    before = write_jsonl(tmp_path / "before.jsonl", [src])
    after = write_jsonl(tmp_path / "after.jsonl", [out])

    result = validate(before, after)

    assert result["passed"] is False
    assert "object=1:scene_only_promotion" in result["errors"]


def test_cli_writes_validation_report(tmp_path: Path) -> None:
    before = write_jsonl(tmp_path / "before.jsonl", [input_row()])
    after = write_jsonl(tmp_path / "after.jsonl", [output_row()])
    out = tmp_path / "validation.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-objects",
            str(before),
            "--output-objects",
            str(after),
            "--output-json",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True
