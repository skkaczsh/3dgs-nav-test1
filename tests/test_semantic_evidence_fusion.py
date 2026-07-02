from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.geometry_input_contract import geometry_only_semantic_fields
from scripts.semantic_evidence_fusion import FusionParams, apply_decision, choose_label


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fuse_object_semantic_evidence.py"


def geometry_object(**overrides) -> dict:
    row = {
        "object_id": 1,
        "geometry_type": "horizontal",
        **geometry_only_semantic_fields("horizontal"),
    }
    row.update(overrides)
    return row


def test_geometry_only_without_evidence_stays_unknown() -> None:
    decision = choose_label(geometry_object())

    assert decision["semantic_label"] == "unknown"
    assert decision["semantic_status"] == "kept_original_insufficient_evidence"
    assert decision["semantic_evidence_scores"] == {}


def test_scene_prior_alone_does_not_promote_geometry_only_object() -> None:
    row = geometry_object(
        scene_prior={
            "scene_expected_label_weights": {"floor": 10},
        }
    )

    decision = choose_label(row, FusionParams(min_total_weight=1.0))

    assert decision["semantic_label"] == "unknown"
    assert decision["semantic_status"] == "kept_original_scene_only_evidence"
    assert "scene_only_label_not_promoted" in decision["conflict_flags"]
    assert decision["semantic_evidence_source_scores"]["scene"] == {"floor": 3.5}
    assert decision["semantic_evidence_source_scores"]["sam"] == {}
    assert decision["semantic_evidence_source_scores"]["teacher"] == {}


def test_sam_and_teacher_evidence_can_promote_allowed_label() -> None:
    row = geometry_object(
        semantic_votes={"floor": 4, "wall": 1},
        teacher_allowed_votes={"floor": 2},
    )

    decision = choose_label(row)

    assert decision["semantic_label"] == "floor"
    assert decision["semantic_status"] == "evidence_fusion_applied"
    assert decision["semantic_confidence"] > 0.70
    assert decision["semantic_vetoed_scores"]["wall"] > 0
    assert decision["semantic_evidence_source_scores"]["sam"]["floor"] == 4.0
    assert decision["semantic_evidence_source_scores"]["teacher"]["floor"] == 2.5
    assert "geometry_vetoed_some_evidence" in decision["conflict_flags"]


def test_geometry_veto_blocks_car_on_horizontal_surface() -> None:
    row = geometry_object(semantic_votes={"car": 10})

    decision = choose_label(row)

    assert decision["semantic_label"] == "unknown"
    assert decision["semantic_status"] == "kept_original_insufficient_evidence"
    assert decision["semantic_vetoed_scores"]["car"] == 10
    assert "geometry_vetoed_some_evidence" in decision["conflict_flags"]


def test_apply_decision_writes_semantic_id_and_status() -> None:
    row = geometry_object(semantic_votes={"floor": 10})
    out = apply_decision(row, choose_label(row))

    assert out["semantic_label"] == "floor"
    assert out["semantic_id"] == 3
    assert out["semantic_fusion_status"] == "evidence_fusion_applied"
    assert out["semantic_label_original"] == "unknown"
    assert out["semantic_evidence_source_scores"]["sam"] == {"floor": 10.0}


def test_cli_writes_fused_jsonl_and_report(tmp_path: Path) -> None:
    objects = tmp_path / "objects.jsonl"
    output = tmp_path / "fused.jsonl"
    report = tmp_path / "report.json"
    objects.write_text(json.dumps(geometry_object(semantic_votes={"floor": 10})) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--objects-jsonl",
            str(objects),
            "--output-jsonl",
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    data = json.loads(report.read_text(encoding="utf-8"))
    assert row["semantic_label"] == "floor"
    assert row["semantic_evidence_source_scores"]["sam"] == {"floor": 10.0}
    assert data["schema"] == "object-semantic-evidence-fusion/v1"
    assert data["label_counts"] == {"floor": 1}
