from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "docs" / "current_dense_patch_state.json"
VALIDATOR = ROOT / "scripts" / "validate_current_dense_patch_state.py"


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def test_dense_patch_state_parses() -> None:
    data = load_state()
    assert data["schema"] == "current-dense-patch-state/v1"
    assert data["dataset"] == "MT20260616-175807"
    assert data["authoritative_source"]["type"] == "las"
    assert data["derived_dense_input"]["voxel_size_m"] == 0.03


def test_dense_patch_state_forbids_sparse_and_rejected_inputs() -> None:
    data = load_state()
    forbidden = {item["pattern"] for item in data["forbidden_inputs"]}
    assert "frame_object_points_stride10.ply" in forbidden
    assert "objects_v12_teacher_v20_grid6_unknown_absorb" in forbidden
    assert "objects_v14_teacher_v20_grid6_geometry_guard_wall_recall" in forbidden
    assert "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor" in forbidden
    assert "objects_v16_teacher_v20_grid6_geometry_guard_surface_recall" in forbidden


def test_dense_patch_state_stage_contract_is_geometry_first() -> None:
    data = load_state()
    stages = [item["stage"] for item in data["stage_contract"]]
    assert stages[:3] == ["dense_source", "patch_generation", "patch_boundary_optimization"]
    semantic_stage = next(item for item in data["stage_contract"] if item["stage"] == "semantic_evidence")
    assert "evidence only" in semantic_stage["rule"]
    approved = {item["path"] for item in data["approved_runners"]}
    assert data["next_action"]["runner"] == "scripts/run_dense_patch_object_refinement_v7.py"
    assert data["next_action"]["remote_runner"] == "scripts/run_scan_train_dense_patch_object_refinement_v7.sh"
    assert data["next_action"]["runner"] in approved
    assert data["next_action"]["remote_runner"] in approved
    assert "scripts/run_semantic_evidence_pipeline.py" in approved
    assert "_cpp_region_grower_input.bin" in data["next_action"]["current_blocker"]


def test_dense_patch_state_records_remote_executable_baseline() -> None:
    data = load_state()
    remote = data["remote_executable_baseline"]
    assert remote["host"] == "scan-train"
    assert remote["metrics"]["r4_region_voxel_count"] > 10_000_000
    assert any(path.endswith("_cpp_region_grower_input.bin") for path in remote["remote_paths"])
    assert any(path.endswith("_labels.bin") for path in remote["remote_paths"])


def test_dense_patch_state_records_latest_remote_run() -> None:
    data = load_state()
    latest = data["latest_remote_run"]
    assert latest["id"] == "dense_patch_object_refinement_v8_tiny_attach_20260624_170619"
    assert latest["status"] == "completed"
    assert latest["runner"] == "scripts/run_scan_train_dense_patch_object_refinement_v7.sh"
    assert latest["object_metrics"]["accepted_candidate_rows"] > 1000
    assert latest["object_metrics"]["output_object_count"] > 0
    assert latest["candidate_metrics"]["structural_multimaterial_candidates"] > 1000
    assert latest["qa_metrics"]["mixed_object_voxel_ratio"] < latest["qa_metrics"]["v7_mixed_object_voxel_ratio"]


def test_dense_patch_state_records_current_qa_report() -> None:
    data = load_state()
    qa = data["current_qa_report"]
    assert qa["schema"] == "current-dense-mainline-qa/v1"
    assert qa["json_path"] == "docs/current_dense_mainline_qa.json"
    assert qa["markdown_path"] == "docs/current_dense_mainline_qa.md"
    assert qa["review_index_html"] == "docs/current_dense_review_index.html"
    assert qa["review_index_url"] == "/docs/current_dense_review_index.html"
    assert qa["promotion_gate_json"] == "docs/current_dense_promotion_gate.json"
    assert qa["visual_acceptance_markdown"] == "docs/current_dense_visual_acceptance.md"
    assert qa["promotion_gate_status"] == "awaiting_required_visual_checks"
    assert qa["visual_acceptance_expected_path"] == "docs/current_dense_visual_acceptance.json"
    assert "update_current_dense_visual_acceptance.py" in qa["visual_acceptance_update_command"]
    assert "gate_current_dense_mainline_promotion.py" in qa["visual_acceptance_gate_command"]
    assert qa["key_findings"]["v17_label_point_delta_vs_v9_all_zero"] is True
    assert qa["key_findings"]["v8_mixed_object_voxel_ratio_delta_vs_v7"] < 0


def test_dense_patch_validator_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--state", str(STATE)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["errors"] == []
