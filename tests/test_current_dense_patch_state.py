from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_current_dense_patch_state import validate
from scripts.current_mainline_contract import REQUIRED_OPERATOR_TOOL_PATHS


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


def test_dense_patch_state_records_operator_tools() -> None:
    data = load_state()
    tools = {item["path"] for item in data["operator_tools"]}
    assert set(REQUIRED_OPERATOR_TOOL_PATHS).issubset(tools)
    assert "scripts/plan_current_dense_promotion.py" in tools
    assert "scripts/validate_current_mainline.py" in tools


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
    assert latest["id"] == "dense_patch_object_refinement_v9_mainline_fixdeps_20260702_2108"
    assert latest["status"] == "completed"
    assert latest["promotion_status"] == "diagnostic_not_promoted"
    assert latest["runner"] == "scripts/run_scan_train_dense_patch_object_refinement_v7.sh"
    assert latest["object_metrics"]["accepted_candidate_rows"] == 96
    assert latest["object_metrics"]["output_object_count"] > 0
    assert latest["candidate_metrics"]["structural_multimaterial_candidates"] == 80
    assert latest["candidate_metrics"]["reject_counts"]["small_patch"] == 7522
    assert latest["object_metrics"]["rejection_counts"]["score"] == 117
    assert "geometry_input_contract.py" in latest["failure_repaired"]
    assert "Keep v8 as the current visual-promotion candidate" in latest["interpretation"]


def test_dense_patch_state_records_current_promotion_candidate() -> None:
    data = load_state()
    candidate = data["current_promotion_candidate"]
    assert candidate["id"] == "v8_object_refinement"
    assert candidate["qa_candidate_id"] == "v8_tiny_attach"
    assert candidate["status"] == "awaiting_required_visual_checks"
    assert candidate["gate_json"] == "docs/current_dense_promotion_gate.json"
    assert candidate["visual_acceptance_json"] == "docs/current_dense_visual_acceptance.json"
    assert candidate["qa_json"] == "docs/current_dense_mainline_qa.json"

    gate = json.loads((ROOT / candidate["gate_json"]).read_text(encoding="utf-8"))
    visual = json.loads((ROOT / candidate["visual_acceptance_json"]).read_text(encoding="utf-8"))
    qa = json.loads((ROOT / candidate["qa_json"]).read_text(encoding="utf-8"))
    assert gate["candidate"] == candidate["id"]
    assert visual["accepted_candidate"] == candidate["id"]
    assert qa["object_refinement"]["candidate"] == candidate["qa_candidate_id"]


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


def test_dense_patch_validator_rejects_weak_latest_run_without_diagnostic_status(tmp_path: Path) -> None:
    state = load_state()
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps(
            {
                "object_refinement": {
                    "metrics": {
                        "v8": {
                            "accepted_candidate_rows": 100,
                            "output_object_count": 1000,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    for key in (
        "markdown_path",
        "review_index_html",
        "promotion_gate_json",
        "visual_acceptance_markdown",
    ):
        placeholder = tmp_path / f"{key}.txt"
        placeholder.write_text("x", encoding="utf-8")
        state["current_qa_report"][key] = str(placeholder)
    state["current_qa_report"]["json_path"] = str(qa)
    state["latest_remote_run"]["promotion_status"] = "candidate"
    state["latest_remote_run"]["object_metrics"]["accepted_candidate_rows"] = 1
    state["latest_remote_run"]["object_metrics"]["output_object_count"] = 2000
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    report = validate(path)

    assert report["passed"] is False
    assert "latest_weaker_than_v8_but_not_diagnostic" in report["errors"]


def test_dense_patch_validator_rejects_promotion_candidate_mismatch(tmp_path: Path) -> None:
    state = load_state()
    qa = tmp_path / "qa.json"
    gate = tmp_path / "gate.json"
    visual = tmp_path / "visual.json"
    qa.write_text(
        json.dumps({"object_refinement": {"candidate": "v8_tiny_attach", "metrics": {"v8": {}}}}),
        encoding="utf-8",
    )
    gate.write_text(json.dumps({"candidate": "wrong_candidate", "status": "fail"}), encoding="utf-8")
    visual.write_text(
        json.dumps({"accepted_candidate": "v8_object_refinement", "status": "pending"}),
        encoding="utf-8",
    )
    state["current_promotion_candidate"]["qa_json"] = str(qa)
    state["current_promotion_candidate"]["gate_json"] = str(gate)
    state["current_promotion_candidate"]["visual_acceptance_json"] = str(visual)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    report = validate(path)

    assert report["passed"] is False
    assert "promotion_candidate_gate_mismatch=wrong_candidate!=v8_object_refinement" in report["errors"]


def test_dense_patch_validator_rejects_wrong_authoritative_source_identity(tmp_path: Path) -> None:
    state = load_state()
    state["authoritative_source"]["id"] = "raw_lx_sections"
    state["authoritative_source"]["local_paths"] = ["/tmp/MANIFOLD_MT20260616-175807.lx"]
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    report = validate(path)

    assert report["passed"] is False
    assert "unexpected_authoritative_source_id=raw_lx_sections" in report["errors"]
    assert "authoritative_source_missing_opt_las_path" in report["errors"]


def test_dense_patch_validator_rejects_wrong_dense_voxel_identity(tmp_path: Path) -> None:
    state = load_state()
    state["derived_dense_input"]["id"] = "dense_las_voxel010_binary"
    state["derived_dense_input"]["voxel_size_m"] = 0.1
    state["derived_dense_input"]["known_voxel_count"] = 1_440_000
    state["remote_executable_baseline"]["metrics"]["voxel003_count"] = 1_440_000
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    report = validate(path)

    assert report["passed"] is False
    assert "unexpected_derived_dense_input_id=dense_las_voxel010_binary" in report["errors"]
    assert "derived_dense_input_not_voxel003" in report["errors"]
    assert "derived_dense_input_voxel_count_mismatch=1440000" in report["errors"]
    assert "remote_baseline_voxel003_count_mismatch=1440000" in report["errors"]
