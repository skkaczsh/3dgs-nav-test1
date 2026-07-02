from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "show_current_mainline.py"


def test_show_current_mainline_json_output() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["dataset"] == "MT20260616-175807"
    assert data["dense_patch_baseline"]["id"] == "dense_las_voxel003_energy_v6_fine_gated_overlap_20260624"
    assert data["dense_object_baseline"]["id"] == "dense_las_voxel003_objects_v3_high_recall_clean_20260624"
    assert data["current_promotion_candidate"]["id"] == "v8_object_refinement"
    assert data["current_promotion_candidate"]["qa_candidate_id"] == "v8_tiny_attach"
    assert data["current_qa_report"]["promotion_gate_status"] == "awaiting_required_visual_checks"
    assert data["current_qa_report"]["review_index_url"] == "/docs/current_dense_review_index.html"
    assert data["current_qa_report"]["review_artifact_allowlist"]["passed"] is True
    assert data["production_input_allowlist"]["passed"] is True
    assert data["production_input_allowlist"]["allowed_count"] == 6
    assert data["state_consistency"]["passed"] is True
    assert data["state_consistency"]["dataset"] == "MT20260616-175807"
    assert data["current_qa_report"]["rejected_guard_diagnostics"]["variants"][0]["unknown_delta_vs_v9"] > 0
    assert any(item["path"] == "scripts/run_dense_patch_object_refinement_v7.py" for item in data["approved_runners"])
    assert any(item["path"] == "scripts/run_semantic_evidence_pipeline.py" for item in data["approved_runners"])
    assert any(item["pattern"] == "frame_object_points_stride10.ply" for item in data["forbidden_inputs"])


def test_show_current_mainline_text_output() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "current dense patch baseline:" in result.stdout
    assert "dense_las_voxel003_energy_v6_fine_gated_overlap_20260624" in result.stdout
    assert "remote executable baseline:" in result.stdout
    assert "latest remote run:" in result.stdout
    assert "dense_patch_object_refinement_v9_mainline_fixdeps_20260702_2108" in result.stdout
    assert "promotion_status: diagnostic_not_promoted" in result.stdout
    assert "Keep v8 as the current visual-promotion candidate" in result.stdout
    assert "current promotion candidate:" in result.stdout
    assert "v8_object_refinement [awaiting_required_visual_checks]" in result.stdout
    assert "qa_candidate_id: v8_tiny_attach" in result.stdout
    assert "runner: scripts/run_dense_patch_object_refinement_v7.py" in result.stdout
    assert "remote_runner: scripts/run_scan_train_dense_patch_object_refinement_v7.sh" in result.stdout
    assert "approved runners:" in result.stdout
    assert "scripts/run_semantic_evidence_pipeline.py [semantic_evidence]" in result.stdout
    assert "blocker:" in result.stdout
    assert "current QA / promotion gate:" in result.stdout
    assert "promotion_gate_status: awaiting_required_visual_checks" in result.stdout
    assert "update_command: python3 scripts/update_current_dense_visual_acceptance.py" in result.stdout
    assert "gate_command: python3 scripts/gate_current_dense_mainline_promotion.py" in result.stdout
    assert "promotion_plan_command: python3 scripts/plan_current_dense_promotion.py" in result.stdout
    assert "review_allowlist: passed=True" in result.stdout
    assert "production input allowlist:" in result.stdout
    assert "allowed_count=6" in result.stdout
    assert "state consistency:" in result.stdout
    assert "dataset=MT20260616-175807" in result.stdout
    assert "rejected_guard: objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor" in result.stdout
    assert "unknown_delta_vs_v9=1006072" in result.stdout
    assert "forbidden inputs:" in result.stdout
    assert "frame_object_points_stride10.ply" in result.stdout


def test_show_current_mainline_resolves_default_paths_outside_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json"],
        cwd=ROOT.parent,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["dataset"] == "MT20260616-175807"


def test_show_current_mainline_supports_json_alias() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["production_input_allowlist"]["allowed_count"] == 6
    assert data["state_consistency"]["passed"] is True
