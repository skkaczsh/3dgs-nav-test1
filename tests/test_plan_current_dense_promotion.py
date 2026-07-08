from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import plan_current_dense_promotion as module
from scripts.plan_current_dense_promotion import build_plan, build_spg_plan


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_current_dense_promotion.py"


def state() -> dict:
    return {
        "current_object_baseline": {
            "id": "dense_las_voxel003_objects_v3_high_recall_clean_20260624",
            "status": "conservative_object_stage_baseline",
        },
        "current_promotion_candidate": {
            "id": "v8_object_refinement",
            "qa_candidate_id": "v8_tiny_attach",
            "source_run_id": "dense_patch_object_refinement_v8_tiny_attach_20260624_170619",
            "status": "awaiting_required_visual_checks",
        },
    }


def qa() -> dict:
    return {
        "object_refinement": {
            "candidate": "v8_tiny_attach",
            "metrics": {
                "v8": {
                    "candidate_count": 6656,
                    "accepted_candidate_rows": 1235,
                    "output_object_count": 196395,
                    "mixed_object_voxel_ratio_020": 0.186092957448242,
                    "object_count_in_overlap_preview": 50831,
                }
            },
        }
    }


def gate(status: str) -> dict:
    return {"candidate": "v8_object_refinement", "status": status}


def test_build_plan_blocks_when_gate_has_not_passed() -> None:
    plan = build_plan(state(), qa(), gate("fail"))

    assert plan["passed"] is False
    assert "promotion_gate_not_passed=fail" in plan["errors"]
    assert plan["proposed_object_baseline"]["id"] == "v8_object_refinement"


def test_build_plan_prepares_geometry_only_object_baseline_after_gate_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    base = (
        tmp_path
        / "server_parking_priority_s10"
        / "geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623"
        / "dense_patch_object_refinement_v8_tiny_attach_20260624_170619"
        / "objects_v7_structural_multimaterial"
    )
    base.mkdir(parents=True)
    (base / "geo_patch_objects_v7_structural_multimaterial_report.json").write_text("{}", encoding="utf-8")
    (base / "geo_patch_objects_v7_structural_multimaterial.jsonl").write_text("", encoding="utf-8")

    plan = build_plan(state(), qa(), gate("pass"))

    assert plan["passed"] is True
    proposed = plan["proposed_object_baseline"]
    assert proposed["id"] == "v8_object_refinement"
    assert proposed["status"] == "promoted_dense_object_geometry_baseline"
    assert proposed["metrics"]["output_object_count"] == 196395
    assert proposed["metrics"]["accepted_candidate_rows"] == 1235
    assert proposed["metrics"]["candidate_count"] == 6656
    assert proposed["metrics"]["input_patch_count"] == 197630
    assert all("stride10" not in path for path in proposed["local_paths"])
    assert any("stride10" in path for path in proposed["qa_only_paths"])
    assert plan["proposed_current_promotion_candidate"]["status"] == "promoted"


def test_cli_reports_current_gate_blocker() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    data = json.loads(result.stdout)
    assert data["schema"] == "current-dense-promotion-plan/v1"
    assert data["candidate"] == "superpoint_graph_v4_nearbbox_s070_e120_20260708_183437"
    assert "spg_visual_acceptance_not_accepted=pending" in data["errors"]


def test_spg_plan_blocks_until_required_visual_checks_are_accepted() -> None:
    spg_state = {
        "current_object_baseline": {"id": "old"},
        "current_promotion_candidate": {
            "id": "spg",
            "qa_candidate_id": "spg",
            "source_run_id": "spg",
            "status": "visual_qa_pending_not_promoted",
        },
    }
    visual = {
        "candidate": "spg",
        "status": "pending",
        "checks": [{"id": "surface", "required": True, "status": "pending"}],
    }

    plan = build_spg_plan(spg_state, visual)

    assert plan["passed"] is False
    assert "spg_visual_acceptance_not_accepted=pending" in plan["errors"]
    assert "spg_visual_required_checks_not_accepted=['surface']" in plan["errors"]
