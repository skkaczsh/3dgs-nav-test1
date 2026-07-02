from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.plan_current_dense_promotion import build_plan


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


def test_build_plan_prepares_geometry_only_object_baseline_after_gate_pass() -> None:
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
    assert data["candidate"] == "v8_object_refinement"
    assert "promotion_gate_not_passed=fail" in data["errors"]
