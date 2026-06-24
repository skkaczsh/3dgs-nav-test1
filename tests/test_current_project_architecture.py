from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "docs" / "current_project_architecture.json"
VALIDATOR = ROOT / "scripts" / "validate_current_project_architecture.py"


def load_architecture() -> dict:
    return json.loads(ARCHITECTURE.read_text(encoding="utf-8"))


def test_current_project_architecture_parses() -> None:
    data = load_architecture()
    assert data["schema"] == "current-project-architecture/v1"
    assert data["dataset"] == "MT20260616-175807"
    assert data["current_diagnosis"]["decision"]


def test_failed_semantic_branches_are_rejected() -> None:
    data = load_architecture()
    rejected = {item["id"] for item in data["rejected_artifacts"]}
    assert "objects_v12_teacher_v20_grid6_unknown_absorb" in rejected
    assert "objects_v14_teacher_v20_grid6_geometry_guard_wall_recall" in rejected
    assert "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor" in rejected
    assert "objects_v16_teacher_v20_grid6_geometry_guard_surface_recall" in rejected
    assert "v23_mimo_rich_highctx_global_relabel" in rejected
    assert "old_transforms_json_project_world_points_route" in rejected


def test_active_baselines_do_not_use_rejected_artifacts() -> None:
    data = load_architecture()
    active = {item["id"] for item in data["active_baselines"]}
    rejected = {item["id"] for item in data["rejected_artifacts"]}
    assert active.isdisjoint(rejected)
    assert {
        "pure_surface_visibility_full_0000_6180",
        "full_scene_objects_refined_v20",
        "objects_v9_teacher_v20_semantic",
        "objects_v17_teacher_v20_surface_preserve_guard",
    }.issubset(active)


def test_architecture_validator_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--architecture", str(ARCHITECTURE)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["errors"] == []
