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
