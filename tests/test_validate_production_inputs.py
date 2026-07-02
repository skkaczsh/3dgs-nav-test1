import json
import subprocess
import sys
from pathlib import Path

from scripts import validate_production_inputs as module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_production_inputs.py"


def test_validate_paths_rejects_viewer_stride_ply() -> None:
    report = module.validate_paths(["/tmp/frame_object_points_stride10.ply"])

    assert report["passed"] is False
    assert report["errors"] == [
        "forbidden_production_input=frame_object_points_stride10.ply:/tmp/frame_object_points_stride10.ply"
    ]


def test_validate_paths_allows_dense_voxel_ply() -> None:
    report = module.validate_paths(["/tmp/dense_las_voxel003_binary.ply"])

    assert report["passed"] is True
    assert report["errors"] == []


def test_validate_paths_requires_current_dense_allowlist() -> None:
    allowed = {"/tmp/dense_las_voxel003_binary.ply"}

    report = module.validate_paths(["/tmp/dense_las_voxel003_binary.ply"], allowed_paths=allowed)

    assert report["passed"] is True
    assert report["checked"][0]["allowlist_match"] is True


def test_validate_paths_rejects_not_current_dense_input() -> None:
    report = module.validate_paths(["/tmp/random_dense_like.ply"], allowed_paths={"/tmp/other.ply"})

    assert report["passed"] is False
    assert report["errors"] == ["not_current_dense_input:/tmp/random_dense_like.ply"]


def test_cli_json_report() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "/tmp/dense_las_voxel003_binary.ply"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True


def test_cli_requires_current_dense_path_from_state() -> None:
    state = ROOT / "docs" / "current_dense_patch_state.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json",
            "--state",
            str(state),
            "--require-current-dense",
            "/root/epfs/SCAN/work_MT20260616-175807/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/_cpp_region_grower_input.bin",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["passed"] is True


def test_cli_rejects_forbidden_path() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "/tmp/objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor/out.ply"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "forbidden_production_input=objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor" in result.stderr
