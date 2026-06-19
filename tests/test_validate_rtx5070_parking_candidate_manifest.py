from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "validate_rtx5070_parking_candidate_manifest_for_test",
    SCRIPTS / "validate_rtx5070_parking_candidate_manifest.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def write_text(path: Path, text: str = "x") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def make_manifest(tmp_path: Path) -> dict:
    files = []
    for role in module.REQUIRED_FILE_ROLES:
        path = Path(write_text(tmp_path / "artifacts" / f"{role}.txt"))
        files.append(
            {
                "role": role,
                "path": str(path),
                "remote_path": f"/remote/{role}",
                "required": True,
                "exists": True,
                "bytes": path.stat().st_size,
            }
        )
    return {
        "manifest_version": 1,
        "generated_at": "2026-06-19T00:00:00+00:00",
        "git_head": "abcdef0",
        "status": module.EXPECTED_STATUS,
        "passed": True,
        "dataset": {
            "route": "guarded_v2 + ground artifact guard + strict surface fusion + object surface relabel",
            "candidate_name": module.EXPECTED_CANDIDATE,
        },
        "commands": {
            "remote_rebuild": "scripts/run_rtx5070_parking_candidate_surface_route.sh",
            "local_pull": "scripts/pull_rtx5070_parking_candidate_surface_route.sh",
        },
        "metrics": {
            "viewer": {
                "output_vertices": 1000,
                "object_count_with_points": 12,
                "missing_target_points": 0,
                "label_counts": {"ground": 10, "wall": 20, "car": 30, "railing": 40},
            },
            "qa": {"all_risk_reason_counts": {"wall_too_flat_low_height": 1}},
            "comparison": {
                "baseline_all_candidate_count": 10,
                "candidate_all_candidate_count": 8,
                "candidate_deltas_from_baseline": {
                    "ground_has_large_height_span": 0,
                    "wall_too_flat_low_height": -2,
                    "wall_normal_too_up": -1,
                },
            },
            "geometry_refine": {"missing_target_points": 0},
        },
        "checks": [{"name": "embedded", "passed": True}],
        "files": files,
    }


def test_validate_manifest_passes_for_current_candidate_shape(tmp_path: Path):
    result = module.validate_manifest(make_manifest(tmp_path))

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["summary"]["candidate_all_candidate_count"] == 8


def test_validate_manifest_fails_when_surface_risks_get_worse(tmp_path: Path):
    manifest = make_manifest(tmp_path)
    manifest["metrics"]["comparison"]["candidate_deltas_from_baseline"] = {
        "ground_has_large_height_span": 3,
        "wall_too_flat_low_height": 0,
        "wall_normal_too_up": -1,
    }

    result = module.validate_manifest(manifest)

    assert result["passed"] is False
    assert any("surface_risk_deltas_improve" in error for error in result["errors"])


def test_validate_manifest_fails_when_required_file_missing_on_disk(tmp_path: Path):
    manifest = make_manifest(tmp_path)
    first_file = Path(manifest["files"][0]["path"])
    first_file.unlink()

    result = module.validate_manifest(manifest)

    assert result["passed"] is False
    assert any("required_files_exist_on_disk" in error for error in result["errors"])


def test_validate_manifest_can_skip_disk_check(tmp_path: Path):
    manifest = make_manifest(tmp_path)
    Path(manifest["files"][0]["path"]).unlink()

    result = module.validate_manifest(manifest, check_disk=False)

    assert result["passed"] is True


def test_cli_result_is_json_serializable(tmp_path: Path):
    result = module.validate_manifest(make_manifest(tmp_path))

    json.dumps(result, ensure_ascii=False)
