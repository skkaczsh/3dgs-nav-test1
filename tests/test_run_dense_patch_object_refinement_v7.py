from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_dense_patch_object_refinement_v7.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_dense_patch_object_refinement_v7", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_args(tmp_path: Path) -> argparse.Namespace:
    region = tmp_path / "_cpp_region_grower_input.bin"
    labels = tmp_path / "geo_patches_energy_v6_fine_gated_overlap_labels.bin"
    region.write_bytes(b"region")
    labels.write_bytes(b"labels")
    return argparse.Namespace(
        state=ROOT / "docs" / "current_dense_patch_state.json",
        region_input=region,
        patch_labels=labels,
        output_dir=tmp_path / "v7",
        python="python",
        run=False,
        plan_json=None,
        mainline_healthcheck=ROOT / "scripts" / "validate_current_mainline.py",
        no_require_current_dense_inputs=False,
        skip_mainline_healthcheck=False,
        edge_source="region",
        grid_voxel_size=0.03,
        min_patch_voxels=40,
        min_shared_edges=3,
        min_contact_ratio=0.006,
        max_bbox_gap=0.20,
        max_color_distance=105.0,
        min_normal_score=0.42,
        min_bucket_score=0.42,
        min_score=0.54,
        contact_ratio_norm=0.18,
        max_candidates=50000,
        min_structural_score=0.70,
        structural_min_contact_ratio=0.025,
        structural_min_shared_edges=12,
        structural_min_normal_score=0.56,
        structural_max_bbox_gap=0.10,
        preview_stride=10,
        accept_min_score=0.80,
        accept_min_contact_ratio=0.08,
        accept_min_shared_edges=32,
        accept_max_color_distance=55.0,
        accept_max_bbox_gap=0.08,
        accept_min_normal_score=0.65,
        accept_min_structural_score=0.74,
        accept_structural_min_contact_ratio=0.035,
        accept_structural_min_shared_edges=24,
        accept_structural_min_normal_score=0.58,
        accept_structural_max_bbox_gap=0.08,
        attachment_min_score=0.82,
        attachment_min_contact_ratio=0.16,
        attachment_min_shared_edges=48,
        attachment_max_color_distance=38.0,
        attachment_min_normal_score=0.65,
        attachment_max_bbox_gap=0.06,
        attachment_max_fragment_voxels=1200,
        attachment_min_anchor_voxels=100000,
        attachment_min_size_ratio=500.0,
    )


def test_v7_runner_builds_structural_and_attachment_commands(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    plan = module.build_commands(args)

    assert plan["schema"] == "dense-patch-object-refinement-v7-plan/v1"
    propose = plan["commands"][0]["argv"]
    build = plan["commands"][1]["argv"]
    assert "scripts/propose_geo_patch_object_merges.py" in propose
    assert "--enable-structural-multimaterial" in propose
    assert "scripts/build_geo_patch_objects_from_candidates.py" in build
    assert "--enable-attachment-model" in build
    assert "--enable-structural-multimaterial" in build


def test_v7_runner_rejects_forbidden_viewer_input(tmp_path: Path) -> None:
    module = load_module()
    forbidden = tmp_path / "frame_object_points_stride10.ply"
    forbidden.write_bytes(b"bad")
    with pytest.raises(ValueError, match="forbidden input"):
        module.existing_file(forbidden, "region input")


def test_v7_runner_main_writes_plan_in_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    args = base_args(tmp_path)
    argv = [
        "run_dense_patch_object_refinement_v7.py",
        "--state",
        str(args.state),
        "--region-input",
        str(args.region_input),
        "--patch-labels",
        str(args.patch_labels),
        "--output-dir",
        str(args.output_dir),
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert module.main() == 0
    plan_path = args.output_dir / "dense_patch_object_refinement_v7_plan.json"
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    assert data["commands"][0]["name"] == "propose_candidates"
    assert data["commands"][1]["name"] == "build_objects"


def test_v7_runner_does_not_require_state_when_patch_labels_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    args = base_args(tmp_path)
    missing_state = tmp_path / "missing_state.json"
    argv = [
        "run_dense_patch_object_refinement_v7.py",
        "--state",
        str(missing_state),
        "--region-input",
        str(args.region_input),
        "--patch-labels",
        str(args.patch_labels),
        "--output-dir",
        str(args.output_dir),
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert module.main() == 0
    assert (args.output_dir / "dense_patch_object_refinement_v7_plan.json").exists()


def test_v7_runner_run_mode_checks_mainline_health_before_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    args = base_args(tmp_path)
    calls: list[tuple[str, list[str]]] = []

    def fake_run_command(argv: list[str], cwd: Path) -> None:
        calls.append(("command", argv))

    def fake_healthcheck(parsed: argparse.Namespace) -> None:
        calls.append(("healthcheck", [str(parsed.mainline_healthcheck)]))

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "run_mainline_healthcheck", fake_healthcheck)
    argv = [
        "run_dense_patch_object_refinement_v7.py",
        "--state",
        str(args.state),
        "--region-input",
        str(args.region_input),
        "--patch-labels",
        str(args.patch_labels),
        "--output-dir",
        str(args.output_dir),
        "--run",
        "--no-require-current-dense-inputs",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 0
    assert calls[0][0] == "healthcheck"
    assert [name for name, _ in calls].count("command") == 2


def test_v7_runner_can_skip_mainline_healthcheck_when_outer_launcher_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    args = base_args(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(module, "run_command", lambda argv, cwd: calls.append("command"))
    argv = [
        "run_dense_patch_object_refinement_v7.py",
        "--state",
        str(args.state),
        "--region-input",
        str(args.region_input),
        "--patch-labels",
        str(args.patch_labels),
        "--output-dir",
        str(args.output_dir),
        "--mainline-healthcheck",
        str(tmp_path / "missing_healthcheck.py"),
        "--skip-mainline-healthcheck",
        "--no-require-current-dense-inputs",
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 0
    assert calls == ["command", "command"]


def test_v7_runner_run_mode_rejects_unregistered_dense_inputs_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    args = base_args(tmp_path)
    argv = [
        "run_dense_patch_object_refinement_v7.py",
        "--state",
        str(args.state),
        "--region-input",
        str(args.region_input),
        "--patch-labels",
        str(args.patch_labels),
        "--output-dir",
        str(args.output_dir),
        "--skip-mainline-healthcheck",
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(ValueError, match="not_current_dense_input"):
        module.main()
