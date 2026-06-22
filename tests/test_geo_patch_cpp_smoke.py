from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_geo_patch_region_model as region_model


def region_args(**overrides):
    values = {
        "max_color_distance": 150.0,
        "max_height_delta": 0.15,
        "max_normal_angle": 35.0,
        "max_plane_residual": 0.10,
        "small_patch_voxels": 2,
        "stable_surface_ratio": 0.72,
        "stable_plane_factor": 2.0,
        "stable_height_factor": 2.0,
        "prototype_distance_scale": 0.54,
        "min_surface_membership_score": 0.45,
        "min_surface_bridge_score": 0.56,
        "enable_surface_multimodal_bridge": True,
        "surface_bridge_texture_score": 0.62,
        "surface_bridge_shape_score": 0.24,
        "surface_bridge_prototype_score": 0.48,
        "min_object_membership_score": 0.42,
        "min_rough_membership_score": 0.44,
        "object_color_factor": 1.85,
        "object_texture_delta": 64.0,
        "object_roughness_delta": 0.34,
        "object_texture_weight": 0.30,
        "object_shape_weight": 0.27,
        "object_prototype_weight": 0.12,
        "object_height_weight": 0.12,
        "object_bucket_weight": 0.12,
        "object_normal_weight": 0.06,
        "object_plane_weight": 0.10,
        "rough_texture_weight": 0.36,
        "rough_shape_weight": 0.29,
        "rough_prototype_weight": 0.18,
        "rough_height_weight": 0.04,
        "rough_bucket_weight": 0.14,
        "rough_normal_weight": 0.03,
        "rough_plane_weight": 0.03,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_arrays(xyz, rgb, normals, buckets):
    n = len(xyz)
    return {
        "keys": np.asarray([[i, 0, 0] for i in range(n)], dtype=np.int64),
        "xyz": np.asarray(xyz, dtype=np.float32),
        "rgb": np.asarray(rgb, dtype=np.float32),
        "normal": np.asarray(normals, dtype=np.float32),
        "roughness": np.asarray([0.02 if b == region_model.BUCKET_IDS["horizontal"] else 0.22 for b in buckets], dtype=np.float32),
        "planarity": np.asarray([0.82 if b == region_model.BUCKET_IDS["horizontal"] else 0.18 for b in buckets], dtype=np.float32),
        "linearity": np.asarray([0.05 if b == region_model.BUCKET_IDS["horizontal"] else 0.25 for b in buckets], dtype=np.float32),
        "height_range": np.asarray([0.02 if b == region_model.BUCKET_IDS["horizontal"] else 0.18 for b in buckets], dtype=np.float32),
        "local_color_std": np.asarray([8.0 if b == region_model.BUCKET_IDS["horizontal"] else 38.0 for b in buckets], dtype=np.float32),
        "buckets": np.asarray(buckets, dtype=np.int16),
    }


def build_cpp(repo: Path) -> tuple[Path, Path]:
    build = subprocess.run(
        ["bash", str(repo / "scripts" / "build_geo_patch_cpp_smoke.sh")],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = [Path(line.strip()) for line in build.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2
    return lines[0], lines[1]


def test_geo_patch_cpp_smoke_builds_and_runs() -> None:
    if shutil.which("g++") is None:
        return

    repo = Path(__file__).resolve().parents[1]
    smoke, grower = build_cpp(repo)
    assert grower.exists()
    run = subprocess.run([str(smoke)], cwd=repo, check=True, text=True, capture_output=True)
    assert "geo_patch_region_model_smoke ok" in run.stdout


def test_geo_patch_cpp_backend_breaks_pairwise_chain(tmp_path: Path) -> None:
    if shutil.which("g++") is None:
        return

    repo = Path(__file__).resolve().parents[1]
    _smoke, grower = build_cpp(repo)
    horizontal = region_model.BUCKET_IDS["horizontal"]
    arrays = make_arrays(
        xyz=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.05], [0.2, 0.0, 0.7]],
        rgb=[[100, 100, 100], [104, 101, 100], [108, 102, 100]],
        normals=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        buckets=[horizontal, horizontal, horizontal],
    )
    args = region_args(cpp_grower=grower, output_dir=tmp_path)
    labels, patches = region_model.grow_region_model_cpp(
        arrays,
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1, 2], dtype=np.int32),
        args,
    )

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert sorted(row["voxel_count"] for row in patches) == [1, 2]
