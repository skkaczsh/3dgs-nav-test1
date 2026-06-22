from __future__ import annotations

import argparse
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
        "min_surface_membership_score": 0.45,
        "min_object_membership_score": 0.42,
        "object_color_factor": 1.85,
        "object_texture_delta": 64.0,
        "object_roughness_delta": 0.34,
        "object_height_factor": 3.5,
        "object_texture_weight": 0.30,
        "object_shape_weight": 0.30,
        "object_height_weight": 0.12,
        "object_bucket_weight": 0.12,
        "object_normal_weight": 0.06,
        "object_plane_weight": 0.10,
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


def test_region_model_breaks_pairwise_chain_bridge():
    horizontal = region_model.BUCKET_IDS["horizontal"]
    arrays = make_arrays(
        xyz=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.05], [0.2, 0.0, 0.7]],
        rgb=[[100, 100, 100], [104, 101, 100], [108, 102, 100]],
        normals=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        buckets=[horizontal, horizontal, horizontal],
    )
    adjacency = [[1], [0, 2], [1]]

    labels, patches = region_model.grow_region_model(arrays, adjacency, region_args())

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert sorted(row["voxel_count"] for row in patches) == [1, 2]


def test_region_model_allows_rough_object_normal_change_when_texture_matches():
    rough = region_model.BUCKET_IDS["rough_mixed"]
    arrays = make_arrays(
        xyz=[[0.0, 0.0, 0.0], [0.08, 0.0, 0.05]],
        rgb=[[35, 120, 42], [39, 126, 45]],
        normals=[[0, 0, 1], [1, 0, 0]],
        buckets=[rough, rough],
    )
    adjacency = [[1], [0]]

    labels, patches = region_model.grow_region_model(arrays, adjacency, region_args(max_normal_angle=20.0))

    assert labels[0] == labels[1]
    assert len(patches) == 1
    assert patches[0]["geometry_type"] == "rough_mixed"
