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


def test_region_model_allows_tall_rough_object_growth_when_edges_are_local():
    rough = region_model.BUCKET_IDS["rough_mixed"]
    arrays = make_arrays(
        xyz=[[0.0, 0.0, z * 0.12] for z in range(8)],
        rgb=[[35 + z, 120 + z, 42] for z in range(8)],
        normals=[[0, 0, 1] if z % 2 == 0 else [1, 0, 0] for z in range(8)],
        buckets=[rough] * 8,
    )
    adjacency = [[] for _ in range(8)]
    for i in range(7):
        adjacency[i].append(i + 1)
        adjacency[i + 1].append(i)

    labels, patches = region_model.grow_region_model(
        arrays,
        adjacency,
        region_args(max_height_delta=0.10, max_normal_angle=20.0),
    )

    assert len(set(labels.tolist())) == 1
    assert len(patches) == 1
    assert patches[0]["voxel_count"] == 8


def test_rough_membership_downweights_plane_normal_and_centroid_height():
    rough = region_model.BUCKET_IDS["rough_mixed"]
    arrays = make_arrays(
        xyz=[
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.1],
            [0.2, 0.0, 0.2],
            [0.3, 0.0, 0.75],
        ],
        rgb=[
            [35, 120, 42],
            [38, 123, 44],
            [36, 121, 43],
            [39, 124, 45],
        ],
        normals=[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ],
        buckets=[rough, rough, rough, rough],
    )
    model = region_model.PatchModel(seed_index=0, seed_bucket=rough)
    for i in range(3):
        model.add(arrays, i, 1.0)

    ok, score, reason, scores = region_model.membership_score(
        arrays,
        model,
        3,
        region_args(max_height_delta=0.10, max_normal_angle=20.0),
    )

    assert ok, (score, reason, scores)
    assert reason == "accepted"


def test_region_model_matches_existing_local_prototype_not_only_mean():
    rough = region_model.BUCKET_IDS["rough_mixed"]
    arrays = make_arrays(
        xyz=[
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        rgb=[
            [35, 120, 42],
            [37, 123, 43],
            [150, 145, 135],
            [152, 147, 136],
            [36, 121, 43],
        ],
        normals=[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
        ],
        buckets=[rough] * 5,
    )
    model = region_model.PatchModel(seed_index=0, seed_bucket=rough)
    for i in range(4):
        model.add(arrays, i, 1.0)

    ok, score, reason, scores = region_model.membership_score(
        arrays,
        model,
        4,
        region_args(max_color_distance=90.0, object_color_factor=1.2),
    )

    assert ok, (score, reason, scores)
    assert scores["prototype"] > scores["color"]


def test_horizontal_surface_can_bridge_rough_same_texture_substructure():
    horizontal = region_model.BUCKET_IDS["horizontal"]
    rough = region_model.BUCKET_IDS["rough_mixed"]
    arrays = make_arrays(
        xyz=[
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.08],
        ],
        rgb=[
            [110, 108, 102],
            [112, 109, 103],
            [111, 110, 104],
            [113, 111, 105],
        ],
        normals=[
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
        ],
        buckets=[horizontal, horizontal, horizontal, rough],
    )
    model = region_model.PatchModel(seed_index=0, seed_bucket=horizontal)
    for i in range(3):
        model.add(arrays, i, 1.0)

    ok, score, reason, scores = region_model.membership_score(
        arrays,
        model,
        3,
        region_args(max_normal_angle=20.0, max_plane_residual=0.02),
    )

    assert ok, (score, reason, scores)
    assert reason == "accepted"


def test_stable_surface_membership_uses_local_chart_not_global_plane():
    horizontal = region_model.BUCKET_IDS["horizontal"]
    arrays = make_arrays(
        xyz=[
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.0, 0.0, 0.45],
            [1.1, 0.0, 0.45],
            [1.2, 0.0, 0.45],
            [1.3, 0.0, 0.45],
        ],
        rgb=[
            [110, 108, 102],
            [111, 109, 103],
            [112, 110, 104],
            [116, 112, 106],
            [117, 113, 107],
            [116, 112, 106],
            [118, 114, 108],
        ],
        normals=[[0, 0, 1]] * 7,
        buckets=[horizontal] * 7,
    )
    model = region_model.PatchModel(seed_index=0, seed_bucket=horizontal)
    for i in range(6):
        model.add(arrays, i, 1.0)

    ok, score, reason, scores = region_model.membership_score(
        arrays,
        model,
        6,
        region_args(max_height_delta=0.06, max_plane_residual=0.03),
    )

    assert ok, (score, reason, scores)
    assert scores["chart_plane"] > scores["plane"]
    assert scores["chart_height"] > scores["height"]
