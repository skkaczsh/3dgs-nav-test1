from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import accumulate_patch_observations as observations
from scripts import build_geo_patches
from scripts import classify_geo_objects


def patch_args(**overrides):
    values = {
        "input_ply": Path("unused.ply"),
        "output_dir": Path("unused"),
        "structural_field": None,
        "seed_property": "object",
        "point_stride": 1,
        "max_points": 0,
        "patch_voxel_size": 0.25,
        "min_patch_points": 12,
        "mixed_min_points": 12,
        "max_clean_seed_points": 1,
        "local_pca_min_points": 3,
        "clean_planarity_min": 0.70,
        "surface_min_planarity": 0.45,
        "surface_max_thickness": 0.08,
        "wall_max_thickness": 0.08,
        "horizontal_normal_z": 0.86,
        "vertical_normal_z": 0.42,
        "linear_thin_min_linearity": 0.90,
        "linear_thin_min_extent": 0.80,
        "vegetation_min_scattering": 0.05,
        "vegetation_max_z_extent": 2.50,
        "bulky_min_extent": 0.80,
        "bulky_min_z_extent": 0.45,
        "mixed_planarity_max": 0.50,
        "mixed_min_extent": 0.80,
        "upper_surface_min_z_extent": 0.15,
        "upper_surface_max_xy_extent": 3.00,
        "axis_plane_bin_size": 0.10,
        "axis_plane_distance": 0.03,
        "axis_plane_max_planes": 8,
        "structural_sample_points": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def classify_args(**overrides):
    values = {
        "merge_compatible_patches": False,
        "merge_bbox_distance": 0.20,
        "merge_normal_angle": 12.0,
        "min_stable_confidence": 0.55,
        "grass_vote_min": 0.25,
        "grass_green_min": 0.05,
        "railing_vote_min": 0.20,
        "railing_vote_warn_ratio": 0.15,
        "car_vote_min": 0.25,
        "fine_vote_warn_ratio": 0.15,
        "equipment_vote_min": 0.35,
        "unknown_vote_accept_ratio": 0.75,
        "horizontal_normal_z": 0.86,
        "vertical_normal_z": 0.42,
        "surface_salvage_wall_vote_min": 0.65,
        "surface_salvage_ground_vote_min": 0.35,
        "car_min_points": 800,
        "car_min_extent": 1.2,
        "car_min_z_extent": 0.45,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_mixed_seed_splits_horizontal_and_vertical_geometry():
    floor = np.array([[x * 0.1, y * 0.1, 0.0] for x in range(5) for y in range(5)], dtype=np.float32)
    wall = np.array([[0.0, y * 0.1, z * 0.1] for y in range(5) for z in range(1, 6)], dtype=np.float32)
    points = np.vstack([floor, wall])

    components = build_geo_patches.split_seed_points(points, patch_args())
    geometry_types = []
    for comp in components:
        pts = points[comp]
        stats = build_geo_patches.pca_stats(pts)
        extent = pts.max(axis=0) - pts.min(axis=0)
        geometry_types.append(build_geo_patches.geometry_type_from_stats(stats, extent, patch_args()))

    assert "horizontal_surface" in geometry_types
    assert "vertical_surface" in geometry_types


def test_patch_observations_decode_semantic_and_scene_votes():
    patch = {
        "patch_id": "patch_000001",
        "patch_index": 1,
        "point_count": 10,
        "geometry_type": "horizontal_surface",
        "source_votes": {"semantic": {"3": 7, "8": 3}, "priority": {"1": 10}, "frame": {"120": 10}},
        "structural_region_votes": {"ground_like_region": 9, "vertical_surface_region": 1},
    }
    scene_prior = {
        "segments": [
            {
                "start_frame": 100,
                "end_frame": 130,
                "area_type": "indoor_lobby",
                "ground_subtypes": ["indoor_floor"],
                "confidence": 1.0,
            }
        ]
    }

    rows, report = observations.enrich_patches([patch], scene_prior)

    assert rows[0]["evidence"]["dominant_semantic_label"] == "floor"
    assert rows[0]["evidence"]["dominant_priority_label"] == "ground"
    assert rows[0]["evidence"]["scene_prior"]["dominant_scene_ground_subtype"] == "indoor_floor"
    assert report["dominant_scene_area_counts"] == {"indoor_lobby": 1}


def test_vertical_surface_vetoes_car_vote():
    patch = {
        "patch_id": "patch_000001",
        "patch_index": 1,
        "point_count": 100,
        "geometry_type": "vertical_surface",
        "bbox_3d": {"min": [0, 0, 0], "max": [0.1, 2, 2]},
        "centroid": [0, 1, 1],
        "normal": [1, 0, 0],
        "planarity": 0.9,
        "linearity": 0.1,
        "roughness": 0.01,
        "thickness": 0.02,
        "evidence": {
            "semantic_votes": {"car": 80, "wall": 20},
            "priority_votes": {"car": 80, "wall": 20},
            "scene_prior": {"dominant_scene_area_type": "indoor_lobby"},
        },
    }

    classification = classify_geo_objects.classify_patch(patch, classify_args())

    assert classification["canonical_label"] == "wall"
    assert "car_vote_on_vertical_surface_vetoed" in classification["conflict_flags"]


def test_unknown_geometry_indoor_car_vote_is_vetoed():
    patch = {
        "patch_id": "patch_000002",
        "patch_index": 2,
        "point_count": 120,
        "geometry_type": "unknown",
        "extent": [0.4, 0.3, 0.2],
        "bbox_3d": {"min": [0, 0, 0], "max": [0.4, 0.3, 0.2]},
        "centroid": [0.2, 0.15, 0.1],
        "normal": [0, 0, 1],
        "evidence": {
            "semantic_votes": {"car": 90, "wall": 10},
            "scene_prior": {"dominant_scene_area_type": "indoor_lobby"},
        },
    }

    classification = classify_geo_objects.classify_patch(patch, classify_args())

    assert classification["canonical_label"] == "unknown"
    assert "indoor_car_vetoed" in classification["conflict_flags"]


def test_linear_thin_wall_evidence_is_salvaged_as_wall():
    patch = {
        "patch_id": "patch_000003",
        "patch_index": 3,
        "point_count": 5000,
        "geometry_type": "linear_thin",
        "bbox_3d": {"min": [0, 0, 0], "max": [0.05, 3.0, 2.0]},
        "centroid": [0.025, 1.5, 1.0],
        "normal": [1, 0, 0],
        "evidence": {
            "semantic_votes": {"wall": 90, "floor": 10},
            "scene_prior": {"dominant_scene_area_type": "indoor_corridor"},
        },
    }

    classification = classify_geo_objects.classify_patch(patch, classify_args())

    assert classification["canonical_label"] == "wall"
    assert "salvaged_wall_from_non_surface_geometry" in classification["conflict_flags"]


def test_bulky_horizontal_stair_patch_uses_scene_subtype():
    patch = {
        "patch_id": "patch_000004",
        "patch_index": 4,
        "point_count": 3000,
        "geometry_type": "bulky_object",
        "bbox_3d": {"min": [0, 0, 0], "max": [2.0, 1.0, 0.25]},
        "centroid": [1.0, 0.5, 0.125],
        "normal": [0, 0, 1],
        "evidence": {
            "semantic_votes": {"wall": 80, "floor": 20},
            "dominant_structural_region": "ground_like_region",
            "scene_prior": {
                "dominant_scene_area_type": "stairwell",
                "dominant_scene_ground_subtype": "stair",
            },
        },
    }

    classification = classify_geo_objects.classify_patch(patch, classify_args())

    assert classification["canonical_label"] == "stair"
    assert "salvaged_horizontal_surface_from_non_surface_geometry" in classification["conflict_flags"]


def test_bulky_wall_salvage_survives_invalid_car_vote():
    patch = {
        "patch_id": "patch_000005",
        "patch_index": 5,
        "point_count": 5000,
        "geometry_type": "bulky_object",
        "bbox_3d": {"min": [0, 0, 0], "max": [0.1, 4.0, 2.0]},
        "centroid": [0.05, 2.0, 1.0],
        "normal": [1, 0, 0],
        "evidence": {
            "semantic_votes": {"wall": 80, "car": 30},
            "scene_prior": {"dominant_scene_area_type": "indoor_lobby"},
        },
    }

    classification = classify_geo_objects.classify_patch(patch, classify_args())

    assert classification["canonical_label"] == "wall"
    assert "salvaged_wall_from_non_surface_geometry" in classification["conflict_flags"]
    assert "indoor_car_vetoed" in classification["conflict_flags"]


def test_upper_horizontal_structural_region_salvages_ceiling():
    patch = {
        "patch_id": "patch_000006",
        "patch_index": 6,
        "point_count": 3000,
        "geometry_type": "unknown",
        "bbox_3d": {"min": [0, 0, 2.8], "max": [2.0, 2.0, 2.9]},
        "centroid": [1.0, 1.0, 2.85],
        "normal": [0, 0, 1],
        "evidence": {
            "semantic_votes": {"wall": 90, "floor": 10},
            "dominant_structural_region": "upper_horizontal_region",
            "scene_prior": {"dominant_scene_area_type": "indoor_lobby"},
        },
    }

    classification = classify_geo_objects.classify_patch(patch, classify_args())

    assert classification["canonical_label"] == "ceiling"
    assert "salvaged_horizontal_surface_from_non_surface_geometry" in classification["conflict_flags"]
