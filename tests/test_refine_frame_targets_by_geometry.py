import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "refine_frame_targets_by_geometry_for_test",
        SCRIPTS / "refine_frame_targets_by_geometry.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = {
        "split_voxel": 0.12,
        "surface_split_voxel": 0.18,
        "railing_split_voxel": 0.06,
        "car_split_voxel": 0.12,
        "split_min_points": 3,
        "surface_split_min_points": 3,
        "min_output_points": 3,
        "min_split_points": 3,
        "surface_min_split_points": 3,
        "surface_max_extent": 12.0,
        "surface_height_split_threshold": 1.2,
        "surface_height_bin": 0.7,
        "surface_planarity": 0.45,
        "railing_min_linearity": 0.45,
        "railing_max_extent": 1.0,
        "car_max_extent": 8.0,
        "car_surface_max_linearity": 0.20,
        "ground_min_normal_z": 0.55,
        "wall_max_normal_z": 0.72,
        "enable_ceiling_label": True,
        "ceiling_min_z": 2.5,
        "keep_residual": False,
    }
    values.update(overrides)
    return type("Args", (), values)()


def target(label="railing", target_index=0):
    return {
        "target_id": "pt_000001_cam0_p5_cc000",
        "target_index": target_index,
        "frame_id": 1,
        "cam_id": 0,
        "mask_id": 5,
        "priority_label_id": 5,
        "label": label,
        "raw_label": label,
        "parent_class": "structure",
        "confidence": 1.0,
        "image_path": "image.jpg",
        "mask_path": "mask.png",
        "point_indices": [0, 1, 2, 3],
        "bbox_2d": {"xyxy": [0, 0, 10, 10], "area": 121},
        "bbox_3d": {"min": [0, 0, 0], "max": [2, 2, 0]},
        "centroid": [1, 1, 0],
        "mean_color": [100, 100, 100],
        "pca": {"normal": [0, 0, 1], "linearity": 0.0, "planarity": 1.0},
        "cluster_size": 4,
    }


def test_refined_label_converts_broad_planar_railing_to_ground():
    module = load_module()
    xs, ys = np.meshgrid(np.linspace(0, 2, 5), np.linspace(0, 2, 5))
    points = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])

    label, reasons = module.refined_label("railing", points.astype(np.float32), args())

    assert label == "ground"
    assert "broad_planar_railing_to_surface" in reasons


def test_refine_targets_reindexes_and_relabels_children():
    module = load_module()
    points = np.array(
        [
            [0, 0, 0],
            [0.05, 0, 0],
            [0, 0.05, 0],
            [0.05, 0.05, 0],
            [2, 2, 0],
            [2.05, 2, 0],
            [2, 2.05, 0],
            [2.05, 2.05, 0],
        ],
        dtype=np.float32,
    )
    base = target()
    base["point_indices"] = list(range(len(points)))
    base["cluster_size"] = len(points)
    base["bbox_3d"] = {"min": [0, 0, 0], "max": [2.05, 2.05, 0]}
    ply_points = {0: {"points": points, "point_indices": np.arange(len(points), dtype=np.int64)}}

    rows, by_target, summary = module.refine_targets([base], ply_points, args(railing_split_voxel=0.10),)

    assert len(rows) == 2
    assert [row["target_index"] for row in rows] == [0, 1]
    assert all(row["label"] == "ground" for row in rows)
    assert all(row["refined_from_target_id"] == base["target_id"] for row in rows)
    assert set(by_target) == {row["target_id"] for row in rows}
    assert summary["split_source_targets"] == 1
    assert summary["relabelled_targets"] == 2


def test_low_wall_with_up_normal_relabels_to_ground():
    module = load_module()
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    points = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])

    label, reasons = module.refined_label("wall", points.astype(np.float32), args())

    assert label == "ground"
    assert "wall_normal_to_ground" in reasons


def test_high_wall_with_up_normal_relabels_to_ceiling():
    module = load_module()
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    points = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 8.0)])

    label, reasons = module.refined_label("wall", points.astype(np.float32), args())

    assert label == "ceiling"
    assert "wall_normal_to_ceiling" in reasons


def test_ground_target_splits_by_height_layers():
    module = load_module()
    low = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0.05, 0.05, 0]], dtype=np.float32)
    high = np.array([[0, 0, 2], [0.05, 0, 2], [0, 0.05, 2], [0.05, 0.05, 2]], dtype=np.float32)
    points = np.vstack([low, high])
    base = target(label="ground")
    base["mask_id"] = 1
    base["priority_label_id"] = 1
    base["parent_class"] = "surface"
    base["point_indices"] = list(range(len(points)))
    base["cluster_size"] = len(points)
    base["bbox_3d"] = {"min": [0, 0, 0], "max": [0.05, 0.05, 2]}
    base["pca"] = {"normal": [0, 0, 1], "linearity": 0.0, "planarity": 1.0}
    ply_points = {0: {"points": points, "point_indices": np.arange(len(points), dtype=np.int64)}}

    rows, _by_target, summary = module.refine_targets(
        [base],
        ply_points,
        args(surface_split_voxel=0.10, split_horizontal_wall_by_height=True),
    )

    assert len(rows) == 2
    assert [row["target_index"] for row in rows] == [0, 1]
    assert all(row["label"] == "ground" for row in rows)
    assert sorted(round(row["centroid"][2], 1) for row in rows) == [0.0, 2.0]
    assert summary["split_source_targets"] == 1


def test_horizontal_wall_splits_by_height_before_relabel():
    module = load_module()
    low = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0.05, 0.05, 0]], dtype=np.float32)
    high = np.array([[0, 0, 8], [0.05, 0, 8], [0, 0.05, 8], [0.05, 0.05, 8]], dtype=np.float32)
    points = np.vstack([low, high])
    base = target(label="wall")
    base["mask_id"] = 2
    base["priority_label_id"] = 2
    base["parent_class"] = "surface"
    base["point_indices"] = list(range(len(points)))
    base["cluster_size"] = len(points)
    base["bbox_3d"] = {"min": [0, 0, 0], "max": [0.05, 0.05, 8]}
    base["pca"] = {"normal": [0, 0, 1], "linearity": 0.0, "planarity": 1.0}
    ply_points = {0: {"points": points, "point_indices": np.arange(len(points), dtype=np.int64)}}

    rows, _by_target, summary = module.refine_targets(
        [base],
        ply_points,
        args(surface_split_voxel=0.10, split_horizontal_wall_by_height=True),
    )

    assert len(rows) == 2
    assert sorted(row["label"] for row in rows) == ["ceiling", "ground"]
    assert sorted(round(row["centroid"][2], 1) for row in rows) == [0.0, 8.0]
    assert summary["split_source_targets"] == 1
    assert summary["relabelled_targets"] == 2


def test_horizontal_wall_height_split_is_opt_in():
    module = load_module()
    low = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0.05, 0.05, 0]], dtype=np.float32)
    high = np.array([[0, 0, 8], [0.05, 0, 8], [0, 0.05, 8], [0.05, 0.05, 8]], dtype=np.float32)
    points = np.vstack([low, high])
    base = target(label="wall")
    base["mask_id"] = 2
    base["priority_label_id"] = 2
    base["parent_class"] = "surface"
    base["point_indices"] = list(range(len(points)))
    base["cluster_size"] = len(points)
    base["bbox_3d"] = {"min": [0, 0, 0], "max": [0.05, 0.05, 8]}
    base["pca"] = {"normal": [0, 0, 1], "linearity": 0.0, "planarity": 1.0}
    ply_points = {0: {"points": points, "point_indices": np.arange(len(points), dtype=np.int64)}}

    rows, _by_target, summary = module.refine_targets([base], ply_points, args(surface_split_voxel=0.10, split_horizontal_wall_by_height=False))

    assert len(rows) == 1
    assert rows[0]["label"] == "wall"
    assert summary["split_source_targets"] == 0
