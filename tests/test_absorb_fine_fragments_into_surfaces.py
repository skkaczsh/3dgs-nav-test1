from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "absorb_fine_fragments_into_surfaces_for_test",
        SCRIPTS / "absorb_fine_fragments_into_surfaces.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = {
        "car_max_absorb_points": 80,
        "railing_max_absorb_points": 80,
        "min_surface_points": 120,
        "max_bbox_gap": 0.35,
        "max_centroid_distance": 1.2,
        "max_color_distance": 90.0,
        "surface_like_normal_z": 0.92,
        "car_surface_like_max_z_span": 0.25,
        "railing_surface_like_max_z_span": 0.25,
        "railing_surface_like_min_planarity": 0.35,
        "railing_keep_linearity": 0.82,
        "horizontal_absorb_max_z_span": 0.25,
        "wall_absorb_max_normal_z": 0.35,
        "wall_absorb_min_z_span": 0.4,
        "wall_absorb_min_planarity": 0.35,
        "demote_unabsorbed_weak_label": None,
        "example_limit": 10,
    }
    values.update(overrides)
    return type("Args", (), values)()


def target(label: str, target_id: str, cluster_size: int, bbox_min, bbox_max, **extra):
    row = {
        "target_id": target_id,
        "frame_id": 10,
        "cam_id": 1,
        "label": label,
        "raw_label": label,
        "cluster_size": cluster_size,
        "bbox_3d": {"min": list(bbox_min), "max": list(bbox_max)},
        "centroid": [(bbox_min[i] + bbox_max[i]) / 2 for i in range(3)],
        "mean_color": [100, 100, 100],
        "pca": {"normal": [0, 0, 1], "linearity": 0.2, "planarity": 0.7},
        "point_indices": list(range(cluster_size)),
    }
    row.update(extra)
    return row


def test_absorbs_flat_car_fragment_into_nearby_ground():
    module = load_module()
    rows = [
        target("ground", "surface", 500, [0, 0, 0], [3, 3, 0.05]),
        target("car", "car_frag", 20, [1, 1, 0.02], [1.5, 1.4, 0.04]),
    ]

    out, report = module.absorb_targets(rows, args())

    absorbed = next(row for row in out if row["target_id"] == "car_frag")
    assert absorbed["label"] == "ground"
    assert absorbed["raw_label"] == "car"
    assert absorbed["absorbed_into_surface_target_id"] == "surface"
    assert report["label_flow_counts"] == {"car->ground": 1}


def test_keeps_weak_fine_fragment_without_near_surface():
    module = load_module()
    rows = [
        target("ground", "surface", 500, [10, 10, 0], [13, 13, 0.05]),
        target("railing", "rail_frag", 20, [1, 1, 0.02], [1.5, 1.4, 0.04]),
    ]

    out, report = module.absorb_targets(rows, args())

    rail = next(row for row in out if row["target_id"] == "rail_frag")
    assert rail["label"] == "railing"
    assert report["absorbed_targets"] == 0
    assert report["unabsorbed_weak_fine_targets"] == 1


def test_keeps_vertical_car_fragment_near_ground():
    module = load_module()
    rows = [
        target("ground", "surface", 500, [0, 0, 0], [3, 3, 0.05]),
        target(
            "car",
            "car_frag",
            20,
            [1, 1, 0.02],
            [1.5, 1.4, 0.9],
            pca={"normal": [1, 0, 0], "linearity": 0.9, "planarity": 0.05},
        ),
    ]

    out, report = module.absorb_targets(rows, args())

    car = next(row for row in out if row["target_id"] == "car_frag")
    assert car["label"] == "car"
    assert report["absorbed_targets"] == 0


def test_absorbs_wall_like_railing_fragment_into_wall():
    module = load_module()
    rows = [
        target(
            "wall",
            "surface",
            500,
            [0, 0, 0],
            [0.08, 3, 2],
            pca={"normal": [1, 0, 0], "linearity": 0.2, "planarity": 0.7},
        ),
        target(
            "railing",
            "rail_frag",
            40,
            [0.02, 1, 0.3],
            [0.12, 1.5, 1.1],
            pca={"normal": [1, 0, 0], "linearity": 0.3, "planarity": 0.6},
        ),
    ]

    out, report = module.absorb_targets(rows, args())

    rail = next(row for row in out if row["target_id"] == "rail_frag")
    assert rail["label"] == "wall"
    assert rail["absorbed_into_surface_target_id"] == "surface"
    assert report["label_flow_counts"] == {"railing->wall": 1}


def test_can_demote_unabsorbed_weak_fragment_to_unknown():
    module = load_module()
    rows = [
        target("ground", "surface", 500, [10, 10, 0], [13, 13, 0.05]),
        target("railing", "rail_frag", 20, [1, 1, 0.02], [1.5, 1.4, 0.04]),
    ]

    out, report = module.absorb_targets(rows, args(demote_unabsorbed_weak_label="unknown"))

    rail = next(row for row in out if row["target_id"] == "rail_frag")
    assert rail["label"] == "unknown"
    assert rail["raw_label"] == "railing"
    assert rail["demotion_reason"] == "unabsorbed_weak_fine"
    assert report["label_flow_counts"] == {"railing->unknown": 1}
