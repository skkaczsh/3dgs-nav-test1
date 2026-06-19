from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "repair_surface_target_labels_for_test",
        SCRIPTS / "repair_surface_target_labels.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = {
        "horizontal_surface_normal_z": 0.92,
        "horizontal_surface_max_z_span": 0.45,
        "horizontal_surface_min_extent": 1.5,
        "horizontal_surface_min_planarity": 0.30,
        "ceiling_min_z": 2.2,
        "ground_to_wall_max_normal_z": 0.45,
        "ground_to_wall_min_z_span": 0.8,
        "ground_to_wall_min_planarity": 0.35,
        "ceiling_to_wall_max_normal_z": 0.45,
        "ceiling_to_wall_min_z_span": 0.6,
        "fine_to_unknown_normal_z": 0.92,
        "fine_to_unknown_max_z_span": 0.25,
        "fine_to_unknown_min_planarity": 0.35,
        "fine_to_unknown_max_linearity": 0.82,
        "example_limit": 10,
    }
    values.update(overrides)
    return type("Args", (), values)()


def target(label: str, target_id: str, bbox_min, bbox_max, normal, planarity=0.4, linearity=0.5):
    return {
        "target_id": target_id,
        "frame_id": 10,
        "cam_id": 1,
        "label": label,
        "raw_label": label,
        "cluster_size": 200,
        "bbox_3d": {"min": list(bbox_min), "max": list(bbox_max)},
        "centroid": [(bbox_min[i] + bbox_max[i]) / 2 for i in range(3)],
        "pca": {"normal": list(normal), "linearity": linearity, "planarity": planarity},
    }


def test_repairs_horizontal_wall_to_ground():
    module = load_module()
    rows = [target("wall", "wall_ground", [0, 0, -0.5], [4, 4, -0.4], [0, 0, 1])]

    out, report = module.repair_targets(rows, args())

    assert out[0]["label"] == "ground"
    assert out[0]["raw_label"] == "wall"
    assert out[0]["surface_repair_reason"] == "wall_horizontal_thin_to_ground"
    assert report["label_flow_counts"] == {"wall->ground": 1}


def test_repairs_high_horizontal_wall_to_ceiling():
    module = load_module()
    rows = [target("wall", "wall_ceiling", [0, 0, 2.7], [3, 3, 2.9], [0, 0, 1])]

    out, report = module.repair_targets(rows, args())

    assert out[0]["label"] == "ceiling"
    assert report["label_flow_counts"] == {"wall->ceiling": 1}


def test_repairs_vertical_ground_to_wall():
    module = load_module()
    rows = [target("ground", "ground_wall", [0, 0, 0], [0.2, 3, 2], [1, 0, 0], planarity=0.6)]

    out, report = module.repair_targets(rows, args())

    assert out[0]["label"] == "wall"
    assert out[0]["surface_repair_reason"] == "ground_vertical_planar_to_wall"
    assert report["label_flow_counts"] == {"ground->wall": 1}


def test_keeps_low_planarity_wall_unchanged():
    module = load_module()
    rows = [target("wall", "rough_wall", [0, 0, -0.5], [4, 4, -0.4], [0, 0, 1], planarity=0.1)]

    out, report = module.repair_targets(rows, args())

    assert out[0]["label"] == "wall"
    assert report["repaired_targets"] == 0
