from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "refine_target_fusion_objects_for_test",
    SCRIPTS / "refine_target_fusion_objects.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def args(**overrides):
    base = {
        "horizontal_surface_label": "ground",
        "geometry_relabel_flat_wall": True,
        "flat_wall_max_z_span": 0.45,
        "flat_wall_min_area": 3.0,
        "flat_wall_min_extent": 1.5,
        "ceiling_min_z": 2.5,
        "wall_floor_max_ceiling_ratio": 0.10,
        "wall_floor_max_z_span": 0.8,
        "wall_floor_min_area": 4.0,
        "wall_floor_min_extent": 2.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def make_obj(label: str, bbox_min, bbox_max, normal, label_votes=None):
    return {
        "object_id": "obj_000001",
        "semantic_label": label,
        "bbox_3d": {"min": bbox_min, "max": bbox_max},
        "normal": normal,
        "label_votes": label_votes or {label: 100},
        "description": "",
    }


def test_flat_low_wall_relabels_to_ground():
    obj = make_obj(
        "wall",
        [-4.0, -2.0, -0.4],
        [4.0, 2.0, -0.2],
        [0.0, 0.0, 1.0],
    )

    label, reason = module.choose_label(obj, {}, args())

    assert label == "ground"
    assert reason == "flat_wall_geometry_to_ground"


def test_flat_high_wall_relabels_to_ceiling():
    obj = make_obj(
        "wall",
        [-4.0, -2.0, 3.0],
        [4.0, 2.0, 3.2],
        [0.0, 0.0, 1.0],
    )

    label, reason = module.choose_label(obj, {}, args())

    assert label == "ceiling"
    assert reason == "flat_wall_geometry_to_ceiling"


def test_vertical_wall_is_kept():
    obj = make_obj(
        "wall",
        [-4.0, -0.1, -0.4],
        [4.0, 0.1, 3.2],
        [1.0, 0.0, 0.0],
    )

    label, reason = module.choose_label(obj, {}, args())

    assert label == "wall"
    assert reason == "geometry_vertical_surface"


def test_wall_ground_text_does_not_override_tall_surface():
    obj = make_obj(
        "wall",
        [-4.0, -2.0, -0.4],
        [4.0, 2.0, 2.0],
        [0.0, 0.0, 1.0],
    )
    obj["description"] = "large rooftop floor concrete surface"

    label, reason = module.choose_label(obj, {}, args())

    assert label == "wall"
    assert reason != "wall_text_ground_large_surface"


def test_wall_ground_text_can_relabel_low_surface():
    obj = make_obj(
        "wall",
        [-4.0, -2.0, -0.4],
        [4.0, 2.0, 0.2],
        [0.0, 0.0, 0.5],
    )
    obj["description"] = "large rooftop floor concrete surface"

    label, reason = module.choose_label(obj, {}, args())

    assert label == "ground"
    assert reason == "wall_text_ground_large_surface"
