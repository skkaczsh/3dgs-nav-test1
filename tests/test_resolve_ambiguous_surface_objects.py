from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "resolve_ambiguous_surface_objects_for_test",
        SCRIPTS / "resolve_ambiguous_surface_objects.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = {
        "min_dominant_ratio": 0.68,
        "min_geometry_ratio": 0.50,
        "horizontal_normal_z": 0.90,
        "strong_horizontal_normal_z": 0.96,
        "high_horizontal_ceiling_z": 2.2,
        "ground_max_z_span_for_geometry": 1.2,
        "ceiling_min_normal_z": 0.86,
        "ceiling_min_z": 2.2,
        "wall_max_normal_z": 0.45,
        "strong_wall_normal_z": 0.35,
        "wall_min_z_span": 1.8,
    }
    values.update(overrides)
    return type("Args", (), values)()


def obj(votes, normal=(0, 0, 1), centroid=(0, 0, 0), bbox_min=(0, 0, 0), bbox_max=(2, 2, 0.2)):
    return {
        "object_id": 7,
        "semantic_label": "ambiguous",
        "status": "ambiguous_object",
        "label_votes": votes,
        "normal": list(normal),
        "centroid": list(centroid),
        "bbox_3d": {"min": list(bbox_min), "max": list(bbox_max)},
        "point_count": sum(votes.values()),
    }


def test_resolves_high_horizontal_ground_vote_to_ceiling():
    module = load_module()
    row = obj({"ground": 80, "wall": 20}, centroid=(0, 0, 3.0), bbox_min=(0, 0, 2.9), bbox_max=(3, 3, 3.1))

    new_label, reason, meta = module.choose_label(row, args())

    assert new_label == "ceiling"
    assert reason == "dominant_ground_high_horizontal_to_ceiling"
    assert meta["dominant_ratio"] == 0.8


def test_resolves_vertical_dominant_wall_to_wall():
    module = load_module()
    row = obj({"wall": 75, "ground": 25}, normal=(1, 0, 0), bbox_min=(0, 0, 0), bbox_max=(0.2, 3, 2))

    new_label, reason, _meta = module.choose_label(row, args())

    assert new_label == "wall"
    assert reason == "dominant_wall_geometry_ok"


def test_keeps_non_surface_ambiguous_unchanged():
    module = load_module()
    row = obj({"wall": 75, "car": 25})

    new_label, reason, meta = module.choose_label(row, args())

    assert new_label is None
    assert reason == "non_surface_votes"
    assert meta["votes"] == {"wall": 75.0, "car": 25.0}


def test_resolve_objects_updates_status_and_preserves_original():
    module = load_module()
    rows = [obj({"wall": 75, "ground": 25}, normal=(1, 0, 0), bbox_min=(0, 0, 0), bbox_max=(0.2, 3, 2))]

    out, report = module.resolve_objects(rows, args())

    assert out[0]["semantic_label"] == "wall"
    assert out[0]["semantic_label_original"] == "ambiguous"
    assert out[0]["status"] == "surface_ambiguous_resolved"
    assert report["changed_objects"] == 1
