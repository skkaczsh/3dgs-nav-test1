import argparse

from scripts.apply_geometry_conflict_relabels import choose_relabel
from scripts.apply_priority_guard_to_full_scene import transform_object


def _args(**overrides):
    values = {
        "allow_surface_relabel": False,
        "clean_horizontal_planarity": 0.8,
        "clean_horizontal_thickness": 0.5,
        "wall_horizontal_z_split": None,
        "wall_horizontal_low_label": "floor",
        "wall_horizontal_high_label": "floor",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_geometry_conflict_guard_preserves_trusted_surface_by_default() -> None:
    finding = {
        "semantic_label": "wall",
        "reasons": ["wall_has_horizontal_normal"],
        "metrics": {"planarity": 0.95, "thickness_rms": 0.1, "centroid_z": 1.0},
    }

    new_label, reason = choose_relabel(finding, _args())

    assert new_label is None
    assert reason == "trusted_surface_relabel_locked"


def test_geometry_conflict_guard_can_explicitly_relabel_surface() -> None:
    finding = {
        "semantic_label": "wall",
        "reasons": ["wall_has_horizontal_normal"],
        "metrics": {"planarity": 0.95, "thickness_rms": 0.1, "centroid_z": 1.0},
    }

    new_label, reason = choose_relabel(finding, _args(allow_surface_relabel=True))

    assert new_label == "floor"
    assert reason == "wall_clean_horizontal_surface_to_floor"


def test_priority_guard_preserves_rejected_surface_label_by_default() -> None:
    obj = {"object_id": 7, "semantic_label": "wall", "status": "input"}
    guard = {7: {"object_id": 7, "priority_guard_status": "geometry_rejected", "priority_guard_reasons": ["demo"]}}

    out = transform_object(obj, guard)

    assert out["semantic_label"] == "wall"
    assert out["status"] == "priority_geometry_rejected_surface_preserved"


def test_priority_guard_can_explicitly_demote_surface() -> None:
    obj = {"object_id": 7, "semantic_label": "wall", "status": "input"}
    guard = {7: {"object_id": 7, "priority_guard_status": "geometry_rejected", "priority_guard_reasons": ["demo"]}}

    out = transform_object(obj, guard, preserve_surface_rejections=False)

    assert out["semantic_label"] == "unknown"
    assert out["status"] == "priority_geometry_rejected"
