from __future__ import annotations

from argparse import Namespace
from collections import Counter


from scripts.split_surface_targets_by_plane import apply_structural_ceiling_relabel


def args(**overrides):
    base = dict(
        ceiling_support_labels=["wall", "building"],
        ceiling_support_source_labels=["floor", "building"],
        ceiling_candidate_labels=["floor"],
        wall_normal_z=0.40,
        floor_normal_z=0.72,
        ceiling_min_z=2.0,
        ceiling_max_xy_area=8.0,
        ceiling_max_z_extent=0.35,
        ceiling_min_minor_extent=0.30,
        ceiling_max_aspect_ratio=4.0,
        ceiling_top_gap_max=0.15,
        ceiling_support_z_gap_max=0.6,
        ceiling_support_xy_gap_max=0.5,
    )
    base.update(overrides)
    return Namespace(**base)


def horizontal_row(label: str, z0: float, z1: float, x1: float = 1.0, y1: float = 1.0) -> dict:
    return {
        "label": label,
        "bbox_3d": {"min": [0.0, 0.0, z0], "max": [x1, y1, z1]},
        "pca": {"normal": [0.0, 0.0, 1.0]},
    }


def vertical_row(label: str, x0: float, x1: float, z0: float, z1: float) -> dict:
    return {
        "label": label,
        "bbox_3d": {"min": [x0, 0.0, z0], "max": [x1, 1.0, z1]},
        "pca": {"normal": [1.0, 0.0, 0.0]},
    }


def test_structural_ceiling_relabel_requires_high_horizontal_candidate_and_vertical_support() -> None:
    ceiling_candidate = horizontal_row("floor", 2.6, 2.7)
    wall_support = vertical_row("wall", 0.0, 0.1, 2.1, 2.55)
    rows = [ceiling_candidate, wall_support]
    counts = Counter({"child_label:floor": 1, "child_label:wall": 1})

    apply_structural_ceiling_relabel(rows, counts, args())

    assert ceiling_candidate["label"] == "ceiling"
    assert ceiling_candidate["semantic_id"] == 4
    assert ceiling_candidate["surface_split_reason"] == "plane_component_structural_ceiling"
    assert counts["ceiling_structural_relabels"] == 1
    assert counts["child_label:floor"] == 0
    assert counts["child_label:ceiling"] == 1


def test_structural_ceiling_relabel_does_not_fire_without_nearby_support() -> None:
    candidate = horizontal_row("floor", 2.6, 2.7)
    far_wall = vertical_row("wall", 5.0, 5.1, 2.1, 2.55)
    rows = [candidate, far_wall]
    counts = Counter({"child_label:floor": 1, "child_label:wall": 1})

    apply_structural_ceiling_relabel(rows, counts, args())

    assert candidate["label"] == "floor"
    assert counts["ceiling_structural_relabels"] == 0
