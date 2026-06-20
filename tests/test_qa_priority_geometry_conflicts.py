import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_priority_geometry_conflicts as mod


def args(**overrides):
    base = {
        "horizontal_normal_z": 0.85,
        "vertical_normal_z": 0.35,
        "surface_min_planarity": 0.70,
        "wall_max_thickness": 1.50,
        "floor_max_z_extent": 4.00,
        "grass_max_z_extent": 8.00,
        "grass_min_planarity": 0.55,
        "car_max_centroid_z": 5.00,
        "car_max_extent": 12.00,
        "car_min_z_extent": 0.45,
        "car_wall_region_min_ratio": 0.90,
        "car_wall_attachment_min_ratio": 0.90,
        "car_wall_min_centroid_z": 2.50,
        "railing_max_extent": 18.00,
        "railing_max_horizontal_thickness": 0.16,
        "railing_surface_min_planarity": 0.80,
        "railing_keep_linearity": 0.82,
        "railing_max_minor_extent": 1.20,
        "tiny_surface_points": 500,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_current_viewer_car_on_vertical_surface_is_flagged():
    obj = {
        "object_id": "obj_003374",
        "viewer_object_id": 3374,
        "semantic_label": "car",
        "point_count": 7505,
        "centroid": [8.4, -15.4, 14.7],
        "bbox_3d": {"min": [0.0, 0.0, 10.0], "max": [3.3, 5.3, 17.1]},
        "normal": [-0.29, 0.78, 0.55],
        "geometry_stats": {"planarity_mean": 0.16, "linearity_mean": 0.82},
        "dominant_structural_region": "vertical_surface_region",
        "dominant_structural_region_ratio": 0.996,
        "dominant_surface_attachment_status": "attached_object_candidate",
        "dominant_surface_attachment_ratio": 0.996,
    }

    severity, reasons, action = mod.assess_object(obj, args())

    assert mod.numeric_object_id(obj) == 3374
    assert severity == "high"
    assert "car_on_vertical_surface_region" in reasons
    assert "car_surface_attached_high" in reasons
    assert action == "relabel_car_to_wall"


def test_low_vehicle_sized_car_is_not_flagged_by_wall_guard():
    obj = {
        "object_id": "obj_001742",
        "viewer_object_id": 1742,
        "semantic_label": "car",
        "point_count": 10000,
        "centroid": [-7.4, 33.8, 0.76],
        "bbox_3d": {"min": [-8.0, 31.0, -0.2], "max": [-5.0, 36.0, 1.8]},
        "normal": [-0.9, -0.1, 0.35],
        "geometry_stats": {"planarity_mean": 0.11, "linearity_mean": 0.84},
        "dominant_structural_region": "unknown",
        "dominant_structural_region_ratio": 1.0,
        "dominant_surface_attachment_status": "ambiguous_surface_attachment",
        "dominant_surface_attachment_ratio": 1.0,
    }

    _severity, reasons, _action = mod.assess_object(obj, args())

    assert "car_on_vertical_surface_region" not in reasons
    assert "car_surface_attached_high" not in reasons


def test_vertical_wall_like_flat_car_preserves_relabel_action():
    obj = {
        "object_id": "obj_003305",
        "viewer_object_id": 3305,
        "semantic_label": "car",
        "point_count": 110,
        "centroid": [0.0, 0.0, 10.3],
        "bbox_3d": {"min": [0.0, 0.0, 10.2], "max": [0.12, 0.14, 10.38]},
        "normal": [0.7, 0.7, 0.2],
        "geometry_stats": {"planarity_mean": 0.20, "linearity_mean": 0.80},
        "dominant_structural_region": "vertical_surface_region",
        "dominant_structural_region_ratio": 1.0,
        "dominant_surface_attachment_status": "attached_object_candidate",
        "dominant_surface_attachment_ratio": 1.0,
    }

    _severity, reasons, action = mod.assess_object(obj, args())

    assert "car_on_vertical_surface_region" in reasons
    assert "car_too_flat" in reasons
    assert action == "relabel_car_to_wall"


def test_horizontal_attached_high_car_fragment_is_not_wall_relabel():
    obj = {
        "object_id": "obj_003140",
        "viewer_object_id": 3140,
        "semantic_label": "car",
        "point_count": 307,
        "centroid": [0.0, 0.0, 11.88],
        "bbox_3d": {"min": [0.0, 0.0, 11.84], "max": [1.66, 2.65, 11.91]},
        "normal": [0.0, 0.0, 1.0],
        "geometry_stats": {"planarity_mean": 0.50, "linearity_mean": 0.70},
        "dominant_structural_region": "ground_like_region",
        "dominant_structural_region_ratio": 1.0,
        "dominant_surface_attachment_status": "attached_object_candidate",
        "dominant_surface_attachment_ratio": 1.0,
    }

    _severity, reasons, action = mod.assess_object(obj, args())

    assert "car_surface_attached_high" not in reasons
    assert "car_high_centroid_z" in reasons
    assert "car_too_flat" in reasons
    assert action == "demote_or_visual_review"


def test_apply_car_wall_geometry_guard_only_relabels_flagged_car(tmp_path):
    from scripts import apply_car_wall_geometry_guard as guard

    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        '{"object_id":"obj_000001","viewer_object_id":1,"semantic_label":"car","point_count":10}\n'
        '{"object_id":"obj_000002","viewer_object_id":2,"semantic_label":"car","point_count":20}\n',
        encoding="utf-8",
    )
    conflicts = tmp_path / "conflicts.jsonl"
    conflicts.write_text(
        '{"object_id":1,"semantic_label":"car","suggested_action":"relabel_car_to_wall","reasons":["car_on_vertical_surface_region"],"metrics":{"centroid_z":8}}\n'
        '{"object_id":2,"semantic_label":"car","suggested_action":"demote_or_visual_review","reasons":["car_too_flat"],"metrics":{}}\n',
        encoding="utf-8",
    )

    rows, report = guard.apply_guard(objects, conflicts, "relabel_car_to_wall")

    assert rows[0]["semantic_label"] == "wall"
    assert rows[0]["semantic_label_original"] == "car"
    assert rows[0]["status"] == "geometry_guard_car_to_wall"
    assert rows[1]["semantic_label"] == "car"
    assert report["candidate_count"] == 1
    assert report["applied_count"] == 1
