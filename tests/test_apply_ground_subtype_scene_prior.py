import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import apply_ground_subtype_scene_prior as mod


def test_relabel_ground_subtypes_only_changes_ground_family():
    rows = [
        {
            "object_id": "obj_1",
            "viewer_object_id": 1,
            "semantic_label": "ground",
            "point_count": 10,
            "scene_prior": {"dominant_scene_ground_subtype": "stair", "scene_prior_confidence_mean": 0.9},
        },
        {
            "object_id": "obj_2",
            "viewer_object_id": 2,
            "semantic_label": "grass",
            "point_count": 20,
            "scene_prior": {"dominant_scene_ground_subtype": "ordinary_ground", "scene_prior_confidence_mean": 0.9},
        },
        {
            "object_id": "obj_3",
            "viewer_object_id": 3,
            "semantic_label": "wall",
            "point_count": 30,
            "scene_prior": {"dominant_scene_ground_subtype": "roof", "scene_prior_confidence_mean": 0.9},
        },
        {
            "object_id": "obj_4",
            "viewer_object_id": 4,
            "semantic_label": "floor",
            "point_count": 40,
            "scene_prior": {"dominant_scene_ground_subtype": "indoor_floor", "scene_prior_confidence_mean": 0.1},
        },
    ]

    out, report = mod.relabel_ground_subtypes(rows, min_scene_score=0.5)

    assert out[0]["semantic_label"] == "stair"
    assert out[0]["semantic_id"] == 18
    assert out[0]["semantic_label_original"] == "ground"
    assert out[1]["semantic_label"] == "ground"
    assert out[2]["semantic_label"] == "wall"
    assert out[3]["semantic_label"] == "floor"
    assert report["changed_count"] == 2
    assert report["label_counts_after"] == {"stair": 1, "ground": 1, "wall": 1, "floor": 1}
