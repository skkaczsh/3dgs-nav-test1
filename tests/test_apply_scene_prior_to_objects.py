import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import apply_scene_prior_to_objects as mod


def test_scene_prior_votes_map_object_frames_to_segments():
    scene_prior = {
        "schema": "mimo-scene-prior/v1",
        "segments": [
            {
                "segment_id": "scene_000",
                "start_frame": 0,
                "end_frame": 99,
                "area_type": "outdoor_parking",
                "area_name_zh": "停车场",
                "expected_labels": ["ground", "car", "wall"],
                "unlikely_labels": ["stair"],
                "ground_subtypes": ["ordinary_ground"],
                "confidence": 0.9,
            },
            {
                "segment_id": "scene_001",
                "start_frame": 100,
                "end_frame": 199,
                "area_type": "stairwell",
                "area_name_zh": "楼梯间",
                "expected_labels": ["stair", "railing", "wall"],
                "unlikely_labels": ["car", "grass"],
                "ground_subtypes": ["stair"],
                "confidence": 0.8,
            },
        ],
    }
    obj = {"object_id": "obj_1", "semantic_label": "car", "frames": [30, 40, 120]}

    context = mod.vote_scene_context(obj, scene_prior["segments"])

    assert context["frames_with_scene_prior"] == 3
    assert context["dominant_scene_area_type"] == "outdoor_parking"
    assert context["scene_expected_label_score"] == 1.8 / 2.6
    assert context["scene_unlikely_label_score"] == 0.8 / 2.6
    assert context["dominant_scene_ground_subtype"] == "ordinary_ground"


def test_enrich_objects_reports_unlikely_labels():
    objects = [
        {"object_id": "obj_1", "semantic_label": "car", "frames": [120]},
        {"object_id": "obj_2", "semantic_label": "floor", "frames": [120]},
    ]
    scene_prior = {
        "schema": "mimo-scene-prior/v1",
        "segments": [
            {
                "segment_id": "scene_001",
                "start_frame": 100,
                "end_frame": 199,
                "area_type": "stairwell",
                "area_name_zh": "楼梯间",
                "expected_labels": ["stair", "railing", "wall"],
                "unlikely_labels": ["car"],
                "ground_subtypes": ["stair"],
                "confidence": 0.8,
            }
        ],
    }

    enriched, report = mod.enrich_objects(objects, scene_prior)

    assert enriched[0]["scene_prior"]["dominant_scene_area_type"] == "stairwell"
    assert enriched[0]["scene_prior"]["scene_unlikely_label_score"] == 1.0
    assert enriched[1]["scene_prior"]["dominant_scene_ground_subtype"] == "stair"
    assert report["labels_with_unlikely_scene_context"] == {"car": 1}
    assert report["ground_subtype_object_counts"] == {"stair": 2}
