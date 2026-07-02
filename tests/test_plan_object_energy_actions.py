from __future__ import annotations

from scripts import plan_object_energy_actions as module


def test_plans_split_before_overlap_for_mixed_overlapping_object() -> None:
    row = {
        "object_id": 7,
        "semantic_label": "wall",
        "geometry_type": "mixed",
        "voxel_count": 10000,
        "patch_count": 1,
        "bucket_purity": 0.4,
        "bucket_entropy": 1.5,
        "dominant_bucket": "unknown",
        "flags": [
            "coarse_voxel_overlap_with_other_object",
            "surface_label_on_mixed_geometry",
            "low_bucket_purity_large_object",
        ],
        "energy_score": 4.0,
    }

    action = module.build_action(row)

    assert action["action"] == "split_then_overlap_review"
    assert action["priority_score"] > 6.0


def test_plans_semantic_review_for_label_only_issue() -> None:
    row = {
        "object_id": 9,
        "semantic_label": "railing",
        "geometry_type": "vertical",
        "voxel_count": 200,
        "flags": ["railing_label_without_linear_support"],
        "energy_score": 1.0,
    }

    action = module.build_action(row)

    assert action["action"] == "semantic_review_only"


def test_plan_actions_filters_monitor_rows() -> None:
    qa = {
        "schema": "object-energy-qa/v1",
        "top_problem_objects": [
            {"object_id": 1, "flags": [], "energy_score": 1.0},
            {"object_id": 2, "flags": ["surface_label_on_mixed_geometry"], "energy_score": 2.0},
        ],
    }

    actions, report = module.plan_actions(qa, top_n=10, include_monitor=False, min_priority_score=0.0)

    assert [row["object_id"] for row in actions] == [2]
    assert report["action_counts"] == {"split_geometry_mixed_object": 1}
