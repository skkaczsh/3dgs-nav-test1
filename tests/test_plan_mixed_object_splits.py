from __future__ import annotations

from scripts import plan_mixed_object_splits as module


def test_single_patch_mixed_object_routes_to_patch_generation_split() -> None:
    action = {"object_id": 10, "action": "split_geometry_mixed_object", "priority_score": 5.0}
    obj = {
        "object_id": 10,
        "semantic_label": "wall",
        "geometry_type": "mixed",
        "voxel_count": 1000,
        "patch_count": 1,
        "patch_ids": [99],
        "bucket_counts": {"0": 400, "4": 350, "3": 250},
    }

    candidate = module.build_candidate(action, obj, min_bucket_ratio=0.15)

    assert candidate["split_scope"] == "split_single_patch_by_bucket_connectivity"
    assert candidate["recommended_stage"] == "patch_generation"
    assert [group["bucket"] for group in candidate["split_groups"]] == ["unknown", "rough_mixed", "thin_linear"]


def test_multi_patch_mixed_object_routes_to_patch_then_object_assembly() -> None:
    action = {"object_id": 11, "action": "split_then_overlap_review", "priority_score": 6.0}
    obj = {
        "object_id": 11,
        "semantic_label": "unknown",
        "geometry_type": "rough_mixed",
        "voxel_count": 1000,
        "patch_count": 2,
        "patch_ids": [7, 8],
        "bucket_counts": {"0": 250, "4": 650, "3": 100},
    }

    candidate = module.build_candidate(action, obj, min_bucket_ratio=0.10)

    assert candidate["split_scope"] == "split_or_regroup_multi_patch_object"
    assert candidate["recommended_stage"] == "patch_then_object_assembly"


def test_plan_splits_ignores_non_split_actions() -> None:
    actions = [
        {"object_id": 1, "action": "semantic_review_only", "priority_score": 1.0},
        {"object_id": 2, "action": "split_geometry_mixed_object", "priority_score": 2.0},
    ]
    objects = [
        {"object_id": 2, "patch_count": 1, "patch_ids": [2], "bucket_counts": {"0": 1, "4": 1}},
    ]

    candidates, report = module.plan_splits(
        actions,
        objects,
        limit=10,
        min_bucket_ratio=0.10,
        include_monitor=False,
    )

    assert [row["object_id"] for row in candidates] == [2]
    assert report["scope_counts"] == {"split_single_patch_by_bucket_connectivity": 1}
