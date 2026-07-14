from scripts.build_superpoint_region_qa_manifest import build_manifest


def test_manifest_uses_anchor_evidence_and_prioritizes_sparse_support() -> None:
    regions = [
        {"region_id": "region:wall:0", "region_label": "wall", "superpoint_count": 16, "max_hops": 2, "source_anchor_ids": [1], "superpoint_ids": [1]},
        {"region_id": "region:floor:1", "region_label": "floor", "superpoint_count": 16, "max_hops": 2, "source_anchor_ids": [2, 3], "superpoint_ids": [2, 3]},
    ]
    observations = [{"superpoint_id": 1, "evidence_score": 3.0, "overlay_path": "best.jpg", "frame_id": 7, "cam_id": 1}]
    reviews = [{"object_id": 1, "parsed": {"controlled_label": "wall", "description_zh": "wall", "confidence": 0.9}}]
    rows = build_manifest(regions, observations, reviews, [{"object_id": 1, "conflict_reason": "thin_linear_labeled_as_surface"}])
    assert rows[0]["region_id"] == "region:wall:0"
    assert rows[0]["source_anchor_evidence"][0]["overlay_path"] == "best.jpg"
    assert rows[0]["member_geometry_conflict_count"] == 1
