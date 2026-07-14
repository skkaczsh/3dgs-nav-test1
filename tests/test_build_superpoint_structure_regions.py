from scripts.build_superpoint_structure_regions import region_row, structural_regions


def test_only_same_high_confidence_label_can_form_region() -> None:
    posteriors = [
        {"object_id": 1, "propagation_eligible": True, "structural_candidate_label": "floor", "structural_source_anchor": 1},
        {"object_id": 2, "propagation_eligible": True, "structural_candidate_label": "floor", "structural_source_anchor": 1},
        {"object_id": 3, "propagation_eligible": True, "structural_candidate_label": "wall", "structural_source_anchor": 3},
    ]
    edges = [
        {"object_a": 1, "object_b": 2, "shared_voxel_faces": 20, "contact_rgb_distance": 5},
        {"object_a": 2, "object_b": 3, "shared_voxel_faces": 20, "contact_rgb_distance": 5},
    ]
    assert structural_regions(posteriors, edges, 10, 100, 40.0) == [("floor", [1, 2]), ("wall", [3])]


def test_same_label_from_different_anchors_cannot_chain_into_one_region() -> None:
    posteriors = [
        {"object_id": 1, "propagation_eligible": True, "structural_candidate_label": "floor", "structural_source_anchor": 1},
        {"object_id": 2, "propagation_eligible": True, "structural_candidate_label": "floor", "structural_source_anchor": 1},
        {"object_id": 3, "propagation_eligible": True, "structural_candidate_label": "floor", "structural_source_anchor": 3},
    ]
    edges = [
        {"object_a": 1, "object_b": 2, "shared_voxel_faces": 20, "contact_rgb_distance": 5},
        {"object_a": 2, "object_b": 3, "shared_voxel_faces": 20, "contact_rgb_distance": 5},
    ]
    assert structural_regions(posteriors, edges, 10, 100, 40.0) == [("floor", [1, 2]), ("floor", [3])]


def test_region_keeps_member_ownership_and_bounds() -> None:
    row = region_row(
        1, "floor", [3, 4],
        {3: {"bbox_min": [0, 1, 2], "bbox_max": [2, 3, 4]}, 4: {"bbox_min": [-1, 2, 0], "bbox_max": [1, 5, 6]}},
        {3: {"structural_source_anchor": 3, "structural_hops": 0}, 4: {"structural_source_anchor": 3, "structural_hops": 2}},
    )
    assert row["bbox_min"] == [-1.0, 1.0, 0.0]
    assert row["bbox_max"] == [2.0, 5.0, 6.0]
    assert row["ownership_policy"] == "members remain immutable official superpoints"
