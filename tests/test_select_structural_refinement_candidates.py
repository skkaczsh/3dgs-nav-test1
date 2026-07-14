from scripts.select_structural_refinement_candidates import select_candidates, select_specific_anchors


def test_selects_only_high_confidence_generic_surface() -> None:
    objects = [{"object_id": 1}, {"object_id": 2}, {"object_id": 3}]
    reviews = [
        {"object_id": 1, "parsed": {"controlled_label": "building_part", "is_surface_fragment": True, "confidence": 0.9, "description_zh": "facade"}},
        {"object_id": 2, "parsed": {"controlled_label": "building_part", "is_surface_fragment": False, "confidence": 0.99}},
        {"object_id": 3, "parsed": {"controlled_label": "wall", "is_surface_fragment": True, "confidence": 0.99}},
    ]
    selected = select_candidates(objects, reviews, 0.8)
    assert [row["object_id"] for row in selected] == [1]
    assert selected[0]["first_pass_description_zh"] == "facade"


def test_selects_specific_structural_anchors_without_fine_objects() -> None:
    objects = [{"object_id": 1}, {"object_id": 2}]
    reviews = [
        {"object_id": 1, "parsed": {"controlled_label": "wall", "is_surface_fragment": True, "confidence": 0.9}},
        {"object_id": 2, "parsed": {"controlled_label": "railing", "is_surface_fragment": False, "confidence": 0.99}},
    ]
    assert [row["object_id"] for row in select_specific_anchors(objects, reviews, 0.8)] == [1]
