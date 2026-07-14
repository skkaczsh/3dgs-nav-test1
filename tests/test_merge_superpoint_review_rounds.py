from scripts.merge_superpoint_review_rounds import merge_rounds


def test_only_specific_high_confidence_structure_overrides_first_pass() -> None:
    first = [{"object_id": 1, "parsed": {"controlled_label": "building_part"}}, {"object_id": 2, "parsed": {"controlled_label": "building_part"}}]
    second = [
        {"object_id": 1, "parsed": {"controlled_label": "wall", "is_surface_fragment": True, "confidence": 0.9}},
        {"object_id": 2, "parsed": {"controlled_label": "building_part", "is_surface_fragment": True, "confidence": 0.99}},
    ]
    rows = merge_rounds(first, second, 0.8)
    assert rows[0]["parsed"]["controlled_label"] == "wall"
    assert rows[0]["review_resolution"] == "specific_structure"
    assert rows[1]["parsed"]["controlled_label"] == "building_part"
    assert rows[1]["review_resolution"] == "first_pass_retained"
