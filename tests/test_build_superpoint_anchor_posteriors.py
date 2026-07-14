from scripts.build_superpoint_anchor_posteriors import anchor_row


def test_only_surface_structure_is_propagation_anchor() -> None:
    object_row = {"object_id": 7, "geometry_type": "horizontal"}
    floor = {"parsed": {"controlled_label": "floor", "confidence": 0.9, "is_surface_fragment": True}}
    car = {"parsed": {"controlled_label": "car", "confidence": 0.99, "is_surface_fragment": False}}
    assert anchor_row(object_row, floor, 0.8)["anchor_label"] == "floor"
    assert anchor_row(object_row, car, 0.8)["anchor_status"] == "local_only"


def test_generic_building_part_requires_refinement_before_propagation() -> None:
    object_row = {"object_id": 8, "geometry_type": "vertical_surface"}
    generic = {"parsed": {"controlled_label": "building_part", "confidence": 0.99, "is_surface_fragment": True}}
    row = anchor_row(object_row, generic, 0.8)
    assert not row["propagation_eligible"]
    assert row["anchor_status"] == "needs_structural_refinement"
