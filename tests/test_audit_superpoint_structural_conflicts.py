from scripts.audit_superpoint_structural_conflicts import audit


def test_audit_flags_incompatible_surface_labels_without_relabeling() -> None:
    objects = [{"object_id": 1, "geometry_type": "thin_linear"}, {"object_id": 2, "geometry_type": "horizontal"}]
    reviews = [
        {"object_id": 1, "parsed": {"controlled_label": "floor", "confidence": 0.9}},
        {"object_id": 2, "parsed": {"controlled_label": "wall", "confidence": 0.8}},
    ]
    rows = audit(objects, reviews)
    assert [row["conflict_reason"] for row in rows] == ["thin_linear_labeled_as_surface", "horizontal_labeled_as_wall"]
