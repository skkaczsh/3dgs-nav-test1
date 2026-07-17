from scripts.build_superpoint_soft_unaries import build_row


def test_unobserved_is_not_treated_as_unknown_evidence() -> None:
    row = build_row({"object_id": 7, "geometry_type": "horizontal"}, [], None)
    assert row["state"] == "unobserved"
    assert row["alpha"] == {}


def test_reviewed_evidence_keeps_unknown_probability_mass() -> None:
    evidence = [{"frame_id": 10, "cam_id": 1, "rank": 1, "score": 2.0, "depth_visible_ratio": 1.0, "sky_filtered_ratio": 0.0}]
    review = {"parsed": {"controlled_label": "floor", "confidence": 0.8}}
    row = build_row({"object_id": 7}, evidence, review)
    assert row["state"] == "reviewed"
    assert row["alpha"]["floor"] > row["alpha"]["unknown"] > 0
