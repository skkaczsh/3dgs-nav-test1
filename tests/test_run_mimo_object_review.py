from scripts.run_mimo_object_review import completed_review_ids, prompt_for_object, validate_controlled_fields


def test_structure_review_requests_specific_surface_label() -> None:
    prompt = prompt_for_object({"object_id": 7, "geometry_features": {"normal": [0.0, 0.0, 1.0]}}, [], "structure")
    assert "second-pass structural review" in prompt
    assert "wall, roof, ceiling, stair, floor, or grass" in prompt
    assert "building_part or unknown" in prompt
    assert "horizontal_like" in prompt
    assert "surface_attachment" in prompt


def test_object_review_keeps_fine_object_rules() -> None:
    prompt = prompt_for_object({"object_id": 7}, [{"camera_pose_hint": "calibrated"}], "object")
    assert "Car must be an actual vehicle body" in prompt
    assert "WORLD UP arrow" in prompt
    assert "thin rail, pipe, light strip" in prompt
    assert "camera_pose facts" in prompt
    assert "object_view_elevation_deg" in prompt


def test_controlled_fields_reject_freeform_labels_but_keep_description() -> None:
    parsed = {
        "controlled_label": "light_strip",
        "surface_attachment": "ceiling",
        "description_zh": "黑色线性灯带",
    }
    warnings = validate_controlled_fields(parsed)
    assert parsed["controlled_label"] == "unknown"
    assert parsed["surface_attachment"] == "ceiling"
    assert parsed["description_zh"] == "黑色线性灯带"
    assert warnings == ["unsupported_controlled_label=light_strip"]


def test_resume_skips_only_parseable_reviews(tmp_path) -> None:
    path = tmp_path / "mimo_object_review.jsonl"
    path.write_text(
        '{"object_id": 1, "parsed": {"controlled_label": "wall"}}\n'
        '{"object_id": 2, "parsed": null}\n',
        encoding="utf-8",
    )
    assert completed_review_ids(path) == {1}
