from scripts.run_mimo_object_review import prompt_for_object


def test_structure_review_requests_specific_surface_label() -> None:
    prompt = prompt_for_object({"object_id": 7, "geometry_features": {"normal": [0.0, 0.0, 1.0]}}, [], "structure")
    assert "second-pass structural review" in prompt
    assert "wall, roof, ceiling, stair, floor, or grass" in prompt
    assert "building_part or unknown" in prompt
    assert "horizontal_like" in prompt


def test_object_review_keeps_fine_object_rules() -> None:
    prompt = prompt_for_object({"object_id": 7}, [], "object")
    assert "Car must be an actual vehicle body" in prompt
    assert "WORLD UP arrow" in prompt
