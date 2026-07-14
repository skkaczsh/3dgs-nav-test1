from semantic_eval.run_eval import complete_mask_labels


def test_failed_vlm_keeps_masks_unknown() -> None:
    assert complete_mask_labels({}, 2, parse_ok=False) == {"1": "unknown", "2": "unknown"}


def test_successful_vlm_can_use_other_as_a_real_label() -> None:
    assert complete_mask_labels({"1": "wall"}, 2, parse_ok=True) == {"1": "wall", "2": "other"}
