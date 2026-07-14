from semantic_eval.run_eval import complete_mask_labels, decode_sam_segmentation


def test_failed_vlm_keeps_masks_unknown() -> None:
    assert complete_mask_labels({}, 2, parse_ok=False) == {"1": "unknown", "2": "unknown"}


def test_successful_vlm_can_use_other_as_a_real_label() -> None:
    assert complete_mask_labels({"1": "wall"}, 2, parse_ok=True) == {"1": "wall", "2": "other"}


def test_uncompressed_rle_matches_sam2_mask_layout() -> None:
    mask = decode_sam_segmentation({"size": [2, 3], "counts": [1, 2, 3]})
    assert mask.tolist() == [[False, True, False], [True, False, False]]
