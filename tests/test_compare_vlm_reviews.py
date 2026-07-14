from scripts.compare_vlm_reviews import compare


def test_compare_tracks_only_label_changes() -> None:
    old = {1: {"object_id": 1, "parsed": {"controlled_label": "wall", "confidence": 0.8}}}
    new = {1: {"object_id": 1, "parsed": {"controlled_label": "floor", "confidence": 0.9}}}
    report, changed = compare(old, new)
    assert report["changed_labels"] == 1
    assert report["transitions"] == {"wall->floor": 1}
    assert changed[0]["confidence_delta"] == 0.1
