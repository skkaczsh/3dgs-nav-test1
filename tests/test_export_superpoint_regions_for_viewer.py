from scripts.export_superpoint_regions_for_viewer import region_lookup


def test_region_lookup_assigns_each_superpoint_once() -> None:
    lookup, objects = region_lookup(
        [{"superpoint_id": 7, "region_id": "region:wall:0"}],
        [{"region_id": "region:wall:0", "region_label": "wall", "superpoint_ids": [7]}],
    )
    assert lookup == {7: 1}
    assert objects[1]["semantic_label"] == "wall"
