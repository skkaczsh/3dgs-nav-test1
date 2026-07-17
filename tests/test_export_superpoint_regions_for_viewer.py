from pathlib import Path

from scripts.build_object_image_evidence import global_depth_map_path
from scripts.export_superpoint_regions_for_viewer import region_lookup


def test_region_lookup_assigns_each_superpoint_once() -> None:
    lookup, objects = region_lookup(
        [{"superpoint_id": 7, "region_id": "region:wall:0"}],
        [{"region_id": "region:wall:0", "region_label": "wall", "superpoint_ids": [7]}],
    )
    assert lookup == {7: 1}
    assert objects[1]["semantic_label"] == "wall"


def test_global_depth_map_path_is_frame_exact() -> None:
    assert global_depth_map_path(Path("/depth/maps"), 2, 71) == Path("/depth/maps/cam2_000071_geometry.npz")
