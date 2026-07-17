from pathlib import Path

import numpy as np

from scripts.build_object_image_evidence import choose_frame_pool, global_depth_map_path
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


def test_global_view_selection_ignores_source_frame_column() -> None:
    points = np.array([[1.0, 2.0, 3.0, 71.0]], dtype="float32")
    poses = [{"frame_id": 0, "pos": [1.0, 2.0, 3.0]}, {"frame_id": 10, "pos": [20.0, 2.0, 3.0]}]
    selected = choose_frame_pool(points, poses, max_frames=1, max_distance=0.0, mode="nearby", min_depth=0.1)
    assert [row["frame_id"] for row in selected] == [0]
