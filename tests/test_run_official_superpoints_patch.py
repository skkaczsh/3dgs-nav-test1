from __future__ import annotations

import json

import numpy as np
import pytest
from plyfile import PlyData

from scripts.run_official_superpoints_patch import crop_points, geometry_by_object, write_objects_jsonl, write_random_color_ply


def test_official_superpoint_export_is_semantic_vote_compatible(tmp_path) -> None:
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    labels = np.array([7, 7, 9], dtype=np.uint32)
    ply_path = tmp_path / "objects.ply"
    jsonl_path = tmp_path / "objects.jsonl"

    write_random_color_ply(ply_path, xyz, labels)
    write_objects_jsonl(
        jsonl_path,
        xyz,
        labels,
        {7: {"geometry_type": "horizontal", "source": "test"}, 9: {"geometry_type": "vertical", "source": "test"}},
    )

    vertex = PlyData.read(str(ply_path))["vertex"].data
    assert {"object", "semantic"} <= set(vertex.dtype.names)
    assert vertex["object"].tolist() == [7, 7, 9]
    assert vertex["semantic"].tolist() == [0, 0, 0]
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert rows[0]["geometry_type"] == "horizontal"
    assert rows[0]["count"] == 2
    assert rows[0]["bbox_min"] == [0.0, 0.0, 0.0]
    assert rows[0]["bbox_max"] == [1.0, 0.0, 0.0]
    assert rows[0]["semantic_label"] == "unknown"
    assert rows[0]["label_policy"] == "geometry_is_not_semantic"


def test_superpoint_pca_distinguishes_horizontal_vertical_and_line() -> None:
    horizontal = np.array([[x, y, 0] for x in range(4) for y in range(4)], dtype=np.float32)
    vertical = np.array([[10, x, y] for x in range(4) for y in range(4)], dtype=np.float32)
    line = np.array([[20 + x, 0, 0] for x in range(12)], dtype=np.float32)
    xyz = np.vstack([horizontal, vertical, line])
    labels = np.array([1] * len(horizontal) + [2] * len(vertical) + [3] * len(line), dtype=np.uint32)

    geometry = geometry_by_object(xyz, labels, None)

    assert geometry[1]["geometry_type"] == "horizontal"
    assert geometry[2]["geometry_type"] == "vertical"
    assert geometry[3]["geometry_type"] == "thin_linear"


def test_spatial_crop_keeps_local_points_and_rejects_invalid_bounds() -> None:
    xyz = np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    rgb = np.zeros((3, 3), dtype=np.uint8)
    cropped, _ = crop_points(xyz, rgb, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    assert cropped.tolist() == [[1.0, 1.0, 1.0]]
    with pytest.raises(ValueError, match="strictly greater"):
        crop_points(xyz, rgb, [1, 1, 1], [1, 2, 2])
