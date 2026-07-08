from pathlib import Path

import laspy
import numpy as np

from scripts.las_to_voxel_ascii_ply import aggregate_las


def test_aggregate_las_merges_voxels_across_chunks(tmp_path: Path):
    path = tmp_path / "tiny.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    las.x = np.array([0.01, 0.02, 1.01, 1.02])
    las.y = np.array([0.01, 0.02, 1.01, 1.02])
    las.z = np.array([0.01, 0.02, 1.01, 1.02])
    las.red = np.array([2560, 5120, 7680, 10240], dtype=np.uint16)
    las.green = np.array([0, 0, 0, 0], dtype=np.uint16)
    las.blue = np.array([0, 0, 0, 0], dtype=np.uint16)
    las.write(path)

    accum, report = aggregate_las(path, voxel_size=1.0, chunk_size=2)

    assert report["point_count"] == 4
    assert report["voxel_count"] == 2
    assert accum["count"].tolist() == [2.0, 2.0]
    assert np.allclose(accum["x"], [0.015, 1.015])
    assert np.allclose(accum["red"], [15.0, 35.0])
