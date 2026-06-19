from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "build_geometry_guidance_maps_for_test",
    SCRIPTS / "build_geometry_guidance_maps.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_read_xyzrgb_ply_supports_binary_little_endian(tmp_path: Path):
    path = tmp_path / "points.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])
    data = np.array([(1.0, 2.0, 3.0, 10, 20, 30), (4.0, 5.0, 6.0, 40, 50, 60)], dtype=dtype)
    path.write_bytes(header + data.tobytes())

    points, colors = module.read_xyzrgb_ply(path)

    assert points.shape == (2, 3)
    assert colors.tolist() == [[10, 20, 30], [40, 50, 60]]


def test_read_xyzrgb_ply_with_metadata_supports_source_frames(tmp_path: Path):
    path = tmp_path / "points_with_frames.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property int frame_min\n"
        "property int frame_max\n"
        "property float frame_mean\n"
        "property uint frame_count\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("frame_min", "<i4"),
        ("frame_max", "<i4"),
        ("frame_mean", "<f4"),
        ("frame_count", "<u4"),
    ])
    data = np.array([(1.0, 2.0, 3.0, 10, 12, 11.0, 2), (4.0, 5.0, 6.0, 30, 35, 32.0, 4)], dtype=dtype)
    path.write_bytes(header + data.tobytes())

    points, colors, metadata = module.read_xyzrgb_ply_with_metadata(path)

    assert points.shape == (2, 3)
    assert colors.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert metadata["frame_min"].tolist() == [10, 30]
    assert metadata["frame_max"].tolist() == [12, 35]
    assert metadata["frame_count"].tolist() == [2, 4]
    assert module.source_frame_mask(metadata, frame_id=12, window=2, mode="mean").tolist() == [True, False]
    assert module.source_frame_mask(metadata, frame_id=29, window=1, mode="span").tolist() == [False, True]


def test_compute_color_edges_marks_lab_boundary():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, :5] = (255, 0, 0)
    rgb[:, 5:] = (0, 255, 0)
    valid = np.ones((10, 10), dtype=bool)

    edge = module.compute_color_edges(rgb, valid, threshold=10.0)

    assert int(edge.sum()) > 0
    assert edge[:, 4].max() == 255 or edge[:, 5].max() == 255
