import numpy as np

from scripts.propose_geo_patch_object_merges import build_grid6_edges


def test_grid6_edges_keep_true_y_neighbor() -> None:
    arrays = {
        "xyz": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    }
    src, dst = build_grid6_edges(arrays, 1.0)
    assert len(src) == 1
    assert len(dst) == 1
    assert {(int(a), int(b)) for a, b in zip(src, dst)} == {(0, 1)}


def test_grid6_edges_keep_true_x_neighbor() -> None:
    arrays = {
        "xyz": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    }
    src, dst = build_grid6_edges(arrays, 1.0)
    assert {(int(a), int(b)) for a, b in zip(src, dst)} == {(0, 1)}


def test_grid6_edges_do_not_wrap_across_linearized_row_boundary() -> None:
    arrays = {
        "xyz": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    }
    src, dst = build_grid6_edges(arrays, 1.0)
    assert len(src) == 0
    assert len(dst) == 0
