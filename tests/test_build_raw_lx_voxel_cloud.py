from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "build_raw_lx_voxel_cloud_for_test",
    SCRIPTS / "build_raw_lx_voxel_cloud.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_voxel_reduce_averages_points_in_same_voxel():
    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.021, 0.001, 0.001],
        ],
        dtype=np.float32,
    )

    frames = np.array([10, 12, 20], dtype=np.int32)

    keys, accum = module.voxel_reduce(points, voxel_size=0.01, frames=frames)

    assert len(keys) == 2
    counts = sorted(int(x) for x in accum[:, 3])
    assert counts == [1, 2]
    merged = accum[np.argmax(accum[:, 3])]
    assert merged[4] == 10
    assert merged[5] == 12
    assert merged[6] == 22


def test_reduce_key_accum_merges_chunk_votes():
    keys_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    accum_a = np.array([[1, 1, 1, 1, 10, 10, 10], [2, 0, 0, 1, 20, 20, 20]], dtype=np.float64)
    keys_b = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.int32)
    accum_b = np.array([[3, 3, 3, 1, 14, 14, 14], [4, 0, 0, 1, 30, 30, 30]], dtype=np.float64)

    keys, accum = module.reduce_key_accum([keys_a, keys_b], [accum_a, accum_b])

    by_key = {tuple(k.tolist()): a for k, a in zip(keys, accum)}
    assert by_key[(0, 0, 0)].tolist() == [4, 4, 4, 2, 10, 14, 24]
    assert by_key[(1, 0, 0)].tolist() == [2, 0, 0, 1, 20, 20, 20]
    assert by_key[(2, 0, 0)].tolist() == [4, 0, 0, 1, 30, 30, 30]
