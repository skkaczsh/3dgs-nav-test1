from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "extract_undistorted_frames_jpeg_for_test",
    SCRIPTS / "extract_undistorted_frames_jpeg.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_find_best_video_frame_returns_nearest_timestamp():
    video_ts = np.asarray(
        [
            (0, 0.0),
            (1, 0.1),
            (2, 0.2),
            (3, 0.3),
        ],
        dtype=np.float64,
    )

    video_idx, delta, rel_ts = module.find_best_video_frame(0.18, video_ts, max_delta=0.05)

    assert video_idx == 2
    assert abs(delta - 0.02) < 1e-6
    assert abs(rel_ts - 0.2) < 1e-6


def test_find_best_video_frame_rejects_large_delta():
    video_ts = np.asarray([(0, 0.0), (1, 0.1)], dtype=np.float64)

    video_idx, delta, rel_ts = module.find_best_video_frame(0.4, video_ts, max_delta=0.05)

    assert video_idx is None
    assert abs(delta - 0.3) < 1e-6
    assert abs(rel_ts - 0.1) < 1e-6
