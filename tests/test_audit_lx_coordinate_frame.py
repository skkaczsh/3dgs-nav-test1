from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_lx_coordinate_frame.py"
    spec = importlib.util.spec_from_file_location("audit_lx_coordinate_frame", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_int_range_supports_lists_and_ranges():
    module = load_module()
    assert module.parse_int_range("1,3:7:2,10") == [1, 3, 5, 7, 10]


def test_classify_coordinate_frame_accepts_world_like_motion():
    module = load_module()
    result = module.classify_coordinate_frame(
        raw_centroid_span=np.asarray([50.0, 60.0, 18.0]),
        pose_span=np.asarray([52.0, 61.0, 18.0]),
        lidar_centroid_span=np.asarray([2.0, 2.2, 1.3]),
        correlations=[0.997, 0.998, 0.996],
        min_corr=0.95,
        max_lidar_span=5.0,
    )
    assert result["world_like"] is True
    assert result["status"] == "world_coordinates_likely"


def test_classify_coordinate_frame_rejects_large_lidar_centroid_span():
    module = load_module()
    result = module.classify_coordinate_frame(
        raw_centroid_span=np.asarray([50.0, 60.0, 18.0]),
        pose_span=np.asarray([52.0, 61.0, 18.0]),
        lidar_centroid_span=np.asarray([12.0, 2.2, 1.3]),
        correlations=[0.997, 0.998, 0.996],
        min_corr=0.95,
        max_lidar_span=5.0,
    )
    assert result["world_like"] is False
