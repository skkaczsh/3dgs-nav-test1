from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_lx_video_frame_mapping.py"
    spec = importlib.util.spec_from_file_location("calibrate_lx_video_frame_mapping", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fit_affine_mapping_accepts_stable_linear_mapping():
    module = load_module()
    rows = [
        {"frame_id": 1000, "video_idx": 900, "score": 0.6},
        {"frame_id": 2000, "video_idx": 1901, "score": 0.7},
        {"frame_id": 3000, "video_idx": 2900, "score": 0.8},
        {"frame_id": 4000, "video_idx": 3902, "score": 0.7},
    ]
    fit = module.fit_affine_mapping(rows, max_rmse=5.0, max_abs_residual=8.0, min_samples=4)
    assert fit["accepted"] is True
    assert fit["status"] == "accepted"
    assert abs(fit["slope"] - 1.0) < 0.01
    assert fit["rmse"] < 2.0


def test_fit_affine_mapping_rejects_unstable_best_offsets():
    module = load_module()
    rows = [
        {"frame_id": 1000, "video_idx": 800, "score": 0.6},
        {"frame_id": 2000, "video_idx": 800, "score": 0.7},
        {"frame_id": 3400, "video_idx": 2600, "score": 0.8},
        {"frame_id": 5000, "video_idx": 5400, "score": 0.7},
        {"frame_id": 6000, "video_idx": 5400, "score": 0.6},
    ]
    fit = module.fit_affine_mapping(rows, max_rmse=150.0, max_abs_residual=300.0, min_samples=4)
    assert fit["accepted"] is False
    assert fit["status"] == "rejected_unstable_fit"
    assert fit["rmse"] > 150.0 or fit["max_abs_residual"] > 300.0


def test_parse_int_range_supports_lists_and_ranges():
    module = load_module()
    assert module.parse_int_range("-4:4:4,10") == [-4, 0, 4, 10]


def test_annotate_direct_rank_adds_summary_and_best_fields():
    module = load_module()
    candidates = [
        {"frame_id": 10, "cam_id": 0, "offset": -1, "video_idx": 9, "score": 0.9},
        {"frame_id": 10, "cam_id": 0, "offset": 0, "video_idx": 10, "score": 0.5},
        {"frame_id": 10, "cam_id": 0, "offset": 1, "video_idx": 11, "score": 0.7},
    ]
    best = [{"frame_id": 10, "cam_id": 0, "offset": -1, "video_idx": 9, "score": 0.9}]
    summary = module.annotate_direct_rank(candidates, best)
    assert summary["count"] == 1
    assert summary["p50"] == 3.0
    assert best[0]["direct_rank"] == 3
    assert best[0]["direct_score"] == 0.5


def test_select_projected_depth_edges_keeps_depth_discontinuity_samples():
    module = load_module()
    uu = np.asarray([1, 2, 3], dtype=np.int32)
    vv = np.asarray([2, 2, 2], dtype=np.int32)
    depths = np.asarray([1.0, 1.1, 3.0], dtype=np.float32)
    edge_u, edge_v, edge_z = module.select_projected_depth_edges(
        uu, vv, depths, width=6, height=5, depth_gap=0.5, dilation_px=1
    )
    assert edge_u.tolist() == [2, 3]
    assert edge_v.tolist() == [2, 2]
    assert np.allclose(edge_z, np.asarray([1.1, 3.0], dtype=np.float32))


def test_select_projected_depth_edges_returns_empty_without_discontinuity():
    module = load_module()
    uu = np.asarray([1, 2, 3], dtype=np.int32)
    vv = np.asarray([2, 2, 2], dtype=np.int32)
    depths = np.asarray([1.0, 1.1, 1.2], dtype=np.float32)
    edge_u, edge_v, edge_z = module.select_projected_depth_edges(
        uu, vv, depths, width=6, height=5, depth_gap=0.5, dilation_px=1
    )
    assert len(edge_u) == 0
    assert len(edge_v) == 0
    assert len(edge_z) == 0


def test_select_projected_silhouette_edges_keeps_boundary_samples():
    module = load_module()
    uu = np.asarray([2, 3, 4, 3], dtype=np.int32)
    vv = np.asarray([3, 3, 3, 4], dtype=np.int32)
    depths = np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    edge_u, edge_v, edge_z = module.select_projected_silhouette_edges(
        uu, vv, depths, width=8, height=8, dilation_px=3
    )
    assert len(edge_u) > 0
    assert len(edge_u) <= len(uu)
    assert len(edge_u) == len(edge_v) == len(edge_z)
