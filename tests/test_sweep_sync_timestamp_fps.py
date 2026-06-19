from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sweep_sync_timestamp_fps.py"
    spec = importlib.util.spec_from_file_location("sweep_sync_timestamp_fps", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def test_parse_float_range_supports_sweep_and_singletons():
    module = load_module()
    assert module.parse_float_range("6:7:0.5,8") == [6.0, 6.5, 7.0, 8.0]


def test_solve_for_fps_prefers_matching_timestamp_rate():
    module = load_module()
    grouped = {
        0: {
            0: [
                {"frame_id": 0, "cam_id": 0, "video_idx": 0, "score": 0.9, "sync_timestamp": 0.0},
            ],
            1: [
                {"frame_id": 1, "cam_id": 0, "video_idx": 7, "score": 0.9, "sync_timestamp": 1.0},
                {"frame_id": 1, "cam_id": 0, "video_idx": 9, "score": 0.91, "sync_timestamp": 1.0},
            ],
            2: [
                {"frame_id": 2, "cam_id": 0, "video_idx": 14, "score": 0.9, "sync_timestamp": 2.0},
                {"frame_id": 2, "cam_id": 0, "video_idx": 18, "score": 0.91, "sync_timestamp": 2.0},
            ],
        }
    }
    args = argparse.Namespace(
        target_ratio=1.0,
        velocity_weight=2.0,
        nonmonotonic_penalty=1000.0,
        score_weight=1.0,
        max_ratio_deviation=0.2,
        max_score_loss_mean=0.20,
        max_score_loss_max=0.30,
        timestamp_phase_fraction=0.0,
    )

    report7, paths7 = module.solve_for_fps(grouped, args, 7.0)
    report10, paths10 = module.solve_for_fps(grouped, args, 10.0)

    assert [row["video_idx"] for row in paths7[0]] == [0, 7, 14]
    assert report7["accepted"] is True
    assert report7["max_step_deviation"] == 0.0
    assert report10["max_step_deviation"] > report7["max_step_deviation"]


def test_solve_for_fps_can_sweep_absolute_intercept():
    module = load_module()
    grouped = {
        0: {
            0: [
                {"frame_id": 0, "cam_id": 0, "video_idx": 0, "score": 0.95, "sync_timestamp": 0.0},
                {"frame_id": 0, "cam_id": 0, "video_idx": 100, "score": 0.80, "sync_timestamp": 0.0},
            ],
            1: [
                {"frame_id": 1, "cam_id": 0, "video_idx": 10, "score": 0.95, "sync_timestamp": 1.0},
                {"frame_id": 1, "cam_id": 0, "video_idx": 110, "score": 0.80, "sync_timestamp": 1.0},
            ],
            2: [
                {"frame_id": 2, "cam_id": 0, "video_idx": 20, "score": 0.95, "sync_timestamp": 2.0},
                {"frame_id": 2, "cam_id": 0, "video_idx": 120, "score": 0.80, "sync_timestamp": 2.0},
            ],
        }
    }
    args = argparse.Namespace(
        target_ratio=1.0,
        velocity_weight=2.0,
        nonmonotonic_penalty=1000.0,
        score_weight=1.0,
        max_ratio_deviation=0.2,
        max_score_loss_mean=0.20,
        max_score_loss_max=0.30,
        intercept_values="100:100:100",
        absolute_prior_weight=1.0,
        absolute_prior_tolerance=25.0,
        timestamp_phase_fraction=0.5,
    )

    report, paths = module.solve_for_fps(grouped, args, 10.0)

    assert report["cam_intercepts"] == {"0": 100.0}
    assert report["timestamp_phase_fraction"] == 0.5
    assert [row["video_idx"] for row in paths[0]] == [100, 110, 120]
    assert report["cam_reports"]["0"]["absolute_prior_error"]["mean"] == 0.0
