from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "solve_sync_path_from_candidates.py"
    spec = importlib.util.spec_from_file_location("solve_sync_path_from_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_solver_prefers_temporally_smooth_path_over_local_best():
    module = load_module()
    frame_candidates = {
        0: [
            {"frame_id": 0, "cam_id": 0, "video_idx": 0, "score": 0.90},
            {"frame_id": 0, "cam_id": 0, "video_idx": 100, "score": 0.91},
        ],
        100: [
            {"frame_id": 100, "cam_id": 0, "video_idx": 100, "score": 0.80},
            {"frame_id": 100, "cam_id": 0, "video_idx": 400, "score": 0.95},
        ],
        200: [
            {"frame_id": 200, "cam_id": 0, "video_idx": 200, "score": 0.80},
            {"frame_id": 200, "cam_id": 0, "video_idx": 500, "score": 0.95},
        ],
    }
    path = module.solve_cam_path(
        frame_candidates,
        target_ratio=1.0,
        velocity_weight=2.0,
        nonmonotonic_penalty=1000.0,
        score_weight=1.0,
    )
    assert [row["video_idx"] for row in path] == [0, 100, 200]
    summary = module.summarize_path(
        path,
        target_ratio=1.0,
        max_ratio_deviation=0.2,
        max_score_loss_mean=0.20,
        max_score_loss_max=0.30,
    )
    assert summary["accepted"] is True


def test_summary_rejects_non_smooth_path():
    module = load_module()
    path = [
        {"frame_id": 0, "video_idx": 0, "score": 0.9, "score_loss_from_best": 0.0},
        {"frame_id": 100, "video_idx": 500, "score": 0.9, "score_loss_from_best": 0.0},
        {"frame_id": 200, "video_idx": 510, "score": 0.9, "score_loss_from_best": 0.0},
    ]
    summary = module.summarize_path(
        path,
        target_ratio=1.0,
        max_ratio_deviation=0.5,
        max_score_loss_mean=0.10,
        max_score_loss_max=0.25,
    )
    assert summary["accepted"] is False
    assert summary["status"] == "rejected_unstable_temporal_path"
    assert summary["step_ratio"]["max_abs_deviation"] > 0.5


def test_summary_rejects_smooth_but_low_score_path():
    module = load_module()
    path = [
        {"frame_id": 0, "video_idx": 0, "score": 0.5, "score_loss_from_best": 0.0},
        {"frame_id": 100, "video_idx": 100, "score": 0.2, "score_loss_from_best": 0.4},
        {"frame_id": 200, "video_idx": 200, "score": 0.2, "score_loss_from_best": 0.4},
    ]
    summary = module.summarize_path(
        path,
        target_ratio=1.0,
        max_ratio_deviation=0.2,
        max_score_loss_mean=0.10,
        max_score_loss_max=0.25,
    )
    assert summary["accepted"] is False
    assert summary["score_loss_from_independent_best"]["mean"] > 0.10
