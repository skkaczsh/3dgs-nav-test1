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


def test_solver_can_use_timestamp_deltas_instead_of_frame_id_deltas():
    module = load_module()
    frame_candidates = {
        0: [
            {"frame_id": 0, "cam_id": 0, "video_idx": 0, "score": 0.9, "sync_timestamp": 0.0},
            {"frame_id": 0, "cam_id": 0, "video_idx": 20, "score": 0.91, "sync_timestamp": 0.0},
        ],
        1: [
            # Real timestamp delta is 2s, so at 10fps the smooth video index is 20.
            {"frame_id": 1, "cam_id": 0, "video_idx": 1, "score": 0.95, "sync_timestamp": 2.0},
            {"frame_id": 1, "cam_id": 0, "video_idx": 20, "score": 0.80, "sync_timestamp": 2.0},
        ],
        2: [
            {"frame_id": 2, "cam_id": 0, "video_idx": 2, "score": 0.95, "sync_timestamp": 4.0},
            {"frame_id": 2, "cam_id": 0, "video_idx": 40, "score": 0.80, "sync_timestamp": 4.0},
        ],
    }

    path = module.solve_cam_path(
        frame_candidates,
        target_ratio=1.0,
        velocity_weight=2.0,
        nonmonotonic_penalty=1000.0,
        score_weight=1.0,
        time_mode="timestamp",
        video_fps=10.0,
    )

    assert [row["video_idx"] for row in path] == [0, 20, 40]
    summary = module.summarize_path(
        path,
        target_ratio=1.0,
        max_ratio_deviation=0.2,
        max_score_loss_mean=0.20,
        max_score_loss_max=0.30,
        time_mode="timestamp",
        video_fps=10.0,
    )
    assert summary["accepted"] is True
    assert summary["step_ratio"]["mode"] == "timestamp"


def test_absolute_timestamp_prior_can_reject_wrong_intercept_path():
    module = load_module()
    frame_candidates = {
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

    without_prior = module.solve_cam_path(
        frame_candidates,
        target_ratio=1.0,
        velocity_weight=2.0,
        nonmonotonic_penalty=1000.0,
        score_weight=1.0,
        time_mode="timestamp",
        video_fps=10.0,
    )
    with_prior = module.solve_cam_path(
        frame_candidates,
        target_ratio=1.0,
        velocity_weight=2.0,
        nonmonotonic_penalty=1000.0,
        score_weight=1.0,
        time_mode="timestamp",
        video_fps=10.0,
        absolute_prior_weight=1.0,
        absolute_prior_tolerance=25.0,
        absolute_intercept=100.0,
    )

    assert [row["video_idx"] for row in without_prior] == [0, 10, 20]
    assert [row["video_idx"] for row in with_prior] == [100, 110, 120]
    assert with_prior[0]["absolute_prior_error"] == 0.0


def test_estimates_absolute_intercept_from_accepted_anchor():
    module = load_module()
    frame_candidates = {
        0: [
            {"frame_id": 0, "cam_id": 0, "video_idx": 100, "score": 0.8, "sync_timestamp": 0.0},
        ],
        1: [
            {"frame_id": 1, "cam_id": 0, "video_idx": 110, "score": 0.8, "sync_timestamp": 1.0, "anchor_status": "accepted"},
        ],
        2: [
            {"frame_id": 2, "cam_id": 0, "video_idx": 120, "score": 0.8, "sync_timestamp": 2.0},
        ],
    }

    assert module.estimate_absolute_intercept_from_anchors(frame_candidates, 10.0, 0.0) == 100.0


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


def test_load_and_apply_accepted_anchors_hard_filters_candidates(tmp_path):
    module = load_module()
    anchors_path = tmp_path / "anchors.jsonl"
    anchors_path.write_text(
        '{"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", '
        '"selected_video_idx": null, "selected_option_idx": 1, '
        '"options": [{"option_idx": 0, "video_idx": 10}, {"option_idx": 1, "video_idx": 12}]}\n',
        encoding="utf-8",
    )
    anchors = module.load_accepted_anchors(anchors_path)
    assert anchors == {(10, 0): 12}
    grouped = {
        0: {
            10: [
                {"frame_id": 10, "cam_id": 0, "video_idx": 10, "score": 0.9},
                {"frame_id": 10, "cam_id": 0, "video_idx": 12, "score": 0.5},
            ]
        }
    }
    filtered = module.apply_anchors(grouped, anchors)
    assert [row["video_idx"] for row in filtered[0][10]] == [12]
    assert filtered[0][10][0]["anchor_status"] == "accepted"


def test_load_frame_timestamps_from_img_pos(tmp_path):
    module = load_module()
    path = tmp_path / "img_pos.txt"
    path.write_text(
        "0 10.0 0 0 0 1 0 0 0 1 0 0 0 0,1 0,1 0,1\n"
        "2 10.2 0 0 0 1 0 0 0 1 0 0 0 0,1 0,1 0,1\n",
        encoding="utf-8",
    )

    assert module.load_frame_timestamps(path) == {0: 10.0, 2: 10.2}


def test_apply_anchors_rejects_missing_candidate():
    module = load_module()
    grouped = {0: {10: [{"frame_id": 10, "cam_id": 0, "video_idx": 10, "score": 0.9}]}}
    try:
        module.apply_anchors(grouped, {(10, 0): 12})
    except ValueError as exc:
        assert "not found in candidates" in str(exc)
    else:
        raise AssertionError("missing anchor candidate should fail")
