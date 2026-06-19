from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_sync_anchor_review_pack.py"
    spec = importlib.util.spec_from_file_location("build_sync_anchor_review_pack", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_choose_review_options_keeps_direct_best_smooth_and_top_unique():
    module = load_module()
    candidates = [
        {"frame_id": 10, "cam_id": 0, "video_idx": 12, "offset": 2, "score": 0.9},
        {"frame_id": 10, "cam_id": 0, "video_idx": 10, "offset": 0, "score": 0.5},
        {"frame_id": 10, "cam_id": 0, "video_idx": 11, "offset": 1, "score": 0.8},
        {"frame_id": 10, "cam_id": 0, "video_idx": 13, "offset": 3, "score": 0.7},
    ]
    smooth = {"frame_id": 10, "cam_id": 0, "video_idx": 11, "offset": 1, "score": 0.8}
    options = module.choose_review_options(candidates, smooth, top_n=3)
    assert [row["video_idx"] for row in options] == [10, 12, 11]
    assert [row["review_source"] for row in options] == ["direct", "independent_best", "smooth_path"]


def test_choose_review_options_handles_missing_direct_and_smooth():
    module = load_module()
    candidates = [
        {"frame_id": 10, "cam_id": 0, "video_idx": 8, "offset": -2, "score": 0.9},
        {"frame_id": 10, "cam_id": 0, "video_idx": 9, "offset": -1, "score": 0.8},
    ]
    options = module.choose_review_options(candidates, None, top_n=2)
    assert [row["video_idx"] for row in options] == [8, 9]
    assert options[0]["review_source"] == "independent_best"
    assert options[1]["review_source"] == "top_candidate"


def test_panel_filename_is_stable_and_informative():
    module = load_module()
    row = {"review_source": "smooth/path", "video_idx": 987}
    assert module.panel_filename(123, 2, 1, row) == "frame_000123_cam2_opt1_smooth_path_v000987.jpg"


def test_build_review_html_contains_export_logic_and_panel_paths():
    module = load_module()
    html = module.build_review_html([
        {
            "frame_id": 10,
            "cam_id": 1,
            "anchor_status": "unreviewed",
            "selected_video_idx": None,
            "selected_option_idx": None,
            "notes": "",
            "options": [
                {
                    "option_idx": 0,
                    "review_source": "direct",
                    "video_idx": 10,
                    "offset": 0,
                    "score": 0.5,
                    "edge_hit": 0.25,
                    "edge_distance_mean": 4.0,
                    "panel_path": "panels/frame_000010_cam1_opt0_direct_v000010.jpg",
                }
            ],
        }
    ])
    assert "LiDAR/Video Sync Anchor Review" in html
    assert "Export accepted JSONL" in html
    assert "accepted_sync_anchors.jsonl" in html
    assert "panels/frame_000010_cam1_opt0_direct_v000010.jpg" in html
