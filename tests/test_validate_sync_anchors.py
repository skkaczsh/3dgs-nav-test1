from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_sync_anchors.py"
    spec = importlib.util.spec_from_file_location("validate_sync_anchors", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = dict(
        anchors_jsonl=tmp_path / "anchors.jsonl",
        img_pos_file=tmp_path / "img_pos.txt",
        timestamp_phase_fraction=0.0,
        expected_fps=6.0,
        max_fps_error=2.0,
        cams=[0, 1],
        min_accepted_per_cam=2,
        output=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_anchor_validation_passes_for_monotonic_per_cam_anchors(tmp_path: Path):
    module = load_module()
    args = make_args(tmp_path)
    write_jsonl(args.anchors_jsonl, [
        {"frame_id": 0, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 10},
        {"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 16},
        {"frame_id": 0, "cam_id": 1, "anchor_status": "accepted", "selected_video_idx": 12},
        {"frame_id": 10, "cam_id": 1, "anchor_status": "accepted", "selected_video_idx": 18},
    ])
    args.img_pos_file.write_text("0 0.0\n10 1.0\n", encoding="utf-8")

    report = module.build_report(args)

    assert report["passed"] is True
    assert report["cam_reports"]["0"]["negative_step_count"] == 0
    assert report["cam_reports"]["1"]["accepted_count"] == 2


def test_anchor_validation_fails_when_cam_coverage_is_insufficient(tmp_path: Path):
    module = load_module()
    args = make_args(tmp_path)
    write_jsonl(args.anchors_jsonl, [
        {"frame_id": 0, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 10},
        {"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 16},
    ])
    args.img_pos_file.write_text("0 0.0\n10 1.0\n", encoding="utf-8")

    report = module.build_report(args)

    assert report["passed"] is False
    assert "accepted_anchors_cam1=0<min2" in report["errors"]


def test_anchor_validation_fails_for_negative_video_step(tmp_path: Path):
    module = load_module()
    args = make_args(tmp_path, cams=[0], min_accepted_per_cam=2)
    write_jsonl(args.anchors_jsonl, [
        {"frame_id": 0, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 20},
        {"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 16},
    ])
    args.img_pos_file.write_text("0 0.0\n10 1.0\n", encoding="utf-8")

    report = module.build_report(args)

    assert report["passed"] is False
    assert "negative_video_steps_cam0=1" in report["errors"]


def test_anchor_validation_applies_timestamp_phase_to_implied_fps(tmp_path: Path):
    module = load_module()
    args = make_args(tmp_path, cams=[0], min_accepted_per_cam=2, timestamp_phase_fraction=0.5)
    write_jsonl(args.anchors_jsonl, [
        {"frame_id": 0, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 10},
        {"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 16},
    ])
    args.img_pos_file.write_text("0 0.0\n10 1.0\n20 3.0\n", encoding="utf-8")

    report = module.build_report(args)

    # phase=0.5 changes timestamps to 0.5 and 2.0, so 6 video frames imply 4 fps.
    assert report["passed"] is True
    assert report["cam_reports"]["0"]["implied_fps"]["mean"] == 4.0
