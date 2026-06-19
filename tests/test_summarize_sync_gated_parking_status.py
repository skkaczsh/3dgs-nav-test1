from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_sync_gated_parking_status.py"
    spec = importlib.util.spec_from_file_location("summarize_sync_gated_parking_status", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        downloads_anchor=tmp_path / "Downloads" / "accepted_sync_anchors.jsonl",
        staged_anchor=tmp_path / "review" / "accepted_sync_anchors.jsonl",
        anchor_validation=tmp_path / "run" / "accepted_sync_anchor_validation.json",
        readiness_json=tmp_path / "run" / "sync_frame_map_readiness.json",
        readiness_exit=tmp_path / "run" / "sync_frame_map_readiness.exit_code",
        expanded_frame_map=tmp_path / "run" / "expanded_frame_map.jsonl",
        expanded_frame_map_report=tmp_path / "run" / "expanded_frame_map_report.json",
        preflight=tmp_path / "preflight.json",
        frames_summary=tmp_path / "frames" / "extract_report.json",
        priority_summary=tmp_path / "priority" / "priority_segmentation_summary.json",
        review_url="http://review",
    )


def test_status_waits_for_manual_anchors_when_none_exist(tmp_path: Path):
    module = load_module()

    report = module.build_report(args(tmp_path))

    assert report["next_action"]["status"] == "waiting_for_manual_anchors"
    assert "stage_accepted_sync_anchors.py" in report["next_action"]["command"]


def test_status_discovers_latest_downloaded_anchor_export(tmp_path: Path):
    module = load_module()
    a = args(tmp_path)
    old_export = a.downloads_anchor
    new_export = old_export.parent / "accepted_sync_anchors (1).jsonl"
    write_jsonl(old_export, [{"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 12}])
    write_jsonl(new_export, [
        {"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 12},
        {"frame_id": 10, "cam_id": 1, "anchor_status": "accepted", "selected_video_idx": 13},
    ])
    now = time.time()
    os.utime(old_export, (now - 10, now - 10))
    os.utime(new_export, (now, now))

    report = module.build_report(a)

    assert report["downloads"]["path"] == str(new_export)
    assert report["downloads"]["requested_path"] == str(old_export)
    assert report["downloads"]["accepted_count"] == 2
    assert report["next_action"]["status"] == "stage_latest_anchors"


def test_status_reports_failed_readiness(tmp_path: Path):
    module = load_module()
    a = args(tmp_path)
    write_jsonl(a.staged_anchor, [{"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 12}])
    write_json(a.anchor_validation, {"passed": True})
    a.readiness_exit.parent.mkdir(parents=True, exist_ok=True)
    a.readiness_exit.write_text("3\n", encoding="utf-8")
    write_json(a.readiness_json, {"passed": False, "errors": ["bad"]})

    report = module.build_report(a)

    assert report["next_action"]["status"] == "sync_readiness_failed"
    assert report["readiness_exit"]["value"] == "3"


def test_status_ready_for_dataset_build_after_sync_gate(tmp_path: Path):
    module = load_module()
    a = args(tmp_path)
    write_jsonl(a.staged_anchor, [{"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 12}])
    write_json(a.anchor_validation, {"passed": True})
    a.readiness_exit.parent.mkdir(parents=True, exist_ok=True)
    a.readiness_exit.write_text("0\n", encoding="utf-8")
    write_json(a.readiness_json, {"passed": True})
    write_jsonl(a.expanded_frame_map, [{"frame_id": 10, "cam_id": 0, "video_idx": 12}])
    write_json(a.expanded_frame_map_report, {"clipped_count": 0})
    write_json(a.preflight, {"passed": True})

    report = module.build_report(a)
    markdown = module.render_markdown(report)

    assert report["next_action"]["status"] == "ready_to_build_sync_gated_dataset"
    assert "RUN=1 scripts/run_rtx5070_sync_gated_parking_dataset.sh" in report["next_action"]["command"]
    assert "expanded frame-map rows" in markdown
