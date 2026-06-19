from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_sync_option_sources.py"
    spec = importlib.util.spec_from_file_location("summarize_sync_option_sources", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def candidate(frame: int, cam: int, video: int, offset: int, score: float, sky: float = 0.0):
    return {
        "frame_id": frame,
        "cam_id": cam,
        "video_idx": video,
        "offset": offset,
        "score": score,
        "raw_score": score + sky,
        "sky_hit": sky,
    }


def test_source_records_include_direct_independent_and_smooth():
    module = load_module()
    candidates = [
        candidate(10, 0, 10, 0, 0.5),
        candidate(10, 0, 12, 2, 0.8, 0.1),
    ]
    smooth = [candidate(10, 0, 11, 1, 0.7, 0.2)]

    records = module.source_records(candidates, smooth)

    assert [row["source"] for row in records] == ["direct", "independent_best", "smooth_path"]
    assert abs(records[0]["score_loss_from_independent_best"] - 0.3) < 1e-9
    assert abs(records[2]["score_loss_from_independent_best"] - 0.1) < 1e-9


def test_smooth_temporal_summary_reports_step_ratio():
    module = load_module()
    smooth = [
        candidate(10, 0, 8, -2, 0.7),
        candidate(20, 0, 18, -2, 0.7),
        candidate(30, 0, 28, -2, 0.7),
    ]

    report = module.smooth_temporal_summary(smooth)

    assert report["0"]["step_ratio"]["mean"] == 1.0
    assert report["0"]["frame_to_video"][0]["offset"] == -2


def test_build_report_writes_expected_source_metrics(tmp_path: Path):
    module = load_module()
    candidates_path = tmp_path / "candidates.jsonl"
    smooth_path = tmp_path / "smooth.jsonl"
    write_jsonl(candidates_path, [
        candidate(10, 0, 10, 0, 0.5),
        candidate(10, 0, 12, 2, 0.8, 0.1),
    ])
    write_jsonl(smooth_path, [candidate(10, 0, 11, 1, 0.7, 0.2)])

    report = module.build_report(argparse.Namespace(
        candidates_jsonl=candidates_path,
        smooth_jsonl=smooth_path,
        max_risks=10,
    ))

    assert report["candidate_count"] == 2
    assert report["smooth_count"] == 1
    assert report["by_source"]["direct"]["count"] == 1
    assert report["by_source"]["independent_best"]["sky_hit"]["mean"] == 0.1
    assert report["by_source"]["smooth_path"]["sky_hit"]["mean"] == 0.2
