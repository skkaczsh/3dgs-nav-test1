from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "expand_sync_frame_map.py"
    spec = importlib.util.spec_from_file_location("expand_sync_frame_map", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_args(tmp_path: Path, status: str = "accepted") -> argparse.Namespace:
    path_jsonl = tmp_path / "path.jsonl"
    solver_report = tmp_path / "report.json"
    img_pos = tmp_path / "img_pos.txt"
    write_jsonl(path_jsonl, [
        {"frame_id": 10, "cam_id": 0, "video_idx": 100, "sync_timestamp": 1.0, "cam_path_status": "accepted"},
        {"frame_id": 20, "cam_id": 0, "video_idx": 160, "sync_timestamp": 2.0, "cam_path_status": "accepted"},
    ])
    solver_report.write_text(json.dumps({
        "status": status,
        "video_fps": 6.0,
        "cam_reports": {"0": {"accepted": status == "accepted", "absolute_intercept": 100.0}},
    }), encoding="utf-8")
    img_pos.write_text(
        "10 1.0 0 0 0\n"
        "20 2.0 0 0 0\n"
        "30 3.0 0 0 0\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        path_jsonl=path_jsonl,
        solver_report=solver_report,
        img_pos_file=img_pos,
        output_jsonl=tmp_path / "expanded.jsonl",
        report=tmp_path / "expanded_report.json",
        start=10,
        end=30,
        stride=10,
        cams=[0],
        video_frame_count=200,
        timestamp_phase_fraction=None,
        allow_rejected_solver=False,
    )


def test_expands_accepted_timestamp_model(tmp_path: Path):
    module = load_module()
    args = make_args(tmp_path)

    rows, report = module.build_rows(args)

    assert report["row_count"] == 3
    assert [row["video_idx"] for row in rows] == [100, 106, 112]
    assert all(row["cam_path_status"] == "accepted" for row in rows)
    assert rows[0]["source"] == "expanded_timestamp_absprior"


def test_expansion_reuses_solver_timestamp_phase(tmp_path: Path):
    module = load_module()
    args = make_args(tmp_path)
    args.solver_report.write_text(json.dumps({
        "status": "accepted",
        "video_fps": 6.0,
        "timestamp_phase_fraction": 0.5,
        "cam_reports": {"0": {"accepted": True, "absolute_intercept": 100.0}},
    }), encoding="utf-8")
    write_jsonl(args.path_jsonl, [
        {"frame_id": 10, "cam_id": 0, "video_idx": 100, "sync_timestamp": 1.5, "cam_path_status": "accepted"},
        {"frame_id": 20, "cam_id": 0, "video_idx": 109, "sync_timestamp": 3.0, "cam_path_status": "accepted"},
    ])
    args.img_pos_file.write_text(
        "10 1.0 0 0 0\n"
        "20 2.0 0 0 0\n"
        "30 4.0 0 0 0\n",
        encoding="utf-8",
    )

    rows, report = module.build_rows(args)

    assert report["timestamp_phase_fraction"] == 0.5
    assert [row["video_idx"] for row in rows] == [100, 109, 121]
    assert rows[0]["raw_sync_timestamp"] == 1.0
    assert rows[0]["sync_timestamp"] == 1.5


def test_rejects_unaccepted_solver_by_default(tmp_path: Path):
    module = load_module()
    args = make_args(tmp_path, status="rejected")

    with pytest.raises(ValueError, match="not accepted"):
        module.build_rows(args)


def test_can_expand_rejected_solver_for_diagnostics(tmp_path: Path):
    module = load_module()
    args = make_args(tmp_path, status="rejected")
    args.allow_rejected_solver = True

    rows, report = module.build_rows(args)

    assert report["row_count"] == 3
    assert [row["video_idx"] for row in rows] == [100, 106, 112]
