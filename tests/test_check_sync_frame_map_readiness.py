import argparse
import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_sync_frame_map_readiness.py"
    spec = importlib.util.spec_from_file_location("check_sync_frame_map_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def args(**kwargs):
    base = dict(
        anchors_jsonl=None,
        frame_map_jsonl=None,
        solver_report=None,
        frames=None,
        start=10,
        end=20,
        stride=10,
        cams=[0, 1],
        min_accepted_per_cam=1,
        allow_rejected=False,
        output=None,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_readiness_fails_when_anchors_are_unreviewed(tmp_path: Path):
    module = load_module()
    anchors = tmp_path / "anchors.jsonl"
    write_jsonl(
        anchors,
        [
            {"frame_id": 10, "cam_id": 0, "anchor_status": "unreviewed"},
            {"frame_id": 10, "cam_id": 1, "anchor_status": "unreviewed"},
        ],
    )

    report = module.build_report(args(anchors_jsonl=anchors))

    assert report["passed"] is False
    assert "accepted_anchors_cam0=0<min1" in report["errors"]
    assert "accepted_anchors_cam1=0<min1" in report["errors"]


def test_readiness_passes_for_accepted_solver_and_full_frame_map(tmp_path: Path):
    module = load_module()
    anchors = tmp_path / "anchors.jsonl"
    frame_map = tmp_path / "frame_map.jsonl"
    solver = tmp_path / "solver.json"
    write_jsonl(
        anchors,
        [
            {"frame_id": 10, "cam_id": 0, "anchor_status": "accepted", "selected_video_idx": 12},
            {"frame_id": 10, "cam_id": 1, "anchor_status": "accepted", "selected_video_idx": 13},
        ],
    )
    write_jsonl(
        frame_map,
        [
            {"frame_id": 10, "cam_id": 0, "video_idx": 12, "cam_path_status": "accepted"},
            {"frame_id": 10, "cam_id": 1, "video_idx": 13, "cam_path_status": "accepted"},
            {"frame_id": 20, "cam_id": 0, "video_idx": 22, "cam_path_status": "accepted"},
            {"frame_id": 20, "cam_id": 1, "video_idx": 23, "cam_path_status": "accepted"},
        ],
    )
    solver.write_text(
        json.dumps(
            {
                "status": "accepted",
                "accepted_anchor_count": 2,
                "cam_reports": {
                    "0": {"accepted": True, "status": "accepted"},
                    "1": {"accepted": True, "status": "accepted"},
                },
            }
        ),
        encoding="utf-8",
    )

    report = module.build_report(args(anchors_jsonl=anchors, frame_map_jsonl=frame_map, solver_report=solver))

    assert report["passed"] is True
    assert report["checks"]["frame_map"]["loaded_pairs"] == 4
    assert report["checks"]["frame_map"]["mapped_non_direct_pairs"] == 4


def test_readiness_fails_for_rejected_solver_or_unsafe_map(tmp_path: Path):
    module = load_module()
    frame_map = tmp_path / "bad_map.jsonl"
    solver = tmp_path / "solver.json"
    write_jsonl(
        frame_map,
        [{"frame_id": 10, "cam_id": 0, "video_idx": 12, "cam_path_status": "rejected_unstable_temporal_path"}],
    )
    solver.write_text(
        json.dumps(
            {
                "status": "rejected",
                "accepted_anchor_count": 0,
                "cam_reports": {"0": {"accepted": False, "status": "rejected_unstable_temporal_path"}},
            }
        ),
        encoding="utf-8",
    )

    report = module.build_report(args(frame_map_jsonl=frame_map, solver_report=solver, cams=[0], start=10, end=10))

    assert report["passed"] is False
    assert "solver_report_status=rejected" in report["errors"]
    assert any("unsafe sync row status" in item for item in report["errors"])


def test_readiness_reports_missing_files_without_traceback(tmp_path: Path):
    module = load_module()

    report = module.build_report(
        args(
            anchors_jsonl=tmp_path / "missing_anchors.jsonl",
            frame_map_jsonl=tmp_path / "missing_map.jsonl",
            solver_report=tmp_path / "missing_solver.json",
        )
    )

    assert report["passed"] is False
    assert any(item.startswith("anchors_missing=") for item in report["errors"])
    assert any(item.startswith("frame_map_missing=") for item in report["errors"])
    assert any(item.startswith("solver_report_missing=") for item in report["errors"])
