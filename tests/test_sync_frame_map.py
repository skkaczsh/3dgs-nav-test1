import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sync_frame_map.py"
    spec = importlib.util.spec_from_file_location("sync_frame_map", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_frame_map_accepts_solver_and_review_rows(tmp_path: Path):
    module = load_module()
    path = tmp_path / "map.jsonl"
    write_jsonl(
        path,
        [
            {"frame_id": 10, "cam_id": 0, "video_idx": 12},
            {"frame_id": 10, "cam_id": 1, "selected_video_idx": 13, "anchor_status": "accepted"},
            {
                "frame_id": 10,
                "cam_id": 2,
                "selected_option_idx": 1,
                "anchor_status": "accepted",
                "options": [
                    {"option_idx": 0, "video_idx": 9},
                    {"option_idx": 1, "video_idx": 14},
                ],
            },
            {"frame_id": 20, "cam_id": 0, "selected_video_idx": 21, "anchor_status": "rejected"},
        ],
    )

    frame_map = module.load_frame_map(path)

    assert frame_map == {(10, 0): 12, (10, 1): 13, (10, 2): 14}
    assert module.resolve_video_idx(frame_map, 10, 2) == 14
    assert module.resolve_video_idx(frame_map, 99, 2) == 99
    assert module.resolve_video_idx(frame_map, 99, 2, fallback_to_direct=False) is None


def test_load_frame_map_rejects_conflicting_rows(tmp_path: Path):
    module = load_module()
    path = tmp_path / "conflict.jsonl"
    write_jsonl(
        path,
        [
            {"frame_id": 10, "cam_id": 0, "video_idx": 12},
            {"frame_id": 10, "cam_id": 0, "video_idx": 13},
        ],
    )

    with pytest.raises(ValueError, match="conflicting video frame"):
        module.load_frame_map(path)


def test_load_frame_map_reports_missing_selection(tmp_path: Path):
    module = load_module()
    path = tmp_path / "missing.jsonl"
    write_jsonl(path, [{"frame_id": 10, "cam_id": 0, "anchor_status": "accepted"}])

    with pytest.raises(ValueError, match="missing selected_video_idx"):
        module.load_frame_map(path)
