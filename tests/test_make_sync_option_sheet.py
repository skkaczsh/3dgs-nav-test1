from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_sync_option_sheet.py"
    spec = importlib.util.spec_from_file_location("make_sync_option_sheet", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_choose_option_by_review_source():
    module = load_module()
    row = {
        "options": [
            {"review_source": "direct", "video_idx": 10},
            {"review_source": "smooth_path", "video_idx": 12},
        ]
    }

    assert module.choose_option(row, "smooth_path")["video_idx"] == 12
    assert module.choose_option(row, "missing") is None


def test_build_sheet_renders_selected_option(tmp_path: Path):
    module = load_module()
    source_dir = tmp_path / "review"
    panel = source_dir / "panels" / "a.jpg"
    panel.parent.mkdir(parents=True)
    cv2.imwrite(str(panel), np.full((20, 30, 3), 180, dtype=np.uint8))
    manifest = source_dir / "manual_anchor_manifest.jsonl"
    write_jsonl(manifest, [
        {
            "frame_id": 10,
            "cam_id": 0,
            "options": [
                {
                    "option_idx": 2,
                    "review_source": "smooth_path",
                    "video_idx": 12,
                    "offset": 2,
                    "score": 0.7,
                    "panel_path": "panels/a.jpg",
                }
            ],
        }
    ])
    output = tmp_path / "sheet.jpg"

    report = module.build_sheet(manifest, source_dir, "smooth_path", output, cols=1, thumb_width=60)

    assert output.exists()
    assert report["selected_count"] == 1
    assert report["rendered_count"] == 1
    assert report["missing_count"] == 0
    assert report["selected"][0]["video_idx"] == 12


def test_build_sheet_reports_missing_option_source(tmp_path: Path):
    module = load_module()
    source_dir = tmp_path / "review"
    manifest = source_dir / "manual_anchor_manifest.jsonl"
    write_jsonl(manifest, [{"frame_id": 10, "cam_id": 0, "options": []}])

    report = module.build_sheet(manifest, source_dir, "smooth_path", tmp_path / "sheet.jpg", cols=1, thumb_width=60)

    assert report["selected_count"] == 0
    assert report["missing_count"] == 1
    assert "missing_option_source=smooth_path" in report["missing"][0]["reason"]
