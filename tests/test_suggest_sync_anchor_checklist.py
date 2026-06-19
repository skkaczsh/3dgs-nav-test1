from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "suggest_sync_anchor_checklist.py"
    spec = importlib.util.spec_from_file_location("suggest_sync_anchor_checklist", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(frame: int, cam: int, score: float, source: str = "independent_best", selected: bool = False) -> dict:
    option = {
        "option_idx": 7,
        "review_source": source,
        "video_idx": frame + 12 + cam,
        "score": score,
        "panel_path": f"panels/f{frame}_c{cam}.jpg",
    }
    return {
        "frame_id": frame,
        "cam_id": cam,
        "anchor_status": "unreviewed",
        "selected_option_idx": 7 if selected else None,
        "selected_video_idx": option["video_idx"] if selected else None,
        "options": [option],
        "score_margin": 0.1,
        "priority_score": score,
        "risk_reasons": [],
    }


def test_selects_temporally_spread_rows_per_camera(tmp_path: Path):
    module = load_module()
    review = tmp_path / "review.jsonl"
    rows = []
    for cam in [0, 1]:
        rows.extend([
            row(1000, cam, 0.9),
            row(2000, cam, 0.4),
            row(3000, cam, 0.8),
            row(4000, cam, 0.7),
        ])
    write_jsonl(review, rows)
    out_jsonl = tmp_path / "checklist.jsonl"
    out_md = tmp_path / "checklist.md"

    report = module.build(argparse.Namespace(
        review_jsonl=review,
        output_jsonl=out_jsonl,
        output_md=out_md,
        per_cam=3,
        bins=3,
        review_url="http://review",
    ))
    selected = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]

    assert report["selected_by_cam"] == {"0": 3, "1": 3}
    assert {item["cam_id"] for item in selected} == {0, 1}
    assert {item["frame_id"] for item in selected if item["cam_id"] == 0} == {1000, 2000, 3000}
    assert selected[0]["recommended_video_idx"] == 1012
    assert "manual review aid" in out_md.read_text(encoding="utf-8")


def test_prefers_selected_smooth_path_over_unselected_best_option():
    module = load_module()
    selected_row = row(1000, 0, 0.5, source="smooth_path", selected=True)
    best_row = row(1000, 0, 0.7, source="independent_best", selected=False)

    assert module.recommended_option(selected_row)["review_source"] == "smooth_path"
    assert module.checklist_score(selected_row) > module.checklist_score(best_row) - 0.3
