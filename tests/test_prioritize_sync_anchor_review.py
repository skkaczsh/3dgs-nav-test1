import argparse
import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prioritize_sync_anchor_review.py"
    spec = importlib.util.spec_from_file_location("prioritize_sync_anchor_review", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(frame, cam, scores):
    options = []
    for idx, score in enumerate(scores):
        options.append(
            {
                "option_idx": idx,
                "review_source": "direct" if idx == 0 else "top_candidate",
                "video_idx": frame + idx * 10,
                "offset": idx * 10,
                "score": score,
                "edge_hit": score,
                "edge_distance_mean": 1.0 + idx,
                "panel_path": f"panels/f{frame}_c{cam}_o{idx}.jpg",
            }
        )
    return {
        "frame_id": frame,
        "cam_id": cam,
        "anchor_status": "unreviewed",
        "selected_video_idx": None,
        "selected_option_idx": None,
        "notes": "",
        "options": options,
    }


def row_with_sources(frame, cam):
    return {
        "frame_id": frame,
        "cam_id": cam,
        "anchor_status": "unreviewed",
        "selected_video_idx": None,
        "selected_option_idx": None,
        "notes": "",
        "options": [
            {"option_idx": 0, "review_source": "direct", "video_idx": frame, "offset": 0, "score": 0.5},
            {"option_idx": 1, "review_source": "smooth_path", "video_idx": frame - 3, "offset": -3, "score": 0.4},
        ],
    }


def test_prioritizer_selects_top_rows_per_camera(tmp_path: Path):
    module = load_module()
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            row(10, 0, [0.50, 0.49]),
            row(20, 0, [0.90, 0.50]),
            row(10, 1, [0.60, 0.55]),
            row(20, 1, [0.95, 0.40]),
        ],
    )

    report = module.build(
        argparse.Namespace(
            manifest=manifest,
            output_dir=tmp_path / "out",
            source_dir=tmp_path,
            per_cam=1,
            preselect_source=None,
        )
    )

    selected = module.read_jsonl(tmp_path / "out" / "anchor_review_priority_batch.jsonl")
    assert report["selected_by_cam"] == {"0": 1, "1": 1}
    assert {(row["frame_id"], row["cam_id"]) for row in selected} == {(20, 0), (20, 1)}
    html = (tmp_path / "out" / "anchor_review_priority.html").read_text(encoding="utf-8")
    assert "Prioritized Sync Anchor Review" in html
    assert "accepted_sync_anchors.jsonl" in html
    assert "Mark selected accepted" in html
    assert "export readiness" in html
    assert "cam${cam} accepted" in html
    assert "sequence issues" in html
    assert "Anchor coverage/sequence is not ready" in html
    assert "panels/f20_c0_o0.jpg" in html


def test_enrich_row_flags_low_margin_and_large_offset():
    module = load_module()
    enriched = module.enrich_row(
        {
            "frame_id": 100,
            "cam_id": 0,
            "options": [
                {"option_idx": 0, "review_source": "direct", "video_idx": 100, "offset": 0, "score": 0.50},
                {"option_idx": 1, "review_source": "top_candidate", "video_idx": 900, "offset": 800, "score": 0.52},
            ],
        }
    )

    assert enriched["best_option_idx"] == 1
    assert "low_score_margin" in enriched["risk_reasons"]
    assert "large_best_offset" in enriched["risk_reasons"]


def test_preselect_source_sets_selected_option_without_accepting(tmp_path: Path):
    module = load_module()
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [row_with_sources(10, 0)])

    report = module.build(
        argparse.Namespace(
            manifest=manifest,
            output_dir=tmp_path / "out",
            source_dir=tmp_path,
            per_cam=1,
            preselect_source="smooth_path",
        )
    )

    selected = module.read_jsonl(tmp_path / "out" / "anchor_review_priority_batch.jsonl")
    assert report["preselected_count"] == 1
    assert selected[0]["selected_option_idx"] == 1
    assert selected[0]["selected_video_idx"] == 7
    assert selected[0]["anchor_status"] == "unreviewed"
