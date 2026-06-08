#!/usr/bin/env python3
"""Summarize new_route color/semantic QA artifacts."""

import argparse
import json
from pathlib import Path
from statistics import mean


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                return None
    return None


def summarize_semantic(report_path: Path) -> dict:
    report = load_json(report_path)
    if not report:
        return {"available": False}
    combos = report.get("combos", {})
    return {
        "available": True,
        "total_rows": report.get("total_rows"),
        "combos": {
            name: {
                "images": row.get("images"),
                "blocked_images": row.get("blocked_images"),
                "avg_mask_count": row.get("avg_mask_count"),
                "avg_coverage": row.get("avg_coverage"),
                "avg_coverage_with_sky": row.get("avg_coverage_with_sky"),
                "vlm_parse_success_rate": row.get("vlm_parse_success_rate"),
                "blockers": row.get("blockers", []),
            }
            for name, row in combos.items()
        },
    }


def summarize_semantic_projection(projection_dir: Path) -> dict:
    report = load_json(projection_dir / "semantic_projection_report.json")
    if not report:
        return {"available": False}
    summary = report.get("summary", {})
    merged = report.get("merged", {})
    merged_path = Path(merged.get("output", "")) if merged.get("output") else None
    if merged_path and not merged_path.exists():
        local_merged_path = projection_dir / merged_path.name
        if local_merged_path.exists():
            merged_path = local_merged_path
    return {
        "available": True,
        "projection_dir": str(projection_dir),
        "combo": report.get("combo"),
        "frame_count": summary.get("frame_count"),
        "ok_count": summary.get("ok_count"),
        "avg_labeled_ratio": summary.get("avg_labeled_ratio"),
        "total_labeled_points": summary.get("total_labeled_points"),
        "total_points": summary.get("total_points"),
        "merged_available": merged.get("available", False),
        "merged_output": merged.get("output"),
        "merged_points": merged.get("points"),
        "merged_labeled_ratio": merged.get("labeled_ratio"),
        "merged_vertices": ply_vertex_count(merged_path) if merged_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, default=Path("/root/epfs/new_route_stage1_skymask"))
    parser.add_argument("--semantic-dir", type=Path, default=None)
    parser.add_argument("--semantic-projection-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stage = args.stage_dir
    stats = load_json(stage / "skymask_0000_0500_stats.json") or {}
    missing = load_json(stage / "skymask_missing_0000_0500.json") or {}

    novoxel = stage / "merged_0000_0500_skymask_novoxel.ply"
    fast = stage / "merged_0000_0500_skymask_v004_fast.ply"

    semantic_dir = args.semantic_dir or (stage / "semantic_eval_0000_0500")
    semantic = summarize_semantic(semantic_dir / "report.json")
    semantic_projection_dir = args.semantic_projection_dir or (stage / "semantic_projection_0000_0050")
    semantic_projection = summarize_semantic_projection(semantic_projection_dir)

    frame_ratios = []
    if "frame_samples" in stats:
        frame_ratios = [x.get("colored_ratio") for x in stats["frame_samples"] if x.get("colored_ratio") is not None]

    summary = {
        "stage_dir": str(stage),
        "skymask": {
            "match_ratio": stats.get("sky_mask_matches", {}).get("ratio"),
            "matched": stats.get("sky_mask_matches", {}).get("matched"),
            "total_camera_frames": stats.get("sky_mask_matches", {}).get("total_camera_frames"),
            "missing_total": missing.get("missing_total"),
            "missing_by_cam": missing.get("missing_by_cam"),
            "runs_by_cam": missing.get("runs_by_cam"),
        },
        "colorize": {
            "frame_count": stats.get("frames", {}).get("count"),
            "colored_ratio_min": stats.get("frames", {}).get("colored_ratio_min"),
            "colored_ratio_mean": stats.get("frames", {}).get("colored_ratio_mean"),
            "colored_ratio_max": stats.get("frames", {}).get("colored_ratio_max"),
            "sample_ratio_mean": mean(frame_ratios) if frame_ratios else None,
            "merged_points_reported": stats.get("merged", {}).get("points"),
            "merged_colored_ratio": stats.get("merged", {}).get("colored_ratio"),
            "bbox_min": stats.get("merged", {}).get("bbox_min"),
            "bbox_max": stats.get("merged", {}).get("bbox_max"),
        },
        "voxel": {
            "novoxel_path": str(novoxel),
            "novoxel_vertices": ply_vertex_count(novoxel),
            "fast_v004_path": str(fast),
            "fast_v004_vertices": ply_vertex_count(fast),
        },
        "semantic": semantic,
        "semantic_projection": semantic_projection,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
