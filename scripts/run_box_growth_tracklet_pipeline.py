#!/usr/bin/env python3
"""Run fine-object box-growth consolidation as one reproducible pipeline.

Stages:
1. build frame-level fine targets from enriched PLY
2. merge targets into short tracklets
3. associate tracklets into long objects
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(args: list[str]) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colored-frame-dir", type=Path, required=True)

    parser.add_argument("--target-voxel-size", type=float, default=0.08)
    parser.add_argument("--min-target-points", type=int, default=5)

    parser.add_argument("--tracklet-max-frame-gap", type=int, default=15)
    parser.add_argument("--tracklet-centroid-distance", type=float, default=0.35)
    parser.add_argument("--tracklet-bbox-distance", type=float, default=0.08)
    parser.add_argument("--tracklet-color-distance", type=float, default=45.0)
    parser.add_argument("--tracklet-normal-angle", type=float, default=180.0)

    parser.add_argument("--same-candidate-centroid-distance", type=float, default=1.5)
    parser.add_argument("--same-candidate-bbox-distance", type=float, default=0.5)
    parser.add_argument("--same-candidate-color-distance", type=float, default=90.0)
    parser.add_argument("--source-frame-gap", type=int, default=240)
    parser.add_argument("--source-centroid-distance", type=float, default=0.8)
    parser.add_argument("--source-bbox-distance", type=float, default=0.25)
    parser.add_argument("--source-color-distance", type=float, default=60.0)
    parser.add_argument("--cross-frame-gap", type=int, default=80)
    parser.add_argument("--cross-centroid-distance", type=float, default=0.35)
    parser.add_argument("--cross-bbox-distance", type=float, default=0.08)
    parser.add_argument("--cross-color-distance", type=float, default=35.0)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_targets_dir = args.output_dir / "frame_targets"
    tracklets_dir = args.output_dir / "tracklets"
    long_assoc_dir = args.output_dir / "long_assoc"

    py = sys.executable

    run_step(
        [
            py,
            str(ROOT / "build_frame_fine_targets_from_enriched.py"),
            "--enriched-ply",
            str(args.enriched_ply),
            "--output-dir",
            str(frame_targets_dir),
            "--colored-frame-dir",
            str(args.colored_frame_dir),
            "--voxel-size",
            str(args.target_voxel_size),
            "--min-target-points",
            str(args.min_target_points),
            "--write-ply",
        ]
    )
    run_step(
        [
            py,
            str(ROOT / "build_tracklets_from_frame_targets.py"),
            "--targets",
            str(frame_targets_dir),
            "--output-dir",
            str(tracklets_dir),
            "--max-frame-gap",
            str(args.tracklet_max_frame_gap),
            "--centroid-distance",
            str(args.tracklet_centroid_distance),
            "--bbox-distance",
            str(args.tracklet_bbox_distance),
            "--color-distance",
            str(args.tracklet_color_distance),
            "--normal-angle",
            str(args.tracklet_normal_angle),
        ]
    )
    run_step(
        [
            py,
            str(ROOT / "associate_tracklets_long_range.py"),
            "--tracklets",
            str(tracklets_dir),
            "--output-dir",
            str(long_assoc_dir),
            "--same-candidate-centroid-distance",
            str(args.same_candidate_centroid_distance),
            "--same-candidate-bbox-distance",
            str(args.same_candidate_bbox_distance),
            "--same-candidate-color-distance",
            str(args.same_candidate_color_distance),
            "--source-frame-gap",
            str(args.source_frame_gap),
            "--source-centroid-distance",
            str(args.source_centroid_distance),
            "--source-bbox-distance",
            str(args.source_bbox_distance),
            "--source-color-distance",
            str(args.source_color_distance),
            "--cross-frame-gap",
            str(args.cross_frame_gap),
            "--cross-centroid-distance",
            str(args.cross_centroid_distance),
            "--cross-bbox-distance",
            str(args.cross_bbox_distance),
            "--cross-color-distance",
            str(args.cross_color_distance),
        ]
    )

    report = {
        "enriched_ply": str(args.enriched_ply),
        "output_dir": str(args.output_dir),
        "frame_targets_report": str(frame_targets_dir / "frame_fine_targets_report.json"),
        "tracklet_report": str(tracklets_dir / "tracklet_report.json"),
        "long_association_report": str(long_assoc_dir / "long_association_report.json"),
        "params": {
            "target_voxel_size": args.target_voxel_size,
            "min_target_points": args.min_target_points,
            "tracklet_max_frame_gap": args.tracklet_max_frame_gap,
            "tracklet_centroid_distance": args.tracklet_centroid_distance,
            "tracklet_bbox_distance": args.tracklet_bbox_distance,
            "tracklet_color_distance": args.tracklet_color_distance,
            "tracklet_normal_angle": args.tracklet_normal_angle,
            "same_candidate_centroid_distance": args.same_candidate_centroid_distance,
            "same_candidate_bbox_distance": args.same_candidate_bbox_distance,
            "same_candidate_color_distance": args.same_candidate_color_distance,
            "source_frame_gap": args.source_frame_gap,
            "source_centroid_distance": args.source_centroid_distance,
            "source_bbox_distance": args.source_bbox_distance,
            "source_color_distance": args.source_color_distance,
            "cross_frame_gap": args.cross_frame_gap,
            "cross_centroid_distance": args.cross_centroid_distance,
            "cross_bbox_distance": args.cross_bbox_distance,
            "cross_color_distance": args.cross_color_distance,
        },
    }
    (args.output_dir / "pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
