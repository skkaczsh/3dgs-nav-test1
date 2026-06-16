#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


ROOT = Path("/Users/skkac/Work/SCAN")
REMOTE = "root@10.0.8.114"
REMOTE_PORT = "31909"
REMOTE_PROJECT = "/root/epfs/vlm_seg_project"
REMOTE_PYTHON = "/root/epfs/conda_envs/conceptseg-r1/bin/python"
REMOTE_SCRIPT_DIR = "/root/epfs/new_route_scripts"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-output-dir", required=True)
    parser.add_argument("--grounding-device", default="cpu")
    parser.add_argument("--sam-device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--max-boxes", type=int, default=12)
    parser.add_argument("--max-boxes-per-group", type=int, default=4)
    parser.add_argument("--equipment-filter-mode", choices=["default", "strict_precision"], default="default")
    parser.add_argument("--box-grow-expand-px", type=int, default=10)
    parser.add_argument("--box-grow-mask-distance-px", type=float, default=12.0)
    parser.add_argument("--box-grow-depth-slack", type=float, default=0.30)
    parser.add_argument("--box-grow-voxel-size", type=float, default=0.08)
    parser.add_argument("--box-grow-min-component-points", type=int, default=8)
    parser.add_argument("--target-voxel-size", type=float, default=0.08)
    parser.add_argument("--min-target-points", type=int, default=5)
    parser.add_argument("--tracklet-max-frame-gap", type=int, default=15)
    parser.add_argument("--tracklet-centroid-distance", type=float, default=0.35)
    parser.add_argument("--tracklet-bbox-distance", type=float, default=0.08)
    parser.add_argument("--tracklet-color-distance", type=float, default=45.0)
    parser.add_argument("--tracklet-normal-angle", type=float, default=180.0)
    parser.add_argument("--same-candidate-centroid-distance", type=float, default=3.5)
    parser.add_argument("--same-candidate-bbox-distance", type=float, default=1.5)
    parser.add_argument("--same-candidate-color-distance", type=float, default=120.0)
    parser.add_argument("--source-frame-gap", type=int, default=240)
    parser.add_argument("--source-centroid-distance", type=float, default=0.8)
    parser.add_argument("--source-bbox-distance", type=float, default=0.25)
    parser.add_argument("--source-color-distance", type=float, default=60.0)
    parser.add_argument("--cross-frame-gap", type=int, default=80)
    parser.add_argument("--cross-centroid-distance", type=float, default=0.35)
    parser.add_argument("--cross-bbox-distance", type=float, default=0.08)
    parser.add_argument("--cross-color-distance", type=float, default=35.0)
    parser.add_argument("--sync-local-dir", type=Path, default=None)
    args = parser.parse_args()

    local_manifest = args.manifest.resolve()
    remote_output = args.remote_output_dir.rstrip("/")
    remote_manifest = f"{remote_output}/manifest.json"
    remote_outputs = f"{remote_output}/outputs"
    remote_filtered = f"{remote_output}/filtered"
    remote_box_growth = f"{remote_output}/box_growth"
    remote_fused = f"{remote_output}/fused"
    remote_tracklets = f"{remote_output}/tracklet_pipeline"

    run(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-p",
            REMOTE_PORT,
            REMOTE,
            f"mkdir -p {shlex.quote(remote_output)}",
        ]
    )

    run(
        [
            "scp",
            "-F",
            "/dev/null",
            "-P",
            REMOTE_PORT,
            str(local_manifest),
            f"{REMOTE}:{remote_manifest}",
        ]
    )

    remote_cmd = f"""
set -e
mkdir -p {shlex.quote(remote_output)}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
{shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SCRIPT_DIR)}/remote_batch_eval.py \
  --manifest {shlex.quote(remote_manifest)} \
  --project-dir {shlex.quote(REMOTE_PROJECT)} \
  --work-dir {shlex.quote(remote_outputs)} \
  --grounding-device {shlex.quote(args.grounding_device)} \
  --sam-device {shlex.quote(args.sam_device)} \
  --box-threshold {args.box_threshold} \
  --text-threshold {args.text_threshold} \
  --nms-threshold {args.nms_threshold} \
  --max-boxes {args.max_boxes} \
  --max-boxes-per-group {args.max_boxes_per_group}
{shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SCRIPT_DIR)}/filter_grouped_detections.py \
  --summary {shlex.quote(remote_outputs)}/summary.json \
  --output-dir {shlex.quote(remote_filtered)} \
  --equipment-filter-mode {shlex.quote(args.equipment_filter_mode)}
{shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SCRIPT_DIR)}/project_detector_box_growth.py \
  --accepted-jsonl {shlex.quote(remote_filtered)}/accepted_detections.jsonl \
  --script-dir {shlex.quote(REMOTE_SCRIPT_DIR)} \
  --color-dir /root/epfs/new_route_stage1_skymask/output \
  --output-dir {shlex.quote(remote_box_growth)} \
  --box-expand-px {args.box_grow_expand_px} \
  --mask-distance-px {args.box_grow_mask_distance_px} \
  --depth-slack {args.box_grow_depth_slack} \
  --voxel-size {args.box_grow_voxel_size} \
  --min-component-points {args.box_grow_min_component_points}
if [ -f {shlex.quote(remote_box_growth)}/accepted_points.ply ]; then
  {shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SCRIPT_DIR)}/review_accepted_fine_objects.py \
    --accepted-report {shlex.quote(remote_box_growth)}/accepted_report.json \
    --accepted-ply {shlex.quote(remote_box_growth)}/accepted_points.ply \
    --output-report {shlex.quote(remote_box_growth)}/guard_review.json \
    --output-csv {shlex.quote(remote_box_growth)}/guard_review.csv \
    --output-status-ply {shlex.quote(remote_box_growth)}/guard_status.ply \
    --output-filtered-ply {shlex.quote(remote_box_growth)}/guarded_points.ply \
    --output-accepted-report {shlex.quote(remote_box_growth)}/guarded_accepted_report.json
  {shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SCRIPT_DIR)}/fuse_accepted_fine_objects.py \
    --accepted-report-json {shlex.quote(remote_box_growth)}/guarded_accepted_report.json \
    --strict-filtered-ply {shlex.quote(remote_box_growth)}/guarded_points.ply \
    --output-objects-jsonl {shlex.quote(remote_fused)}/fine_objects.jsonl \
    --output-decisions-jsonl {shlex.quote(remote_fused)}/fine_object_decisions.jsonl \
    --output-report {shlex.quote(remote_fused)}/fine_object_report.json \
    --output-ply {shlex.quote(remote_fused)}/fine_object_points.ply
  {shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SCRIPT_DIR)}/run_box_growth_tracklet_pipeline.py \
    --enriched-ply {shlex.quote(remote_box_growth)}/guarded_points.ply \
    --output-dir {shlex.quote(remote_tracklets)} \
    --colored-frame-dir /root/epfs/new_route_stage1_skymask/output \
    --target-voxel-size {args.target_voxel_size} \
    --min-target-points {args.min_target_points} \
    --tracklet-max-frame-gap {args.tracklet_max_frame_gap} \
    --tracklet-centroid-distance {args.tracklet_centroid_distance} \
    --tracklet-bbox-distance {args.tracklet_bbox_distance} \
    --tracklet-color-distance {args.tracklet_color_distance} \
    --tracklet-normal-angle {args.tracklet_normal_angle} \
    --same-candidate-centroid-distance {args.same_candidate_centroid_distance} \
    --same-candidate-bbox-distance {args.same_candidate_bbox_distance} \
    --same-candidate-color-distance {args.same_candidate_color_distance} \
    --source-frame-gap {args.source_frame_gap} \
    --source-centroid-distance {args.source_centroid_distance} \
    --source-bbox-distance {args.source_bbox_distance} \
    --source-color-distance {args.source_color_distance} \
    --cross-frame-gap {args.cross_frame_gap} \
    --cross-centroid-distance {args.cross_centroid_distance} \
    --cross-bbox-distance {args.cross_bbox_distance} \
    --cross-color-distance {args.cross_color_distance}
fi
"""
    run(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-p",
            REMOTE_PORT,
            REMOTE,
            remote_cmd,
        ]
    )

    if args.sync_local_dir:
        args.sync_local_dir.mkdir(parents=True, exist_ok=True)
        sync_files = [
            ("outputs/summary.json", "summary.json"),
            ("filtered/filter_summary.json", "filter_summary.json"),
            ("box_growth/projection_report.json", "projection_report.json"),
            ("box_growth/accepted_report.json", "accepted_report.json"),
            ("box_growth/guard_review.json", "guard_review.json"),
            ("fused/fine_object_report.json", "fine_object_report.json"),
            ("tracklet_pipeline/frame_targets/frame_fine_targets_report.json", "frame_fine_targets_report.json"),
            ("tracklet_pipeline/tracklets/tracklet_report.json", "tracklet_report.json"),
            ("tracklet_pipeline/long_assoc/long_association_report.json", "long_association_report.json"),
            ("tracklet_pipeline/pipeline_report.json", "pipeline_report.json"),
        ]
        for remote_rel, local_name in sync_files:
            run(
                [
                    "scp",
                    "-F",
                    "/dev/null",
                    "-P",
                    REMOTE_PORT,
                    f"{REMOTE}:{remote_output}/{remote_rel}",
                    str(args.sync_local_dir / local_name),
                ]
            )

    print(remote_output)


if __name__ == "__main__":
    main()
