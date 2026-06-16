#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path("/Users/skkac/Work/SCAN")
EXPERIMENT_DIR = ROOT / "new_route/experiments/fine_object_grounded_small_eval"
REMOTE = "root@10.0.8.114"
REMOTE_PORT = "31909"
REMOTE_PROJECT = "/root/epfs/vlm_seg_project"
REMOTE_PYTHON = "/root/epfs/conda_envs/conceptseg-r1/bin/python"
REMOTE_SCRIPT_DIR = "/root/epfs/new_route_scripts"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


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
    parser.add_argument("--sync-local-dir", type=Path, default=None)
    args = parser.parse_args()

    local_manifest = args.manifest.resolve()
    remote_output = args.remote_output_dir.rstrip("/")
    remote_manifest = f"{remote_output}/manifest.json"
    remote_outputs = f"{remote_output}/outputs"
    remote_filtered = f"{remote_output}/filtered"
    remote_projected = f"{remote_output}/projected"
    remote_fused = f"{remote_output}/fused"

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
  --output-dir {shlex.quote(remote_filtered)}
{shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SCRIPT_DIR)}/project_filtered_grouped_masks.py \
  --accepted-jsonl {shlex.quote(remote_filtered)}/accepted_detections.jsonl \
  --script-dir {shlex.quote(REMOTE_SCRIPT_DIR)} \
  --color-dir /root/epfs/new_route_stage1_skymask/output \
  --output-dir {shlex.quote(remote_projected)}
if [ -f {shlex.quote(remote_projected)}/accepted_points.ply ]; then
  {shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SCRIPT_DIR)}/fuse_accepted_fine_objects.py \
    --accepted-report-json {shlex.quote(remote_projected)}/accepted_report.json \
    --strict-filtered-ply {shlex.quote(remote_projected)}/accepted_points.ply \
    --output-objects-jsonl {shlex.quote(remote_fused)}/fine_objects.jsonl \
    --output-decisions-jsonl {shlex.quote(remote_fused)}/fine_object_decisions.jsonl \
    --output-report {shlex.quote(remote_fused)}/fine_object_report.json \
    --output-ply {shlex.quote(remote_fused)}/fine_object_points.ply
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
        for rel in [
            "outputs/summary.json",
            "filtered/filter_summary.json",
            "projected/projection_report.json",
            "projected/accepted_report.json",
            "fused/fine_object_report.json",
        ]:
            local_target = args.sync_local_dir / Path(rel).name
            run(
                [
                    "scp",
                    "-F",
                    "/dev/null",
                    "-P",
                    REMOTE_PORT,
                    f"{REMOTE}:{remote_output}/{rel}",
                    str(local_target),
                ]
            )
    print(remote_output)


if __name__ == "__main__":
    main()
