#!/usr/bin/env python3
"""Run surface absorption plus fine residual clustering as one reproducible step.

This is intentionally an orchestrator over the existing diagnostic scripts:

1. Assign surface-compatible residuals to a hybrid surface index.
2. Sweep clustering parameters for the remaining fine-object residuals.
3. Write a fine residual cluster PLY/report for review.

The output is a QA/refinement artifact. It does not rewrite the canonical
target/object JSONL files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-min-object-targets", type=int, default=1)
    parser.add_argument("--surface-min-object-points", type=int, default=100)
    parser.add_argument("--surface-bbox-padding", type=float, default=0.75)
    parser.add_argument("--surface-plane-distance", type=float, default=0.20)
    parser.add_argument("--surface-color-distance", type=float, default=90.0)
    parser.add_argument("--fine-labels", nargs="+", default=["equipment", "railing", "pipe"])
    parser.add_argument("--fine-voxel-size", type=float, default=0.12)
    parser.add_argument("--fine-min-cluster-points", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--write-fine-ply", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assignment_report = args.output_dir / "surface_residual_assignment.json"
    assignment_ply = args.output_dir / "surface_residual_assignment.ply"
    fine_sweep = args.output_dir / "fine_residual_cluster_sweep.json"
    fine_report = args.output_dir / "fine_residual_clusters.json"
    fine_ply = args.output_dir / "fine_residual_clusters.ply"
    summary_path = args.output_dir / "residual_refinement_summary.json"

    run(
        [
            sys.executable,
            str(script_dir / "assign_residuals_to_surface_objects.py"),
            "--residual-dir",
            str(args.residual_dir),
            "--objects-jsonl",
            str(args.objects_jsonl),
            "--output-report",
            str(assignment_report),
            "--output-ply",
            str(assignment_ply),
            "--write-ply",
            "--min-object-targets",
            str(args.surface_min_object_targets),
            "--min-object-points",
            str(args.surface_min_object_points),
            "--bbox-padding",
            str(args.surface_bbox_padding),
            "--max-plane-distance",
            str(args.surface_plane_distance),
            "--max-color-distance",
            str(args.surface_color_distance),
        ]
    )

    sweep_configs = [
        "voxel=0.08,min=30",
        "voxel=0.12,min=50",
        "voxel=0.16,min=80",
    ]
    run(
        [
            sys.executable,
            str(script_dir / "sweep_fine_residual_clustering.py"),
            "--residual-assignment-ply",
            str(assignment_ply),
            "--output",
            str(fine_sweep),
            "--labels",
            *args.fine_labels,
            "--config",
            sweep_configs[0],
            "--config",
            sweep_configs[1],
            "--config",
            sweep_configs[2],
            "--top-n",
            str(min(args.top_n, 30)),
        ]
    )

    cluster_cmd = [
        sys.executable,
        str(script_dir / "cluster_fine_residual_objects.py"),
        "--residual-assignment-ply",
        str(assignment_ply),
        "--output-report",
        str(fine_report),
        "--output-ply",
        str(fine_ply),
        "--labels",
        *args.fine_labels,
        "--voxel-size",
        str(args.fine_voxel_size),
        "--min-cluster-points",
        str(args.fine_min_cluster_points),
        "--top-n",
        str(args.top_n),
    ]
    if args.write_fine_ply:
        cluster_cmd.append("--write-ply")
    run(cluster_cmd)

    assignment = load_json(assignment_report)
    sweep = load_json(fine_sweep)
    fine = load_json(fine_report)
    summary = {
        "residual_dir": str(args.residual_dir),
        "objects_jsonl": str(args.objects_jsonl),
        "output_dir": str(args.output_dir),
        "surface_assignment": {
            "surface_objects": assignment.get("surface_objects"),
            "residual_points": assignment.get("residual_points"),
            "assigned_points": assignment.get("assigned_points"),
            "assigned_ratio": assignment.get("assigned_ratio"),
            "by_label": assignment.get("by_label", {}),
            "assigned_by_label": assignment.get("assigned_by_label", {}),
            "params": assignment.get("params", {}),
        },
        "fine_clustering": {
            "labels": fine.get("labels"),
            "selected_points": fine.get("selected_points"),
            "cluster_count": fine.get("cluster_count"),
            "clustered_points": fine.get("clustered_points"),
            "small_cluster_points": fine.get("small_cluster_points"),
            "by_label": fine.get("by_label", {}),
            "clustered_by_label": fine.get("clustered_by_label", {}),
            "params": fine.get("params", {}),
            "output_ply": fine.get("output_ply", ""),
        },
        "fine_sweep": [
            {
                "name": row.get("name"),
                "cluster_count": row.get("cluster_count"),
                "clustered_ratio": row.get("clustered_ratio"),
                "clustered_points": row.get("clustered_points"),
                "small_cluster_points": row.get("small_cluster_points"),
            }
            for row in sweep.get("configs", [])
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
