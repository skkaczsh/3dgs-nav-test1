#!/usr/bin/env python3
"""Run a bounded surface plane-split + strict-fusion evaluation on an existing target/object bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(args: list[str]) -> None:
    print("+", " ".join(str(x) for x in args), flush=True)
    subprocess.run(args, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label_counts(objects_jsonl: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in load_jsonl(objects_jsonl):
        label = str(row.get("semantic_label", "unknown"))
        bucket = counts.setdefault(label, {"objects": 0, "points": 0, "targets": 0})
        bucket["objects"] += 1
        bucket["points"] += int(row.get("point_count", 0))
        bucket["targets"] += int(row.get("target_count", len(row.get("targets", []))))
    return dict(sorted(counts.items()))


def report_surface_slice(report: dict) -> dict:
    out = {}
    for label in ("floor", "wall", "building", "ceiling"):
        entry = (report.get("by_label") or {}).get(label)
        if entry:
            out[label] = {
                "target_count": entry.get("target_count"),
                "point_count": entry.get("point_count"),
            }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Existing target_object_fusion bundle root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--write-centroid-ply", action="store_true")
    parser.add_argument("--min-split-points", type=int, default=240)
    parser.add_argument("--min-plane-points", type=int, default=120)
    parser.add_argument("--min-component-points", type=int, default=40)
    parser.add_argument("--min-residual-points", type=int, default=120)
    parser.add_argument("--plane-distance", type=float, default=0.055)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--max-planes", type=int, default=4)
    parser.add_argument("--ransac-iters", type=int, default=96)
    parser.add_argument("--max-fit-points", type=int, default=1200)
    parser.add_argument("--floor-normal-z", type=float, default=0.72)
    parser.add_argument("--wall-normal-z", type=float, default=0.40)
    parser.add_argument("--enable-ceiling-heuristic", action="store_true")
    parser.add_argument("--ceiling-source-labels", nargs="+", default=["floor", "building"])
    parser.add_argument("--ceiling-min-z", type=float, default=2.0)
    parser.add_argument("--ceiling-max-xy-area", type=float, default=8.0)
    parser.add_argument("--ceiling-max-z-extent", type=float, default=0.35)
    parser.add_argument("--ceiling-min-minor-extent", type=float, default=0.30)
    parser.add_argument("--ceiling-max-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--enable-ceiling-support-heuristic", action="store_true")
    parser.add_argument("--ceiling-candidate-labels", nargs="+", default=["floor"])
    parser.add_argument("--ceiling-support-source-labels", nargs="+", default=["floor", "building"])
    parser.add_argument("--ceiling-support-labels", nargs="+", default=["wall", "building"])
    parser.add_argument("--ceiling-top-gap-max", type=float, default=0.15)
    parser.add_argument("--ceiling-support-z-gap-max", type=float, default=0.6)
    parser.add_argument("--ceiling-support-xy-gap-max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--surface-labels", nargs="+", default=["floor", "wall", "building"])
    parser.add_argument("--surface-min-points", type=int, default=100)
    parser.add_argument("--surface-max-bbox-gap", type=float, default=0.35)
    parser.add_argument("--surface-max-centroid-distance", type=float, default=1.0)
    parser.add_argument("--surface-max-normal-angle", type=float, default=15.0)
    parser.add_argument("--surface-max-plane-distance", type=float, default=0.20)
    parser.add_argument("--surface-max-color-distance", type=float, default=65.0)
    parser.add_argument(
        "--label-config",
        action="append",
        default=[],
        help="Pass-through same-label consolidation config, e.g. label=floor,min_points=200,bbox=0.2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    targets_dir = input_dir / "targets"
    baseline_objects = input_dir / "objects" / "objects.jsonl"
    baseline_decisions = input_dir / "objects" / "fusion_decisions.jsonl"
    if not targets_dir.is_dir():
        raise SystemExit(f"missing targets dir: {targets_dir}")
    if not baseline_objects.exists():
        raise SystemExit(f"missing baseline objects: {baseline_objects}")

    out = args.output_dir
    reports_dir = out / "reports"
    split_targets_dir = out / "split_targets"
    strict_fused_dir = out / "fused_strict_surface"
    consolidated_dir = out / "same_label_surface_consolidated"
    out.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_bottleneck = reports_dir / "baseline_surface_bottleneck.json"
    run_cmd(
        [
            args.python,
            str(SCRIPT_DIR / "analyze_surface_target_fusion_bottleneck.py"),
            "--targets-dir",
            str(targets_dir),
            "--objects-jsonl",
            str(baseline_objects),
            "--fusion-decisions-jsonl",
            str(baseline_decisions),
            "--output-json",
            str(baseline_bottleneck),
        ]
    )

    split_report = reports_dir / "split_surface_targets_report.json"
    split_cmd = [
        args.python,
        str(SCRIPT_DIR / "split_surface_targets_by_plane.py"),
        "--input-targets",
        str(targets_dir),
        "--output-targets",
        str(split_targets_dir),
        "--report",
        str(split_report),
        "--min-split-points",
        str(args.min_split_points),
        "--min-plane-points",
        str(args.min_plane_points),
        "--min-component-points",
        str(args.min_component_points),
        "--min-residual-points",
        str(args.min_residual_points),
        "--plane-distance",
        str(args.plane_distance),
        "--voxel-size",
        str(args.voxel_size),
        "--max-planes",
        str(args.max_planes),
        "--ransac-iters",
        str(args.ransac_iters),
        "--max-fit-points",
        str(args.max_fit_points),
        "--floor-normal-z",
        str(args.floor_normal_z),
        "--wall-normal-z",
        str(args.wall_normal_z),
        "--seed",
        str(args.seed),
    ]
    if args.enable_ceiling_heuristic:
        split_cmd.extend(
            [
                "--enable-ceiling-heuristic",
                "--ceiling-source-labels",
                *args.ceiling_source_labels,
                "--ceiling-min-z",
                str(args.ceiling_min_z),
                "--ceiling-max-xy-area",
                str(args.ceiling_max_xy_area),
                "--ceiling-max-z-extent",
                str(args.ceiling_max_z_extent),
                "--ceiling-min-minor-extent",
                str(args.ceiling_min_minor_extent),
                "--ceiling-max-aspect-ratio",
                str(args.ceiling_max_aspect_ratio),
            ]
        )
    if args.enable_ceiling_support_heuristic:
        split_cmd.extend(
            [
                "--enable-ceiling-support-heuristic",
                "--ceiling-candidate-labels",
                *args.ceiling_candidate_labels,
                "--ceiling-support-source-labels",
                *args.ceiling_support_source_labels,
                "--ceiling-support-labels",
                *args.ceiling_support_labels,
                "--ceiling-top-gap-max",
                str(args.ceiling_top_gap_max),
                "--ceiling-support-z-gap-max",
                str(args.ceiling_support_z_gap_max),
                "--ceiling-support-xy-gap-max",
                str(args.ceiling_support_xy_gap_max),
            ]
        )
    run_cmd(split_cmd)

    fuse_cmd = [
        args.python,
        str(SCRIPT_DIR / "fuse_targets_to_objects.py"),
        "--targets",
        str(split_targets_dir),
        "--output-dir",
        str(strict_fused_dir),
        "--strict-surface-labels",
    ]
    if args.write_centroid_ply:
        fuse_cmd.append("--write-ply")
    run_cmd(fuse_cmd)

    strict_bottleneck = reports_dir / "strict_surface_bottleneck.json"
    run_cmd(
        [
            args.python,
            str(SCRIPT_DIR / "analyze_surface_target_fusion_bottleneck.py"),
            "--targets-dir",
            str(split_targets_dir),
            "--objects-jsonl",
            str(strict_fused_dir / "objects.jsonl"),
            "--fusion-decisions-jsonl",
            str(strict_fused_dir / "fusion_decisions.jsonl"),
            "--output-json",
            str(strict_bottleneck),
        ]
    )

    consolidate_cmd = [
        args.python,
        str(SCRIPT_DIR / "consolidate_same_label_surface_objects.py"),
        "--objects-jsonl",
        str(strict_fused_dir / "objects.jsonl"),
        "--output-jsonl",
        str(consolidated_dir / "objects.jsonl"),
        "--output-report",
        str(reports_dir / "same_label_surface_consolidation_report.json"),
        "--output-mapping",
        str(consolidated_dir / "source_to_consolidated.jsonl"),
        "--labels",
        *args.surface_labels,
        "--min-points",
        str(args.surface_min_points),
        "--max-bbox-gap",
        str(args.surface_max_bbox_gap),
        "--max-centroid-distance",
        str(args.surface_max_centroid_distance),
        "--max-normal-angle",
        str(args.surface_max_normal_angle),
        "--max-plane-distance",
        str(args.surface_max_plane_distance),
        "--max-color-distance",
        str(args.surface_max_color_distance),
    ]
    for text in args.label_config:
        consolidate_cmd.extend(["--label-config", text])
    run_cmd(consolidate_cmd)

    baseline_counts = label_counts(baseline_objects)
    strict_counts = label_counts(strict_fused_dir / "objects.jsonl")
    consolidated_counts = label_counts(consolidated_dir / "objects.jsonl")
    split_summary = load_json(split_report)
    baseline_surface = report_surface_slice(load_json(baseline_bottleneck))
    strict_surface = report_surface_slice(load_json(strict_bottleneck))
    fusion_report = load_json(strict_fused_dir / "fusion_report.json")
    consolidation_report = load_json(reports_dir / "same_label_surface_consolidation_report.json")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(out),
        "baseline": {
            "object_label_counts": baseline_counts,
            "surface_target_summary": baseline_surface,
        },
        "split": {
            "summary": split_summary.get("summary", {}),
            "split_ratio": split_summary.get("split_ratio"),
        },
        "strict_surface_fusion": {
            "fusion_report": fusion_report,
            "object_label_counts": strict_counts,
            "surface_target_summary": strict_surface,
        },
        "same_label_surface_consolidation": {
            "report": consolidation_report,
            "object_label_counts": consolidated_counts,
        },
        "headline": {
            "baseline_objects": sum(v["objects"] for v in baseline_counts.values()),
            "strict_objects": sum(v["objects"] for v in strict_counts.values()),
            "consolidated_objects": sum(v["objects"] for v in consolidated_counts.values()),
            "wall_objects_baseline": baseline_counts.get("wall", {}).get("objects", 0),
            "wall_objects_strict": strict_counts.get("wall", {}).get("objects", 0),
            "wall_objects_consolidated": consolidated_counts.get("wall", {}).get("objects", 0),
            "floor_objects_baseline": baseline_counts.get("floor", {}).get("objects", 0),
            "floor_objects_strict": strict_counts.get("floor", {}).get("objects", 0),
            "floor_objects_consolidated": consolidated_counts.get("floor", {}).get("objects", 0),
            "ceiling_objects_baseline": baseline_counts.get("ceiling", {}).get("objects", 0),
            "ceiling_objects_strict": strict_counts.get("ceiling", {}).get("objects", 0),
            "ceiling_objects_consolidated": consolidated_counts.get("ceiling", {}).get("objects", 0),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["headline"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
