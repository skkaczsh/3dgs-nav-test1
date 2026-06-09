#!/usr/bin/env python3
"""Classify manual equipment subclusters for QA.

This consumes split_manual_equipment_clusters.py JSON output and assigns each
subcluster to a conservative review bucket.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def span(row: dict) -> list[float]:
    bbox = row["bbox_3d"]
    return [float(bbox["max"][i] - bbox["min"][i]) for i in range(3)]


def classify(row: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    sx, sy, sz = span(row)
    max_xy = max(sx, sy)
    max_span = max(sx, sy, sz)
    points = int(row["points"])
    linearity = float(row.get("linearity", 0.0))
    planarity = float(row.get("planarity", 0.0))
    reasons = []

    if points >= args.large_points:
        reasons.append("large_points")
    if max_span >= args.large_span:
        reasons.append("large_span")
    if max_xy >= args.long_xy_span and linearity >= args.linear_threshold:
        reasons.append("long_linear")
    if max_xy >= args.large_xy_span and planarity >= args.planar_threshold:
        reasons.append("large_planar")
    if sz >= args.tall_z_span and linearity >= args.linear_threshold:
        reasons.append("tall_linear")

    if "long_linear" in reasons or "tall_linear" in reasons:
        return "linear_edge_review", reasons
    if "large_points" in reasons or "large_span" in reasons or "large_planar" in reasons:
        return "large_mixed_review", reasons
    return "fine_candidate", reasons or ["compact"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--large-points", type=int, default=5000)
    parser.add_argument("--large-span", type=float, default=6.0)
    parser.add_argument("--large-xy-span", type=float, default=4.0)
    parser.add_argument("--long-xy-span", type=float, default=4.0)
    parser.add_argument("--tall-z-span", type=float, default=2.5)
    parser.add_argument("--linear-threshold", type=float, default=0.90)
    parser.add_argument("--planar-threshold", type=float, default=0.50)
    args = parser.parse_args()

    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    rows = []
    action_counts = Counter()
    point_counts = Counter()
    for row in split.get("top_subclusters", []):
        action, reasons = classify(row, args)
        sx, sy, sz = span(row)
        out = {
            "subcluster_id": int(row["subcluster_id"]),
            "source_cluster": int(row["source_cluster"]),
            "points": int(row["points"]),
            "span_x": sx,
            "span_y": sy,
            "span_z": sz,
            "linearity": float(row.get("linearity", 0.0)),
            "planarity": float(row.get("planarity", 0.0)),
            "mean_visual_color": row.get("mean_visual_color", []),
            "recommended_action": action,
            "reasons": reasons,
        }
        rows.append(out)
        action_counts[action] += 1
        point_counts[action] += int(row["points"])
    rows.sort(key=lambda r: (r["recommended_action"] != "fine_candidate", -r["points"]))

    summary = {
        "split_report": str(args.split_report),
        "params": {
            "large_points": args.large_points,
            "large_span": args.large_span,
            "large_xy_span": args.large_xy_span,
            "long_xy_span": args.long_xy_span,
            "tall_z_span": args.tall_z_span,
            "linear_threshold": args.linear_threshold,
            "planar_threshold": args.planar_threshold,
        },
        "subcluster_count": len(rows),
        "action_counts": dict(action_counts),
        "point_counts": dict(point_counts),
        "subclusters": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subcluster_id",
        "source_cluster",
        "points",
        "span_x",
        "span_y",
        "span_z",
        "linearity",
        "planarity",
        "recommended_action",
        "reasons",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    print(json.dumps({k: summary[k] for k in ["subcluster_count", "action_counts", "point_counts"]}, indent=2))


if __name__ == "__main__":
    main()
