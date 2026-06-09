#!/usr/bin/env python3
"""Evaluate oversized-mask hygiene candidates from fine-cluster mask traces.

This script does not mutate target/object artifacts. It consumes the trace JSON
created by trace_fine_clusters_to_masks.py and estimates which suspicious fine
clusters are dominated by oversized source masks, making them candidates for
pre-fusion surface subtraction, 3D splitting, or demotion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def decide_action(row: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    reasons: list[str] = []
    label = str(row["label"])
    oversized_share = float(row["oversized_source_point_share"])
    max_area = float(row["max_mask_area_ratio"])
    max_bbox = float(row["max_mask_bbox_area_ratio"])
    source_coverage = float(row["source_point_coverage"])
    source_count = int(row["source_rows"])

    if max_area >= args.huge_mask_area_ratio:
        reasons.append("huge_mask_area")
    if max_bbox >= args.huge_bbox_area_ratio:
        reasons.append("huge_mask_bbox")
    if oversized_share >= args.oversized_point_share:
        reasons.append("oversized_source_dominant")
    if source_coverage < args.min_source_coverage:
        reasons.append("trace_source_coverage_low")
    if source_count >= args.many_sources:
        reasons.append("many_source_masks")

    if oversized_share >= args.oversized_point_share or max_area >= args.huge_mask_area_ratio:
        action = "pre_fusion_split_or_demote"
    elif label in {"equipment", "railing"} and max_bbox < args.large_bbox_area_ratio and oversized_share < 0.25:
        action = "fine_object_candidate"
    else:
        action = "manual_review"
    return action, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--large-mask-area-ratio", type=float, default=0.10)
    parser.add_argument("--large-bbox-area-ratio", type=float, default=0.30)
    parser.add_argument("--huge-mask-area-ratio", type=float, default=0.50)
    parser.add_argument("--huge-bbox-area-ratio", type=float, default=0.80)
    parser.add_argument("--oversized-point-share", type=float, default=0.35)
    parser.add_argument("--min-source-coverage", type=float, default=0.30)
    parser.add_argument("--many-sources", type=int, default=50)
    args = parser.parse_args()

    trace = json.loads(args.trace_json.read_text(encoding="utf-8"))
    source_rows = trace.get("source_rows", [])
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for row in source_rows:
        by_cluster[int(row["cluster_id"])].append(row)

    review_rows_by_cluster = {int(row["cluster_id"]): row for row in trace.get("clusters", [])}
    rows = []
    action_counts = Counter()
    label_counts = Counter()
    for cluster_id, sources in sorted(by_cluster.items()):
        cluster_meta = review_rows_by_cluster.get(cluster_id, {})
        cluster_points = int(cluster_meta.get("cluster_points") or sources[0].get("cluster_points", 0))
        source_points = sum(int(row.get("points", 0)) for row in sources)
        oversized = [
            row
            for row in sources
            if float(row.get("mask_area_ratio", 0.0)) >= args.large_mask_area_ratio
            or float(row.get("mask_bbox_area_ratio", 0.0)) >= args.large_bbox_area_ratio
        ]
        oversized_points = sum(int(row.get("points", 0)) for row in oversized)
        labels = Counter(str(row.get("label", "unknown")) for row in sources)
        label, label_votes = labels.most_common(1)[0] if labels else ("unknown", 0)
        out = {
            "cluster_id": cluster_id,
            "label": label,
            "label_source_rows": int(label_votes),
            "cluster_points": cluster_points,
            "source_rows": len(sources),
            "source_points": int(source_points),
            "source_point_coverage": float(source_points / max(cluster_points, 1)),
            "oversized_source_rows": len(oversized),
            "oversized_source_points": int(oversized_points),
            "oversized_source_point_share": float(oversized_points / max(source_points, 1)),
            "max_mask_area_ratio": max((float(row.get("mask_area_ratio", 0.0)) for row in sources), default=0.0),
            "max_mask_bbox_area_ratio": max((float(row.get("mask_bbox_area_ratio", 0.0)) for row in sources), default=0.0),
            "mean_mask_area_ratio": float(
                sum(float(row.get("mask_area_ratio", 0.0)) for row in sources) / max(len(sources), 1)
            ),
            "mean_mask_bbox_area_ratio": float(
                sum(float(row.get("mask_bbox_area_ratio", 0.0)) for row in sources) / max(len(sources), 1)
            ),
        }
        action, reasons = decide_action(out, args)
        out["recommended_action"] = action
        out["reasons"] = reasons
        rows.append(out)
        action_counts[action] += 1
        label_counts[label] += 1

    summary = {
        "trace_json": str(args.trace_json),
        "params": {
            "large_mask_area_ratio": args.large_mask_area_ratio,
            "large_bbox_area_ratio": args.large_bbox_area_ratio,
            "huge_mask_area_ratio": args.huge_mask_area_ratio,
            "huge_bbox_area_ratio": args.huge_bbox_area_ratio,
            "oversized_point_share": args.oversized_point_share,
            "min_source_coverage": args.min_source_coverage,
            "many_sources": args.many_sources,
        },
        "cluster_count": len(rows),
        "source_rows": len(source_rows),
        "action_counts": dict(action_counts),
        "label_counts": dict(label_counts),
        "clusters": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_id",
        "label",
        "cluster_points",
        "source_rows",
        "source_points",
        "source_point_coverage",
        "oversized_source_rows",
        "oversized_source_points",
        "oversized_source_point_share",
        "max_mask_area_ratio",
        "max_mask_bbox_area_ratio",
        "mean_mask_area_ratio",
        "mean_mask_bbox_area_ratio",
        "recommended_action",
        "reasons",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(json.dumps({k: summary[k] for k in ["cluster_count", "source_rows", "action_counts"]}, indent=2))


if __name__ == "__main__":
    main()
