#!/usr/bin/env python3
"""Build a promotion/noise plan from fine residual cluster review outputs.

This script does not mutate point clouds or object files. It converts the
review heuristics plus mask trace metadata into a small decision table:

- promote_seed: good automatic object-seed candidates.
- hold_noise: likely surface-like contamination or over-merged residuals.
- manual_review: candidates that need visual/model review before promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trace_by_cluster(trace: dict) -> dict[int, dict]:
    return {int(row["cluster_id"]): row for row in trace.get("clusters", [])}


def source_rows_by_cluster(trace: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for row in trace.get("source_rows", []):
        out.setdefault(int(row["cluster_id"]), []).append(row)
    return out


def action_for(row: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    status = str(row.get("status", ""))
    label = str(row.get("label", "unknown"))
    points = int(row.get("points", 0))
    reasons = [str(x) for x in row.get("reasons", [])]
    linearity = float(row.get("linearity", 0.0))
    span_x = float(row.get("span_x", 0.0))
    span_y = float(row.get("span_y", 0.0))
    span_z = float(row.get("span_z", 0.0))
    max_xy = max(span_x, span_y)

    if status == "likely_fine_object":
        return "promote_seed", ["review_status_likely_fine_object"]
    if status == "review_suspicious":
        return "hold_noise", ["review_status_suspicious", *reasons]
    if label == "railing" and linearity >= args.railing_linearity and points >= args.min_promote_points:
        return "promote_seed", ["railing_linear_candidate"]
    if label in {"equipment", "pipe"} and points >= args.min_promote_points and max_xy <= args.max_compact_xy and span_z <= args.max_compact_z:
        return "manual_review", ["compact_fine_candidate"]
    return "manual_review", reasons or ["review_candidate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-promote-points", type=int, default=50)
    parser.add_argument("--railing-linearity", type=float, default=0.75)
    parser.add_argument("--max-compact-xy", type=float, default=4.0)
    parser.add_argument("--max-compact-z", type=float, default=2.5)
    parser.add_argument("--top-sources", type=int, default=6)
    args = parser.parse_args()

    review = load_json(args.review_json)
    trace = load_json(args.trace_json)
    trace_clusters = trace_by_cluster(trace)
    trace_sources = source_rows_by_cluster(trace)

    rows = []
    for row in review.get("review_rows", []):
        cluster_id = int(row["cluster_id"])
        action, action_reasons = action_for(row, args)
        trace_row = trace_clusters.get(cluster_id, {})
        sources = trace_sources.get(cluster_id, [])[: args.top_sources]
        rows.append(
            {
                "cluster_id": cluster_id,
                "label": row.get("label", "unknown"),
                "points": int(row.get("points", 0)),
                "action": action,
                "action_reasons": action_reasons,
                "review_status": row.get("status", ""),
                "review_reasons": row.get("reasons", []),
                "linearity": float(row.get("linearity", 0.0)),
                "planarity": float(row.get("planarity", 0.0)),
                "span_x": float(row.get("span_x", 0.0)),
                "span_y": float(row.get("span_y", 0.0)),
                "span_z": float(row.get("span_z", 0.0)),
                "centroid": row.get("centroid", []),
                "bbox_3d": row.get("bbox_3d", {}),
                "trace_matched_ratio": trace_row.get("matched_ratio"),
                "top_sources": [
                    {
                        "frame_id": int(src["frame_id"]),
                        "cam_id": int(src["cam_id"]),
                        "mask_id": int(src["mask_id"]),
                        "label": src.get("label", ""),
                        "points": int(src["points"]),
                        "share_of_cluster": float(src.get("share_of_cluster", 0.0)),
                        "mask_area_ratio": float(src.get("mask_area_ratio", 0.0)),
                        "overlay_path": src.get("overlay_path", ""),
                    }
                    for src in sources
                ],
            }
        )

    action_order = {"promote_seed": 0, "hold_noise": 1, "manual_review": 2}
    rows.sort(key=lambda r: (action_order.get(r["action"], 9), -int(r["points"]), int(r["cluster_id"])))
    action_counts = Counter(row["action"] for row in rows)
    label_action_counts = Counter((row["label"], row["action"]) for row in rows)
    summary = {
        "review_json": str(args.review_json),
        "trace_json": str(args.trace_json),
        "cluster_count": len(rows),
        "action_counts": dict(action_counts),
        "label_action_counts": {
            f"{label}:{action}": count for (label, action), count in sorted(label_action_counts.items())
        },
        "params": {
            "min_promote_points": args.min_promote_points,
            "railing_linearity": args.railing_linearity,
            "max_compact_xy": args.max_compact_xy,
            "max_compact_z": args.max_compact_z,
            "top_sources": args.top_sources,
        },
        "promotion_candidates": [row for row in rows if row["action"] == "promote_seed"],
        "hold_noise": [row for row in rows if row["action"] == "hold_noise"],
        "manual_review": [row for row in rows if row["action"] == "manual_review"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cluster_id",
        "label",
        "points",
        "action",
        "action_reasons",
        "review_status",
        "review_reasons",
        "linearity",
        "planarity",
        "span_x",
        "span_y",
        "span_z",
        "trace_matched_ratio",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], list) else row[key] for key in fields})

    print(json.dumps({k: summary[k] for k in ["cluster_count", "action_counts", "label_action_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
