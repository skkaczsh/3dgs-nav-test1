#!/usr/bin/env python3
"""Find non-stable Objects that could safely act as surface seeds.

Residual absorption currently depends on stable surface Objects. If large
surface-like Objects remain `single_target` or `ambiguous_object`, nearby
residual points report `no_candidate_cell`. This diagnostic does not mutate the
dataset; it ranks candidate Objects that could be promoted or split in a later
surface-seed pass.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SURFACE_LABELS = {"floor", "wall", "building", "road"}
SURFACE_COMPATIBLE_LABELS = SURFACE_LABELS | {"ambiguous"}


def extent_from_bbox(bbox: dict[str, Any]) -> np.ndarray:
    bmin = np.array(bbox.get("min", [0, 0, 0]), dtype=np.float64)
    bmax = np.array(bbox.get("max", [0, 0, 0]), dtype=np.float64)
    return np.maximum(bmax - bmin, 0.0)


def surface_vote_stats(obj: dict[str, Any]) -> dict[str, Any]:
    votes = {str(k): float(v) for k, v in obj.get("label_votes", {}).items()}
    total = float(sum(votes.values()))
    surface_votes = {k: v for k, v in votes.items() if k in SURFACE_LABELS}
    surface_total = float(sum(surface_votes.values()))
    dominant_surface_label = ""
    dominant_surface_ratio = 0.0
    if surface_votes:
        dominant_surface_label, dominant_surface_count = max(surface_votes.items(), key=lambda kv: kv[1])
        dominant_surface_ratio = float(dominant_surface_count / max(surface_total, 1.0))
    return {
        "vote_total": total,
        "surface_vote_total": surface_total,
        "surface_vote_ratio": float(surface_total / max(total, 1.0)),
        "dominant_surface_label": dominant_surface_label,
        "dominant_surface_ratio": dominant_surface_ratio,
        "surface_votes": surface_votes,
    }


def candidate_reason(obj: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str, dict[str, Any]]:
    status = obj.get("status", "")
    label = obj.get("semantic_label", "unknown")
    points = int(obj.get("point_count", 0))
    targets = int(obj.get("target_count", 0))
    extent = extent_from_bbox(obj.get("bbox_3d", {}))
    sorted_extent = np.sort(extent)[::-1]
    max_extent = float(sorted_extent[0]) if len(sorted_extent) else 0.0
    mid_extent = float(sorted_extent[1]) if len(sorted_extent) > 1 else 0.0
    min_extent = float(sorted_extent[2]) if len(sorted_extent) > 2 else 0.0
    planarity = float(obj.get("geometry_stats", {}).get("planarity_mean", 0.0))
    linearity = float(obj.get("geometry_stats", {}).get("linearity_mean", 0.0))
    vote = surface_vote_stats(obj)
    quality = obj.get("quality_stats", {})
    low_conf = int(quality.get("low_confidence_targets", 0))
    mixed = int(quality.get("mixed_targets", 0))

    meta = {
        "status": status,
        "semantic_label": label,
        "point_count": points,
        "target_count": targets,
        "extent": [float(x) for x in extent],
        "max_extent": max_extent,
        "mid_extent": mid_extent,
        "min_extent": min_extent,
        "planarity_mean": planarity,
        "linearity_mean": linearity,
        "low_confidence_targets": low_conf,
        "mixed_targets": mixed,
        **vote,
    }

    if status == "stable":
        return False, "already_stable", meta
    if label not in SURFACE_COMPATIBLE_LABELS and vote["surface_vote_ratio"] < args.min_surface_vote_ratio:
        return False, "not_surface_semantic", meta
    if points < args.min_points:
        return False, "too_few_points", meta
    if targets < args.min_targets:
        return False, "too_few_targets", meta
    if max_extent < args.min_max_extent or mid_extent < args.min_mid_extent:
        return False, "too_small_extent", meta
    if planarity < args.min_planarity and min_extent > args.max_thickness:
        return False, "not_planar_enough", meta
    if low_conf > 0 or mixed > args.max_mixed_targets:
        return False, "quality_blocked", meta
    if label == "ambiguous" and vote["surface_vote_ratio"] >= args.min_surface_vote_ratio:
        return True, "ambiguous_surface_votes", meta
    if label in SURFACE_LABELS:
        return True, "surface_label_nonstable", meta
    return False, "not_promoted", meta


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    counts = Counter()
    point_counts = Counter()
    candidates = []
    rejected_examples: dict[str, list[dict[str, Any]]] = {}
    total_objects = 0
    with args.objects_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_objects += 1
            obj = json.loads(line)
            ok, reason, meta = candidate_reason(obj, args)
            counts[reason] += 1
            point_counts[reason] += int(meta["point_count"])
            row = {
                "object_id": obj.get("object_id"),
                "reason": reason,
                "candidate": ok,
                "label_votes": obj.get("label_votes", {}),
                "bbox_3d": obj.get("bbox_3d", {}),
                "centroid": obj.get("centroid", []),
                **meta,
            }
            if ok:
                candidates.append(row)
            elif len(rejected_examples.setdefault(reason, [])) < args.example_limit:
                rejected_examples[reason].append(row)
    candidates.sort(key=lambda row: row["point_count"], reverse=True)
    return {
        "objects_jsonl": str(args.objects_jsonl),
        "total_objects": total_objects,
        "candidate_count": len(candidates),
        "candidate_points": int(sum(row["point_count"] for row in candidates)),
        "reason_counts": dict(counts),
        "reason_point_counts": dict(point_counts),
        "params": {
            "min_points": args.min_points,
            "min_targets": args.min_targets,
            "min_surface_vote_ratio": args.min_surface_vote_ratio,
            "min_max_extent": args.min_max_extent,
            "min_mid_extent": args.min_mid_extent,
            "min_planarity": args.min_planarity,
            "max_thickness": args.max_thickness,
            "max_mixed_targets": args.max_mixed_targets,
        },
        "top_candidates": candidates[: args.top_n],
        "rejected_examples": rejected_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=500)
    parser.add_argument("--min-targets", type=int, default=2)
    parser.add_argument("--min-surface-vote-ratio", type=float, default=0.8)
    parser.add_argument("--min-max-extent", type=float, default=0.6)
    parser.add_argument("--min-mid-extent", type=float, default=0.2)
    parser.add_argument("--min-planarity", type=float, default=0.08)
    parser.add_argument("--max-thickness", type=float, default=0.25)
    parser.add_argument("--max-mixed-targets", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--example-limit", type=int, default=10)
    args = parser.parse_args()

    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "total_objects": report["total_objects"],
            "candidate_count": report["candidate_count"],
            "candidate_points": report["candidate_points"],
            "reason_counts": report["reason_counts"],
            "reason_point_counts": report["reason_point_counts"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
