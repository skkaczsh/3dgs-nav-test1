#!/usr/bin/env python3
"""Diagnose why geo primitive candidate edges are rejected.

This mirrors the candidate construction and scoring used by
cluster_geo_primitives_unified.py, but records rejected candidates instead of
only accepted merge edges.  It is intended for tuning small-object and
fragment-absorption behavior without guessing which term is doing the damage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from optimize_geo_patch_merges import compute_patch_stats, read_labels, read_region_input
from cluster_geo_primitives_unified import build_bbox_candidate_pairs, patch_edges, score_edge


ROUGH_TYPES = {"rough_mixed", "thin_linear", "unknown", "mixed"}
STABLE_TYPES = {"horizontal", "vertical"}


def size_class(a_count: int, b_count: int) -> str:
    if a_count < 64 and b_count < 64:
        return "both_lt64"
    if min(a_count, b_count) < 64:
        return "one_lt64"
    if min(a_count, b_count) < 512:
        return "one_lt512"
    return "both_ge512"


def edge_threshold(reason: str, source: str, args: argparse.Namespace) -> float:
    threshold = args.min_edge_score
    if reason == "porous":
        threshold = args.min_porous_score
    elif reason == "interleaved":
        threshold = args.min_interleaved_score
    elif source == "bbox":
        threshold = max(threshold, args.min_bbox_edge_score)
    return threshold


def failure_flags(features: dict[str, float | str], args: argparse.Namespace) -> list[str]:
    geom_a = str(features["geom_a"])
    geom_b = str(features["geom_b"])
    flags: list[str] = []
    checks = {
        "very_low_contact": float(features["contact"]) < 0.03,
        "low_contact": float(features["contact"]) < 0.12,
        "low_color": float(features["color"]) < 0.50,
        "hard_color_penalty": float(features["color_dist"]) > args.hard_color_distance,
        "low_bucket": float(features["bucket"]) < 0.50,
        "low_normal": float(features["normal"]) < 0.50,
        "low_balance": float(features["balance"]) < 0.08,
        "low_bbox_overlap": float(features["bbox_ratio_min"]) < 0.20,
        "stable_mismatch_penalty": geom_a in STABLE_TYPES and geom_b in STABLE_TYPES and geom_a != geom_b,
        "stable_to_rough_penalty": (
            (geom_a in STABLE_TYPES and geom_b in ROUGH_TYPES)
            or (geom_b in STABLE_TYPES and geom_a in ROUGH_TYPES)
        )
        and float(features["bbox_ratio_min"]) < args.stable_rough_min_bbox_ratio,
    }
    for name, hit in checks.items():
        if hit:
            flags.append(name)
    return flags


def edge_record(
    pa: int,
    pb: int,
    score: float,
    threshold: float,
    reason: str,
    source: str,
    features: dict[str, float | str],
    stats: dict[int, Any],
) -> dict[str, Any]:
    return {
        "a": int(pa),
        "b": int(pb),
        "score": round(float(score), 6),
        "threshold": round(float(threshold), 6),
        "gap": round(float(threshold - score), 6),
        "reason": reason,
        "source": source,
        "geom": [features["geom_a"], features["geom_b"]],
        "counts": [int(stats[pa].count), int(stats[pb].count)],
        "contact": round(float(features["contact"]), 6),
        "color": round(float(features["color"]), 6),
        "color_dist": round(float(features["color_dist"]), 3),
        "bucket": round(float(features["bucket"]), 6),
        "normal": round(float(features["normal"]), 6),
        "balance": round(float(features["balance"]), 6),
        "bbox_ratio_min": round(float(features["bbox_ratio_min"]), 6),
        "bbox_iou": round(float(features["bbox_iou"]), 6),
        "penalty": round(float(features["penalty"]), 6),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"label count mismatch: labels={len(labels)} voxels={len(arrays['xyz'])}")

    stats = compute_patch_stats(arrays, labels)
    adjacency = patch_edges(labels, src, dst)
    bbox_pairs = build_bbox_candidate_pairs(stats, args)
    candidate_pairs = set(adjacency)
    candidate_pairs.update(bbox_pairs)

    accepted_count = 0
    rejected_count = 0
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rejected_size_counts: Counter[str] = Counter()
    accepted_size_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    geom_pair_counts: Counter[tuple[str, str]] = Counter()
    feature_sums: defaultdict[str, float] = defaultdict(float)
    feature_counts: Counter[str] = Counter()
    near_miss_edges: list[dict[str, Any]] = []

    for pa, pb in candidate_pairs:
        if pa not in stats or pb not in stats:
            continue
        shared = adjacency.get((pa, pb), 0)
        source = "adjacency" if shared else "bbox"
        score, reason, features = score_edge(stats[pa], stats[pb], shared, source, args)
        threshold = edge_threshold(reason, source, args)
        pair_size_class = size_class(stats[pa].count, stats[pb].count)
        if score >= threshold:
            accepted_count += 1
            accepted_size_counts[pair_size_class] += 1
            continue

        rejected_count += 1
        reason_counts[reason] += 1
        source_counts[source] += 1
        rejected_size_counts[pair_size_class] += 1
        geom_pair_counts[tuple(sorted((str(features["geom_a"]), str(features["geom_b"])) ))] += 1
        for flag in failure_flags(features, args):
            failure_counts[flag] += 1
        for key in [
            "score_after_penalty",
            "contact",
            "color",
            "color_dist",
            "bucket",
            "normal",
            "balance",
            "bbox_ratio_min",
            "bbox_iou",
            "penalty",
        ]:
            feature_sums[key] += float(features[key])
            feature_counts[key] += 1
        if threshold - score <= args.near_miss_gap:
            near_miss_edges.append(edge_record(pa, pb, score, threshold, reason, source, features, stats))

    near_miss_edges.sort(key=lambda row: row["gap"])
    return {
        "schema": "geo-primitive-rejected-edge-diagnosis/v1",
        "patch_count": len(stats),
        "candidate_count": len(candidate_pairs),
        "adjacency_candidate_count": len(adjacency),
        "bbox_candidate_count": len(bbox_pairs),
        "accepted_initial_edges": accepted_count,
        "rejected_initial_edges": rejected_count,
        "rejected_by_best_score_reason": dict(reason_counts),
        "rejected_by_source": dict(source_counts),
        "rejected_by_size_class": dict(rejected_size_counts),
        "accepted_by_size_class": dict(accepted_size_counts),
        "rejected_failure_flags": dict(failure_counts),
        "rejected_feature_means": {
            key: round(feature_sums[key] / max(feature_counts[key], 1), 6)
            for key in sorted(feature_sums)
        },
        "top_rejected_geom_pairs": [
            {"pair": list(pair), "count": int(count)}
            for pair, count in geom_pair_counts.most_common(args.top_geom_pairs)
        ],
        "near_miss_edges": near_miss_edges[: args.top_near_miss],
        "params": vars(args),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bbox-top-n", type=int, default=3000)
    parser.add_argument("--bbox-min-patch-voxels", type=int, default=64)
    parser.add_argument("--bbox-candidate-min-ratio", type=float, default=0.72)
    parser.add_argument("--bbox-candidate-min-iou", type=float, default=0.10)
    parser.add_argument("--bbox-candidate-max-centroid", type=float, default=18.0)
    parser.add_argument("--long-pair-min-ratio", type=float, default=0.85)
    parser.add_argument("--long-patch-min-voxels", type=int, default=100000)
    parser.add_argument("--long-patch-aspect-ratio", type=float, default=2.5)
    parser.add_argument("--long-patch-min-bbox-ratio", type=float, default=0.80)
    parser.add_argument("--long-patch-min-bbox-iou", type=float, default=0.12)
    parser.add_argument("--min-edge-score", type=float, default=0.70)
    parser.add_argument("--min-bbox-edge-score", type=float, default=0.76)
    parser.add_argument("--min-porous-score", type=float, default=0.74)
    parser.add_argument("--min-interleaved-score", type=float, default=0.80)
    parser.add_argument("--stable-mismatch-penalty", type=float, default=0.24)
    parser.add_argument("--stable-rough-penalty", type=float, default=0.18)
    parser.add_argument("--stable-rough-min-bbox-ratio", type=float, default=0.80)
    parser.add_argument("--hard-color-distance", type=float, default=150.0)
    parser.add_argument("--hard-color-penalty", type=float, default=0.18)
    parser.add_argument("--near-miss-gap", type=float, default=0.08)
    parser.add_argument("--top-near-miss", type=int, default=80)
    parser.add_argument("--top-geom-pairs", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
