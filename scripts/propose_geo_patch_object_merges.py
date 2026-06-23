#!/usr/bin/env python3
"""Propose geometry-patch object merge candidates without changing ownership.

This is an object-building aid after the patch optimizer.  It keeps the
one-voxel-one-owner patch labels intact and reports adjacent patch pairs that
look like the same higher-level object according to local contact, color,
geometry bucket, normal, bbox gap, and size balance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from optimize_patch_graph_energy import (
    PatchStats,
    bbox_gap,
    compatible_bucket_score,
    compute_patch_stats,
    normal_score,
    read_labels,
    read_region_input,
)


def build_edge_counts(labels: np.ndarray, src: np.ndarray, dst: np.ndarray) -> dict[tuple[int, int], int]:
    if len(src) == 0:
        return {}
    a = labels[src]
    b = labels[dst]
    mask = a != b
    if not np.any(mask):
        return {}

    a = a[mask].astype(np.int64, copy=False)
    b = b[mask].astype(np.int64, copy=False)
    hi = a > b
    aa = a.copy()
    a = np.where(hi, b, a)
    b = np.where(hi, aa, b)
    max_label = int(labels.max())
    keys = a * (max_label + 1) + b
    uk, uc = np.unique(keys, return_counts=True)
    return {
        (int(k // (max_label + 1)), int(k % (max_label + 1))): int(c)
        for k, c in zip(uk.tolist(), uc.tolist())
    }


def size_balance_score(a: PatchStats, b: PatchStats) -> float:
    small = max(float(min(a.count, b.count)), 1.0)
    large = max(float(max(a.count, b.count)), 1.0)
    return math.sqrt(small / large)


def candidate_features(a: PatchStats, b: PatchStats, shared_edges: int, args: argparse.Namespace) -> dict[str, float]:
    color_dist = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
    color_score = max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
    bucket_score = compatible_bucket_score(a.geometry_type, b.geometry_type)
    n_score = normal_score(a.mean_normal, b.mean_normal)
    gap = bbox_gap(a, b)
    gap_score = max(0.0, min(1.0, 1.0 - gap / max(args.max_bbox_gap, 1e-6)))
    contact_ratio = float(shared_edges) / max(float(min(a.count, b.count)), 1.0)
    contact_score = max(0.0, min(1.0, contact_ratio / max(args.contact_ratio_norm, 1e-6)))
    balance = size_balance_score(a, b)
    size_ratio = float(max(a.count, b.count)) / max(float(min(a.count, b.count)), 1.0)
    big_mixed_attachment = 1.0 if (
        max(a.count, b.count) >= args.big_anchor_voxels
        and min(a.count, b.count) <= args.small_fragment_voxels
        and "mixed" in {a.geometry_type, b.geometry_type}
    ) else 0.0
    score = (
        args.color_weight * color_score
        + args.bucket_weight * bucket_score
        + args.normal_weight * n_score
        + args.gap_weight * gap_score
        + args.contact_weight * contact_score
        + args.balance_weight * balance
    ) / max(
        args.color_weight
        + args.bucket_weight
        + args.normal_weight
        + args.gap_weight
        + args.contact_weight
        + args.balance_weight,
        1e-9,
    )
    return {
        "score": score,
        "color_distance": color_dist,
        "color_score": color_score,
        "bucket_score": bucket_score,
        "normal_score": n_score,
        "bbox_gap": gap,
        "gap_score": gap_score,
        "shared_edges": float(shared_edges),
        "contact_ratio_min": contact_ratio,
        "contact_score": contact_score,
        "size_balance": balance,
        "size_ratio": size_ratio,
        "big_mixed_attachment": big_mixed_attachment,
    }


def reject_reason(a: PatchStats, b: PatchStats, f: dict[str, float], args: argparse.Namespace) -> str | None:
    if min(a.count, b.count) < args.min_patch_voxels:
        return "small_patch"
    if f["shared_edges"] < args.min_shared_edges:
        return "low_shared_edges"
    if f["contact_ratio_min"] < args.min_contact_ratio:
        return "low_contact_ratio"
    if f["bbox_gap"] > args.max_bbox_gap:
        return "bbox_gap"
    if f["color_distance"] > args.max_color_distance:
        return "color_distance"
    if f["normal_score"] < args.min_normal_score and {a.geometry_type, b.geometry_type} <= {"horizontal", "vertical"}:
        return "stable_normal_mismatch"
    if f["bucket_score"] < args.min_bucket_score:
        return "bucket_mismatch"
    if f["score"] < args.min_score:
        return "score"
    return None


def propose(arrays: dict[str, np.ndarray], labels: np.ndarray, src: np.ndarray, dst: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats = compute_patch_stats(arrays, labels)
    edge_counts = build_edge_counts(labels, src, dst)
    rows: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    for (a_id, b_id), shared in edge_counts.items():
        a = stats.get(a_id)
        b = stats.get(b_id)
        if a is None or b is None:
            continue
        f = candidate_features(a, b, shared, args)
        reason = reject_reason(a, b, f, args)
        if reason:
            reject_counts[reason] += 1
            continue
        rows.append(
            {
                "patch_a": int(a_id),
                "patch_b": int(b_id),
                "voxels_a": int(a.count),
                "voxels_b": int(b.count),
                "geometry_a": a.geometry_type,
                "geometry_b": b.geometry_type,
                "centroid_a": a.centroid.tolist(),
                "centroid_b": b.centroid.tolist(),
                "mean_rgb_a": a.mean_rgb.tolist(),
                "mean_rgb_b": b.mean_rgb.tolist(),
                **{k: float(v) for k, v in f.items()},
            }
        )
    rows.sort(key=lambda row: (row["score"], row["shared_edges"]), reverse=True)
    if args.max_candidates > 0:
        rows = rows[: args.max_candidates]
    report = {
        "schema": "geo-patch-object-merge-candidates/v1",
        "patch_count": len(stats),
        "edge_pair_count": len(edge_counts),
        "candidate_count": len(rows),
        "reject_counts": dict(reject_counts),
        "candidate_geometry_pairs": {
            " + ".join(pair): int(count)
            for pair, count in Counter(tuple(sorted((row["geometry_a"], row["geometry_b"]))) for row in rows).most_common()
        },
        "big_mixed_attachment_count": sum(1 for row in rows if row["big_mixed_attachment"] > 0),
        "params": vars(args),
    }
    return rows, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "patch_a",
        "patch_b",
        "voxels_a",
        "voxels_b",
        "geometry_a",
        "geometry_b",
        "score",
        "color_distance",
        "bucket_score",
        "normal_score",
        "bbox_gap",
        "shared_edges",
        "contact_ratio_min",
        "size_balance",
        "size_ratio",
        "big_mixed_attachment",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="geo_patch_object_merge_candidates")

    parser.add_argument("--min-patch-voxels", type=int, default=400)
    parser.add_argument("--min-shared-edges", type=int, default=8)
    parser.add_argument("--min-contact-ratio", type=float, default=0.015)
    parser.add_argument("--max-bbox-gap", type=float, default=0.18)
    parser.add_argument("--max-color-distance", type=float, default=90.0)
    parser.add_argument("--min-normal-score", type=float, default=0.45)
    parser.add_argument("--min-bucket-score", type=float, default=0.55)
    parser.add_argument("--min-score", type=float, default=0.62)
    parser.add_argument("--contact-ratio-norm", type=float, default=0.18)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--big-anchor-voxels", type=int, default=100000)
    parser.add_argument("--small-fragment-voxels", type=int, default=1200)

    parser.add_argument("--color-weight", type=float, default=0.25)
    parser.add_argument("--bucket-weight", type=float, default=0.20)
    parser.add_argument("--normal-weight", type=float, default=0.18)
    parser.add_argument("--gap-weight", type=float, default=0.12)
    parser.add_argument("--contact-weight", type=float, default=0.20)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    rows, report = propose(arrays, labels, src, dst, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.output_stem}.jsonl"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    report_path = args.output_dir / f"{args.output_stem}_report.json"
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    report.update(
        {
            "output_jsonl": str(jsonl_path),
            "output_csv": str(csv_path),
            "output_report": str(report_path),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
