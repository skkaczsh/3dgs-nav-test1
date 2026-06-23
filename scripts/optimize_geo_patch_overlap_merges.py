#!/usr/bin/env python3
"""Merge interleaved large patches when the merged patch model improves.

This handles a case that small-patch absorption and fine-cell boundary transfer
do not cover: two large patches have heavily overlapping AABBs, almost no
same-cell ownership conflict, and many points close to each other.  These are
candidate "interleaved structure" patches.  We only merge them when a combined
patch model has a better objective than keeping two separate models.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from optimize_geo_patch_merges import (
    PatchStats,
    compatible_bucket_score,
    compute_patch_stats,
    normal_score,
    read_labels,
    read_region_input,
    write_jsonl,
    write_ply,
)


@dataclass
class Candidate:
    patch_a: int
    patch_b: int
    ratio_min_volume: float
    ratio_max_volume: float
    bbox_iou: float
    centroid_distance: float
    overlap_dims: np.ndarray


class DSU:
    def __init__(self, values: list[int]) -> None:
        self.parent = {int(v): int(v) for v in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, keep: int, drop: int) -> None:
        keep_root = self.find(keep)
        drop_root = self.find(drop)
        if keep_root != drop_root:
            self.parent[drop_root] = keep_root


def bbox_overlap(a: PatchStats, b: PatchStats) -> tuple[float, float, float, float, np.ndarray, float] | None:
    dims = np.maximum(0.0, np.minimum(a.bbox_max, b.bbox_max) - np.maximum(a.bbox_min, b.bbox_min))
    overlap_volume = float(np.prod(dims))
    if overlap_volume <= 0:
        return None
    volume_a = float(np.prod(np.maximum(a.bbox_max - a.bbox_min, 1e-3)))
    volume_b = float(np.prod(np.maximum(b.bbox_max - b.bbox_min, 1e-3)))
    min_volume = min(volume_a, volume_b)
    max_volume = max(volume_a, volume_b)
    union = volume_a + volume_b - overlap_volume
    centroid_distance = float(np.linalg.norm(a.centroid - b.centroid))
    return (
        overlap_volume,
        overlap_volume / max(min_volume, 1e-9),
        overlap_volume / max(max_volume, 1e-9),
        overlap_volume / max(union, 1e-9),
        dims,
        centroid_distance,
    )


def build_candidates(stats: dict[int, PatchStats], args: argparse.Namespace) -> list[Candidate]:
    large = [s for s in stats.values() if s.count >= args.min_patch_voxels]
    large.sort(key=lambda s: s.count, reverse=True)
    if args.top_n > 0:
        large = large[: args.top_n]
    candidates: list[Candidate] = []
    for i, a in enumerate(large):
        for b in large[i + 1 :]:
            overlap = bbox_overlap(a, b)
            if overlap is None:
                continue
            _ov, ratio_min, ratio_max, iou, dims, centroid_distance = overlap
            if ratio_min < args.min_bbox_ratio:
                continue
            if iou < args.min_bbox_iou:
                continue
            if centroid_distance > args.max_centroid_distance:
                continue
            candidates.append(
                Candidate(
                    patch_a=a.patch_id,
                    patch_b=b.patch_id,
                    ratio_min_volume=ratio_min,
                    ratio_max_volume=ratio_max,
                    bbox_iou=iou,
                    centroid_distance=centroid_distance,
                    overlap_dims=dims,
                )
            )
    candidates.sort(key=lambda c: (c.ratio_min_volume, c.bbox_iou), reverse=True)
    return candidates[: args.max_candidates]


def sample_patch_indices(indices: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= max_points:
        return indices
    selected = rng.choice(len(indices), size=max_points, replace=False)
    return indices[selected]


def nearest_summary(
    xyz: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample_a = sample_patch_indices(idx_a, args.nn_sample_points, rng)
    sample_b = sample_patch_indices(idx_b, args.nn_sample_points, rng)
    if len(sample_a) == 0 or len(sample_b) == 0:
        return {"median_ab": float("inf"), "median_ba": float("inf"), "near_ratio_ab": 0.0, "near_ratio_ba": 0.0}
    tree_b = cKDTree(xyz[sample_b])
    dist_ab, _ = tree_b.query(xyz[sample_a], k=1, workers=-1)
    tree_a = cKDTree(xyz[sample_a])
    dist_ba, _ = tree_a.query(xyz[sample_b], k=1, workers=-1)
    return {
        "median_ab": float(np.median(dist_ab)),
        "median_ba": float(np.median(dist_ba)),
        "p25_ab": float(np.quantile(dist_ab, 0.25)),
        "p25_ba": float(np.quantile(dist_ba, 0.25)),
        "near_ratio_ab": float(np.mean(dist_ab <= args.near_distance)),
        "near_ratio_ba": float(np.mean(dist_ba <= args.near_distance)),
    }


def remap_merged_stats(anchor: PatchStats, other: PatchStats) -> PatchStats:
    total = anchor.count + other.count
    centroid = (anchor.centroid * anchor.count + other.centroid * other.count) / total
    mean_rgb = (anchor.mean_rgb * anchor.count + other.mean_rgb * other.count) / total
    normal = anchor.mean_normal * anchor.count + other.mean_normal * other.count
    norm = float(np.linalg.norm(normal))
    mean_normal = normal / norm if norm > 1e-9 else normal
    bucket_counts = Counter(anchor.bucket_counts)
    bucket_counts.update(other.bucket_counts)
    source_patch_ids = set(anchor.source_patch_ids)
    source_patch_ids.update(other.source_patch_ids)
    return PatchStats(
        patch_id=anchor.patch_id,
        count=total,
        centroid=centroid,
        mean_rgb=mean_rgb,
        mean_normal=mean_normal,
        bbox_min=np.minimum(anchor.bbox_min, other.bbox_min),
        bbox_max=np.maximum(anchor.bbox_max, other.bbox_max),
        bucket_counts=bucket_counts,
        geometry_type=dominant_geometry_name(bucket_counts),
        source_patch_ids=source_patch_ids,
    )


def dominant_geometry_name(bucket_counts: Counter[int]) -> str:
    from optimize_geo_patch_merges import dominant_geometry

    return dominant_geometry(bucket_counts)


def model_cohesion_score(a: PatchStats, b: PatchStats, nn: dict[str, float], args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    color_dist = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
    color = max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
    bucket = compatible_bucket_score(a.geometry_type, b.geometry_type)
    normal = normal_score(a.mean_normal, b.mean_normal)
    near = min(1.0, 0.5 * (nn["near_ratio_ab"] + nn["near_ratio_ba"]) / max(args.min_near_ratio, 1e-6))
    median = max(0.0, min(1.0, 1.0 - min(nn["median_ab"], nn["median_ba"]) / max(args.max_nn_median, 1e-6)))
    balance = min(a.count, b.count) / max(max(a.count, b.count), 1)
    score = 0.28 * color + 0.18 * bucket + 0.12 * normal + 0.24 * near + 0.12 * median + 0.06 * balance
    return score, {
        "color": color,
        "bucket": bucket,
        "normal": normal,
        "near": near,
        "median": median,
        "balance": balance,
        "color_dist": color_dist,
    }


def split_penalty(a: PatchStats, b: PatchStats, nn: dict[str, float], candidate: Candidate, args: argparse.Namespace) -> float:
    near_raw = 0.5 * (nn["near_ratio_ab"] + nn["near_ratio_ba"])
    median_raw = min(nn["median_ab"], nn["median_ba"])
    color_dist = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
    bucket = compatible_bucket_score(a.geometry_type, b.geometry_type)
    normal = normal_score(a.mean_normal, b.mean_normal)
    return (
        0.34 * min(1.0, near_raw / max(args.min_near_ratio, 1e-6))
        + 0.20 * max(0.0, min(1.0, 1.0 - median_raw / max(args.max_nn_median, 1e-6)))
        + 0.16 * max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
        + 0.12 * bucket
        + 0.08 * normal
        + 0.10 * min(1.0, candidate.ratio_min_volume)
    )


def merge_decision(
    a: PatchStats,
    b: PatchStats,
    candidate: Candidate,
    nn: dict[str, float],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float | str]]:
    merged = remap_merged_stats(a, b)
    merge_score, terms = model_cohesion_score(a, b, nn, args)
    separate_score = split_penalty(a, b, nn, candidate, args)
    # A geometry mismatch is acceptable for interleaved architectural structures
    # only if proximity and color support are strong.  Otherwise we keep the
    # patches separate even when their bounding boxes overlap.
    geometry_guard = 0.0
    if {a.geometry_type, b.geometry_type} in [{"horizontal", "vertical"}, {"vertical", "rough_mixed"}, {"horizontal", "rough_mixed"}]:
        geometry_guard = args.geometry_mismatch_penalty
    multimodal_score = None
    if args.allow_multimodal_merge:
        near_raw = 0.5 * (nn["near_ratio_ab"] + nn["near_ratio_ba"])
        bbox_score = min(1.0, 0.5 * (candidate.ratio_min_volume + candidate.bbox_iou / max(args.multimodal_bbox_iou_scale, 1e-6)))
        # Multi-modal objects are allowed to contain several geometry/color
        # modes.  We still require spatial interleaving, reasonable color
        # relation, and normal/chart compatibility; we simply stop treating a
        # bucket mismatch as a hard semantic penalty.
        multimodal_score = (
            0.36 * min(1.0, near_raw / max(args.min_near_ratio, 1e-6))
            + 0.22 * bbox_score
            + 0.18 * terms["normal"]
            + 0.14 * terms["color"]
            + 0.10 * terms["balance"]
        )
        merge_score = max(merge_score, multimodal_score)

    gain = merge_score - separate_score - geometry_guard
    near_raw = 0.5 * (nn["near_ratio_ab"] + nn["near_ratio_ba"])
    accept = (
        gain >= args.min_gain
        and merge_score >= args.min_merge_score
        and near_raw >= args.min_near_ratio
        and min(nn["median_ab"], nn["median_ba"]) <= args.max_nn_median
    )
    details: dict[str, float | str] = {
        "merge_score": merge_score,
        "separate_score": separate_score,
        "gain": gain,
        "geometry_guard": geometry_guard,
        "merged_geometry_type": merged.geometry_type,
        "near_ratio": near_raw,
        "multimodal_score": multimodal_score if multimodal_score is not None else -1.0,
        **terms,
        **nn,
    }
    return accept, details


def optimize_overlap_merges(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    stats = compute_patch_stats(arrays, labels)
    candidates = build_candidates(stats, args)
    label_to_indices: dict[int, np.ndarray] = {
        int(pid): np.flatnonzero(labels == int(pid)).astype(np.int64, copy=False) for pid in stats
    }
    dsu = DSU(list(stats))
    optimized_stats = {pid: s for pid, s in stats.items()}
    rng = np.random.default_rng(args.random_seed)
    merge_log: list[dict[str, Any]] = []
    reject_log: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()

    for candidate in candidates:
        root_a = dsu.find(candidate.patch_a)
        root_b = dsu.find(candidate.patch_b)
        if root_a == root_b or root_a not in optimized_stats or root_b not in optimized_stats:
            continue
        a = optimized_stats[root_a]
        b = optimized_stats[root_b]
        idx_a = np.concatenate([label_to_indices[pid] for pid in a.source_patch_ids if pid in label_to_indices])
        idx_b = np.concatenate([label_to_indices[pid] for pid in b.source_patch_ids if pid in label_to_indices])
        nn = nearest_summary(arrays["xyz"], idx_a, idx_b, args, rng)
        accepted, details = merge_decision(a, b, candidate, nn, args)
        if not accepted:
            reason = "score_low"
            if details["near_ratio"] < args.min_near_ratio:
                reason = "near_ratio_low"
            elif min(float(details["median_ab"]), float(details["median_ba"])) > args.max_nn_median:
                reason = "median_distance_high"
            reject_counts[reason] += 1
            if len(reject_log) < args.max_reject_log_rows:
                reject_log.append(
                    {
                        "reason": reason,
                        "patch_a": int(candidate.patch_a),
                        "patch_b": int(candidate.patch_b),
                        "current_root_a": int(root_a),
                        "current_root_b": int(root_b),
                        "voxels_a": int(a.count),
                        "voxels_b": int(b.count),
                        "ratio_min_volume": float(candidate.ratio_min_volume),
                        "ratio_max_volume": float(candidate.ratio_max_volume),
                        "bbox_iou": float(candidate.bbox_iou),
                        "centroid_distance": float(candidate.centroid_distance),
                        **details,
                    }
                )
            continue

        keep, drop = (root_a, root_b) if a.count >= b.count else (root_b, root_a)
        keep_stats = optimized_stats[keep]
        drop_stats = optimized_stats[drop]
        merged = remap_merged_stats(keep_stats, drop_stats)
        optimized_stats[keep] = merged
        del optimized_stats[drop]
        dsu.union(keep, drop)
        merge_log.append(
            {
                "keep_patch_id": int(keep),
                "drop_patch_id": int(drop),
                "candidate_patch_a": int(candidate.patch_a),
                "candidate_patch_b": int(candidate.patch_b),
                "voxels_keep_before": int(keep_stats.count),
                "voxels_drop_before": int(drop_stats.count),
                "voxels_after": int(merged.count),
                "ratio_min_volume": float(candidate.ratio_min_volume),
                "ratio_max_volume": float(candidate.ratio_max_volume),
                "bbox_iou": float(candidate.bbox_iou),
                "centroid_distance": float(candidate.centroid_distance),
                **details,
            }
        )

    remap_table = {pid: dsu.find(pid) for pid in stats}
    max_label = int(labels.max())
    table = np.arange(max_label + 1, dtype=np.int32)
    for old, new in remap_table.items():
        if old <= max_label:
            table[old] = new
    optimized_labels = table[labels]
    report = {
        "schema": "geo-patch-overlap-merge/v1",
        "input_patch_count": int(len(stats)),
        "candidate_count": int(len(candidates)),
        "accepted_merge_count": int(len(merge_log)),
        "output_patch_count": int(len(np.unique(optimized_labels))),
        "reject_counts": dict(reject_counts),
        "params": vars(args),
    }
    report["_reject_log"] = reject_log
    return optimized_labels, report, merge_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--min-patch-voxels", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=4000)
    parser.add_argument("--min-bbox-ratio", type=float, default=0.72)
    parser.add_argument("--min-bbox-iou", type=float, default=0.08)
    parser.add_argument("--max-centroid-distance", type=float, default=8.0)
    parser.add_argument("--near-distance", type=float, default=0.35)
    parser.add_argument("--min-near-ratio", type=float, default=0.35)
    parser.add_argument("--max-nn-median", type=float, default=0.45)
    parser.add_argument("--nn-sample-points", type=int, default=12000)
    parser.add_argument("--max-color-distance", type=float, default=95.0)
    parser.add_argument("--geometry-mismatch-penalty", type=float, default=0.06)
    parser.add_argument("--min-merge-score", type=float, default=0.50)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--allow-multimodal-merge", action="store_true")
    parser.add_argument("--multimodal-bbox-iou-scale", type=float, default=0.65)
    parser.add_argument("--max-reject-log-rows", type=int, default=20000)
    parser.add_argument("--small-patch-voxels", type=int, default=8)
    parser.add_argument("--preview-stride", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260623)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, _src, _dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"label count mismatch: labels={len(labels)} voxels={len(arrays['xyz'])}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimized, report, merge_log = optimize_overlap_merges(arrays, labels, args)
    report["output_ply"] = str(args.output_dir / f"geo_patches_overlap_merge_stride{args.preview_stride}.ply")
    report["output_jsonl"] = str(args.output_dir / "geo_patches_overlap_merge.jsonl")
    report["preview_points"] = write_ply(Path(report["output_ply"]), arrays, optimized, args.preview_stride)
    report["jsonl_patch_count"] = write_jsonl(Path(report["output_jsonl"]), arrays, optimized, args)
    reject_log = report.pop("_reject_log", [])
    (args.output_dir / "overlap_merge_log.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merge_log),
        encoding="utf-8",
    )
    (args.output_dir / "overlap_merge_reject_log.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reject_log),
        encoding="utf-8",
    )
    (args.output_dir / "overlap_merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
