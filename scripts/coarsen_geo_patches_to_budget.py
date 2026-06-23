#!/usr/bin/env python3
"""Coarsen conservative geo patches to a target patch budget.

The region model intentionally over-segments geometry into conservative patches.
That is useful for boundary safety, but too fine for semantic object reasoning.
This stage builds a larger-scale adjacency graph between patch centroids and
greedily merges compatible neighbors until the requested patch budget is met.

This is not a semantic classifier.  It produces coarse geometry/object
candidates with one label per voxel, preserving the invariant that each voxel
belongs to exactly one patch.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from optimize_geo_patch_merges import (
    BUCKET_NAMES,
    PatchStats,
    compatible_bucket_score,
    compute_patch_stats,
    dominant_geometry,
    normal_score,
    read_labels,
    read_region_input,
    write_ply,
)


STABLE_TYPES = {"horizontal", "vertical"}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


class DSU:
    def __init__(self, values: list[int]) -> None:
        self.parent = {int(v): int(v) for v in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, keep: int, drop: int) -> bool:
        keep_root = self.find(keep)
        drop_root = self.find(drop)
        if keep_root == drop_root:
            return False
        self.parent[drop_root] = keep_root
        return True


def bbox_volume(stats: PatchStats) -> float:
    return float(np.prod(np.maximum(stats.bbox_max - stats.bbox_min, 1e-3)))


def bbox_gap(a: PatchStats, b: PatchStats) -> float:
    gap = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    return float(np.linalg.norm(gap))


def merged_extent(a: PatchStats, b: PatchStats) -> np.ndarray:
    return np.maximum(a.bbox_max, b.bbox_max) - np.minimum(a.bbox_min, b.bbox_min)


def aspect_ratio(extent: np.ndarray) -> float:
    extent = np.maximum(extent, 1e-3)
    return float(np.max(extent) / max(float(np.min(extent)), 1e-3))


def merge_stats(keep_id: int, a: PatchStats, b: PatchStats) -> PatchStats:
    total = a.count + b.count
    centroid = (a.centroid * a.count + b.centroid * b.count) / total
    mean_rgb = (a.mean_rgb * a.count + b.mean_rgb * b.count) / total
    normal = a.mean_normal * a.count + b.mean_normal * b.count
    norm = float(np.linalg.norm(normal))
    mean_normal = normal / norm if norm > 1e-9 else normal
    bucket_counts = Counter(a.bucket_counts)
    bucket_counts.update(b.bucket_counts)
    source_patch_ids = set(a.source_patch_ids)
    source_patch_ids.update(b.source_patch_ids)
    return PatchStats(
        patch_id=int(keep_id),
        count=int(total),
        centroid=centroid,
        mean_rgb=mean_rgb,
        mean_normal=mean_normal,
        bbox_min=np.minimum(a.bbox_min, b.bbox_min),
        bbox_max=np.maximum(a.bbox_max, b.bbox_max),
        bucket_counts=bucket_counts,
        geometry_type=dominant_geometry(bucket_counts),
        source_patch_ids=source_patch_ids,
    )


def score_pair(a: PatchStats, b: PatchStats, args: argparse.Namespace) -> tuple[float, dict[str, float | str]]:
    centroid_dist = float(np.linalg.norm(a.centroid - b.centroid))
    gap = bbox_gap(a, b)
    color_dist = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
    color = max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
    bucket = compatible_bucket_score(a.geometry_type, b.geometry_type)
    normal = normal_score(a.mean_normal, b.mean_normal)
    dist_score = max(0.0, min(1.0, 1.0 - centroid_dist / max(args.max_centroid_distance, 1e-6)))
    gap_score = max(0.0, min(1.0, 1.0 - gap / max(args.max_bbox_gap, 1e-6)))
    balance = min(a.count, b.count) / max(float(max(a.count, b.count)), 1.0)
    extent = merged_extent(a, b)
    max_extent = float(np.max(extent))
    aspect = aspect_ratio(extent)

    stable_mismatch = a.geometry_type in STABLE_TYPES and b.geometry_type in STABLE_TYPES and a.geometry_type != b.geometry_type
    hard_color = color_dist > args.hard_color_distance and not ({a.geometry_type, b.geometry_type} <= {"rough_mixed", "thin_linear", "unknown", "mixed"})
    over_extent = max_extent > args.max_merged_extent and aspect > args.max_merged_aspect

    score = (
        args.color_weight * color
        + args.bucket_weight * bucket
        + args.normal_weight * normal
        + args.distance_weight * dist_score
        + args.gap_weight * gap_score
        + args.balance_weight * min(1.0, balance * 8.0)
    )
    penalty = 0.0
    if stable_mismatch:
        penalty += args.stable_mismatch_penalty
    if hard_color:
        penalty += args.hard_color_penalty
    if over_extent:
        penalty += args.over_extent_penalty
    score -= penalty
    return score, {
        "centroid_dist": centroid_dist,
        "gap": gap,
        "color": color,
        "color_dist": color_dist,
        "bucket": bucket,
        "normal": normal,
        "dist_score": dist_score,
        "gap_score": gap_score,
        "balance": balance,
        "max_extent": max_extent,
        "aspect": aspect,
        "penalty": penalty,
        "geom_a": a.geometry_type,
        "geom_b": b.geometry_type,
    }


def build_grid_candidates(stats: dict[int, PatchStats], args: argparse.Namespace) -> set[tuple[int, int]]:
    ids = np.array(sorted(stats), dtype=np.int64)
    centroids = np.stack([stats[int(pid)].centroid for pid in ids], axis=0)
    tree = cKDTree(centroids)
    k = min(args.neighbors_per_patch + 1, len(ids))
    distances, indices = tree.query(centroids, k=k, distance_upper_bound=args.max_centroid_distance, workers=-1)
    pairs: set[tuple[int, int]] = set()
    for row_idx, patch_id in enumerate(ids.tolist()):
        for dist, nbr_idx in zip(np.atleast_1d(distances[row_idx]).tolist(), np.atleast_1d(indices[row_idx]).tolist(), strict=True):
            if not np.isfinite(dist) or nbr_idx >= len(ids):
                continue
            other = int(ids[int(nbr_idx)])
            if patch_id == other:
                continue
            a, b = sorted((int(patch_id), other))
            pairs.add((a, b))
    return pairs


def write_component_jsonl(path: Path, component_stats: dict[int, PatchStats], args: argparse.Namespace) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for patch_id in sorted(component_stats):
            s = component_stats[patch_id]
            source_patch_ids = sorted(int(v) for v in s.source_patch_ids)
            row = {
                "patch_id": int(patch_id),
                "object": int(patch_id),
                "voxel_count": int(s.count),
                "status": "coarse_geo_patch",
                "geometry_type": s.geometry_type,
                "semantic_label": s.geometry_type,
                "description": f"coarse geometry patch: {s.geometry_type}",
                "bucket_counts": {BUCKET_NAMES[int(k)]: int(v) for k, v in s.bucket_counts.items()},
                "centroid": s.centroid.astype(float).tolist(),
                "bbox_3d": {"min": s.bbox_min.astype(float).tolist(), "max": s.bbox_max.astype(float).tolist()},
                "extent": (s.bbox_max - s.bbox_min).astype(float).tolist(),
                "mean_rgb": s.mean_rgb.astype(float).tolist(),
                "mean_normal": s.mean_normal.astype(float).tolist(),
                "source_patch_count": len(source_patch_ids),
                "source_patch_ids": source_patch_ids[: args.max_source_patch_ids],
                "source_patch_ids_truncated": len(source_patch_ids) > args.max_source_patch_ids,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(component_stats)


def remap_labels(labels: np.ndarray, stats: dict[int, PatchStats], dsu: DSU, noise_source_ids: set[int], noise_id: int) -> np.ndarray:
    table = np.arange(noise_id + 1, dtype=np.int32)
    for patch_id in stats:
        if patch_id in noise_source_ids:
            table[patch_id] = noise_id
        else:
            table[patch_id] = dsu.find(patch_id)
    return table[labels]


def coarse_cell_pair_counts(
    xyz: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
    max_labels_per_cell: int,
) -> tuple[Counter[tuple[int, int]], Counter[int]]:
    origin = np.floor(xyz.min(axis=0) / voxel_size).astype(np.int64)
    grid = np.floor(xyz / voxel_size).astype(np.int64) - origin
    shape = grid.max(axis=0).astype(np.int64) + 1
    ny = int(shape[1])
    nz = int(shape[2])
    keys = (grid[:, 0] * ny + grid[:, 1]) * nz + grid[:, 2]
    order = np.lexsort((labels, keys))
    sorted_keys = keys[order]
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    ends = np.r_[starts[1:], len(sorted_keys)]
    pair_counts: Counter[tuple[int, int]] = Counter()
    cell_counts: Counter[int] = Counter()
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        unique_labels = np.unique(sorted_labels[start:end]).astype(np.int64, copy=False).tolist()
        if not unique_labels:
            continue
        for label in unique_labels:
            cell_counts[int(label)] += 1
        if len(unique_labels) < 2 or len(unique_labels) > max_labels_per_cell:
            continue
        for i, a in enumerate(unique_labels):
            for b in unique_labels[i + 1 :]:
                pair_counts[(int(a), int(b))] += 1
    return pair_counts, cell_counts


def suppress_overlap(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    component_stats: dict[int, PatchStats],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[int, PatchStats], dict[str, Any], list[dict[str, Any]]]:
    if args.overlap_voxel_size <= 0 or args.overlap_merge_passes <= 0:
        return labels, component_stats, {"enabled": False}, []

    total_merges = 0
    logs: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "enabled": True,
        "voxel_size": args.overlap_voxel_size,
        "passes": [],
    }
    current_labels = labels
    current_stats = component_stats

    for pass_idx in range(args.overlap_merge_passes):
        pair_counts, cell_counts = coarse_cell_pair_counts(
            arrays["xyz"],
            current_labels,
            args.overlap_voxel_size,
            args.overlap_max_labels_per_cell,
        )
        candidates: list[tuple[float, int, int, int, float, dict[str, float | str]]] = []
        for (a, b), shared in pair_counts.items():
            if a not in current_stats or b not in current_stats:
                continue
            if a == b:
                continue
            if current_stats[a].geometry_type == "noise_residual" or current_stats[b].geometry_type == "noise_residual":
                continue
            geom_a = current_stats[a].geometry_type
            geom_b = current_stats[b].geometry_type
            if args.overlap_block_stable_mismatch and geom_a in STABLE_TYPES and geom_b in STABLE_TYPES and geom_a != geom_b:
                continue
            if args.overlap_block_stable_rough:
                rough_types = {"rough_mixed", "thin_linear", "unknown", "mixed"}
                stable_rough = (geom_a in STABLE_TYPES and geom_b in rough_types) or (geom_b in STABLE_TYPES and geom_a in rough_types)
                if stable_rough:
                    continue
            if args.overlap_same_geometry_only and geom_a != geom_b:
                continue
            if float(np.max(merged_extent(current_stats[a], current_stats[b]))) > args.overlap_hard_max_extent:
                continue
            min_cells = min(cell_counts.get(a, 0), cell_counts.get(b, 0))
            if shared < args.overlap_min_cells or min_cells <= 0:
                continue
            overlap_ratio = shared / max(float(min_cells), 1.0)
            if overlap_ratio < args.overlap_min_ratio:
                continue
            score, features = score_pair(current_stats[a], current_stats[b], args)
            overlap_boosted_score = score + args.overlap_ratio_weight * min(1.0, overlap_ratio)
            if overlap_boosted_score < args.overlap_min_score:
                continue
            candidates.append((overlap_boosted_score, int(a), int(b), int(shared), float(overlap_ratio), features))
        candidates.sort(reverse=True)

        dsu = DSU(list(current_stats))
        next_stats = dict(current_stats)
        pass_merges = 0
        for score, a, b, shared, overlap_ratio, features in candidates:
            ra = dsu.find(a)
            rb = dsu.find(b)
            if ra == rb or ra not in next_stats or rb not in next_stats:
                continue
            if next_stats[ra].count + next_stats[rb].count > args.overlap_max_component_voxels:
                continue
            keep, drop = (ra, rb) if next_stats[ra].count >= next_stats[rb].count else (rb, ra)
            if not dsu.union(keep, drop):
                continue
            next_stats[keep] = merge_stats(keep, next_stats[keep], next_stats[drop])
            next_stats.pop(drop, None)
            pass_merges += 1
            total_merges += 1
            if len(logs) < args.max_log_rows:
                logs.append(
                    {
                        "pass": pass_idx,
                        "keep": int(keep),
                        "drop": int(drop),
                        "score": float(score),
                        "shared_overlap_cells": int(shared),
                        "overlap_ratio_min": float(overlap_ratio),
                        **features,
                    }
                )
        if pass_merges == 0:
            report["passes"].append(
                {
                    "pass": pass_idx,
                    "candidate_pair_count": len(candidates),
                    "merge_count": 0,
                    "patch_count": len(current_stats),
                }
            )
            break
        max_label = int(current_labels.max())
        table = np.arange(max(max_label, max(current_stats)) + 1, dtype=np.int32)
        for label in current_stats:
            if label < len(table):
                table[label] = dsu.find(label)
        current_labels = table[current_labels]
        current_stats = next_stats
        report["passes"].append(
            {
                "pass": pass_idx,
                "candidate_pair_count": len(candidates),
                "merge_count": pass_merges,
                "patch_count": len(current_stats),
            }
        )

    report["merge_count"] = total_merges
    report["output_patch_count"] = len(current_stats)
    return current_labels, current_stats, report, logs


def split_large_components(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    component_stats: dict[int, PatchStats],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[int, PatchStats], dict[str, Any]]:
    if not args.split_large_patches:
        return labels, component_stats, {"enabled": False}

    split_ids = []
    for patch_id, stats in component_stats.items():
        if stats.geometry_type == "noise_residual":
            continue
        extent = stats.bbox_max - stats.bbox_min
        if stats.count >= args.split_large_min_voxels or float(np.max(extent)) >= args.split_large_min_extent:
            split_ids.append(int(patch_id))

    if not split_ids:
        return labels, component_stats, {
            "enabled": True,
            "candidate_patch_count": 0,
            "split_patch_count": 0,
            "new_patch_count": int(len(component_stats)),
        }

    next_labels = labels.copy()
    next_id = int(max(int(labels.max()), max(component_stats))) + 1
    rows: list[dict[str, Any]] = []
    xyz = arrays["xyz"]
    buckets = arrays["buckets"].astype(np.int64, copy=False)

    for patch_id in split_ids:
        stats = component_stats.get(patch_id)
        if stats is None:
            continue
        indices = np.flatnonzero(next_labels == patch_id)
        if len(indices) == 0:
            continue
        local_xyz = xyz[indices]
        cell = np.floor((local_xyz - stats.bbox_min) / max(args.split_large_cell_size, 1e-6)).astype(np.int64)
        cell -= cell.min(axis=0)
        span = cell.max(axis=0) + 1
        keys = (cell[:, 0] * max(int(span[1]), 1) + cell[:, 1]) * max(int(span[2]), 1) + cell[:, 2]
        if args.split_large_by_bucket:
            keys = keys * 16 + buckets[indices]
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        if len(unique_keys) <= 1:
            continue
        keep_group = int(np.argmax(counts))
        created = 0
        for group_idx in range(len(unique_keys)):
            group_mask = inverse == group_idx
            group_count = int(np.count_nonzero(group_mask))
            if group_count < args.split_large_min_child_voxels:
                continue
            if group_idx == keep_group:
                continue
            next_labels[indices[group_mask]] = next_id
            next_id += 1
            created += 1
        if created:
            rows.append(
                {
                    "patch_id": int(patch_id),
                    "old_voxels": int(stats.count),
                    "old_geometry_type": stats.geometry_type,
                    "old_extent": (stats.bbox_max - stats.bbox_min).astype(float).tolist(),
                    "raw_group_count": int(len(unique_keys)),
                    "created_child_count": int(created),
                }
            )

    next_stats = compute_patch_stats(arrays, next_labels)
    report = {
        "enabled": True,
        "cell_size": float(args.split_large_cell_size),
        "by_bucket": bool(args.split_large_by_bucket),
        "min_voxels": int(args.split_large_min_voxels),
        "min_extent": float(args.split_large_min_extent),
        "min_child_voxels": int(args.split_large_min_child_voxels),
        "candidate_patch_count": int(len(split_ids)),
        "split_patch_count": int(len(rows)),
        "created_child_count": int(sum(row["created_child_count"] for row in rows)),
        "old_patch_count": int(len(component_stats)),
        "new_patch_count": int(len(next_stats)),
        "rows": rows[: args.max_log_rows],
    }
    return next_labels, next_stats, report


def coarsen(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    stats: dict[int, PatchStats],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[int, PatchStats], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    max_input_label = int(labels.max())
    noise_id = max_input_label + 1
    noise_source_ids = {
        patch_id
        for patch_id, patch_stats in stats.items()
        if patch_stats.count <= args.noise_patch_voxels
    }
    precollapsed_noise_id = getattr(args, "precollapsed_noise_id", None)
    if precollapsed_noise_id is not None and int(precollapsed_noise_id) in stats:
        noise_source_ids.add(int(precollapsed_noise_id))
    active = {patch_id: patch_stats for patch_id, patch_stats in stats.items() if patch_id not in noise_source_ids}
    if not active:
        raise ValueError("all patches were filtered as noise; lower --noise-patch-voxels")
    dsu = DSU(list(stats) + [noise_id])
    pairs = build_grid_candidates(active, args)
    heap: list[tuple[float, int, int]] = []
    for a, b in pairs:
        score, _ = score_pair(active[a], active[b], args)
        if score >= args.min_merge_score:
            heapq.heappush(heap, (-score, a, b))

    merge_log: list[dict[str, Any]] = []
    stale = 0
    rejected = Counter()
    evaluated = 0

    effective_target = max(1, args.target_patches - (1 if noise_source_ids else 0))
    while heap and len(active) > effective_target:
        _, a, b = heapq.heappop(heap)
        ra = dsu.find(a)
        rb = dsu.find(b)
        if ra == rb or ra not in active or rb not in active:
            stale += 1
            continue
        score, features = score_pair(active[ra], active[rb], args)
        evaluated += 1
        if score < args.min_merge_score:
            rejected["low_score"] += 1
            continue
        if active[ra].count + active[rb].count > args.max_component_voxels:
            rejected["max_component_voxels"] += 1
            continue
        keep, drop = (ra, rb) if active[ra].count >= active[rb].count else (rb, ra)
        if not dsu.union(keep, drop):
            continue
        active[keep] = merge_stats(keep, active[keep], active[drop])
        active.pop(drop, None)
        if len(merge_log) < args.max_log_rows:
            merge_log.append({"keep": int(keep), "drop": int(drop), "score": float(score), **features})

    out = remap_labels(labels, stats, dsu, noise_source_ids, noise_id)
    if noise_source_ids:
        noise_stats: PatchStats | None = None
        for patch_id in noise_source_ids:
            patch_stats = stats[patch_id]
            noise_stats = patch_stats if noise_stats is None else merge_stats(noise_id, noise_stats, patch_stats)
        if noise_stats is not None:
            noise_stats.patch_id = noise_id
            noise_stats.geometry_type = "noise_residual"
            active[noise_id] = noise_stats
    out, active, overlap_report, overlap_log = suppress_overlap(arrays=arrays, labels=out, component_stats=active, args=args)
    out, active, split_report = split_large_components(arrays=arrays, labels=out, component_stats=active, args=args)
    report = {
        "schema": "geo-patch-coarsen-budget/v1",
        "input_patch_count": int(len(stats)),
        "output_patch_count": int(len(active)),
        "effective_output_patch_count": int(len(active) - (1 if noise_source_ids else 0)),
        "target_patches": int(args.target_patches),
        "effective_target_patches": int(effective_target),
        "noise_patch_count": int(len(noise_source_ids)),
        "noise_voxel_count": int(sum(stats[pid].count for pid in noise_source_ids)),
        "noise_patch_id": int(noise_id) if noise_source_ids else None,
        "initial_candidate_pairs": int(len(pairs)),
        "merge_count": int(len(merge_log)),
        "stale_heap_pops": int(stale),
        "evaluated_edges": int(evaluated),
        "rejected": dict(rejected),
        "overlap_suppression": overlap_report,
        "large_patch_split": split_report,
        "params": vars(args),
    }
    return out, active, report, merge_log, overlap_log


def precollapse_tiny_patches(
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    arrays: dict[str, np.ndarray],
    threshold: int,
    mode: str,
    grid_size: float,
    min_component_voxels: int,
    compatible_color_distance: float,
    compatible_normal_dot: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if threshold <= 0:
        return labels, {"enabled": False}
    max_label = int(labels.max())
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=max_label + 1)
    present = counts > 0
    tiny = present & (counts <= threshold)
    tiny_ids = np.flatnonzero(tiny)
    if len(tiny_ids) == 0:
        return labels, {
            "enabled": True,
            "threshold": int(threshold),
            "tiny_patch_count": 0,
            "tiny_voxel_count": 0,
            "noise_patch_id": None,
        }
    if mode == "global":
        noise_id = max_label + 1
        table = np.arange(noise_id + 1, dtype=np.int32)
        table[tiny_ids] = noise_id
        out = table[labels]
        return out, {
            "enabled": True,
            "mode": mode,
            "threshold": int(threshold),
            "tiny_patch_count": int(len(tiny_ids)),
            "tiny_voxel_count": int(counts[tiny_ids].sum()),
            "noise_patch_id": int(noise_id),
            "residual_component_count": 1,
            "grid_residual_count": 0,
        }

    tiny_mask = np.zeros(max_label + 1, dtype=bool)
    tiny_mask[tiny_ids] = True
    parent = np.arange(max_label + 1, dtype=np.int32)
    rank = np.zeros(max_label + 1, dtype=np.uint8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    edge_a = labels[src]
    edge_b = labels[dst]
    tiny_edges = tiny_mask[edge_a] & tiny_mask[edge_b] & (edge_a != edge_b)
    if mode == "connected-compatible":
        rgb = arrays["rgb"].astype(np.float32, copy=False)
        normal = arrays["normal"].astype(np.float32, copy=False)
        buckets = arrays["buckets"].astype(np.int16, copy=False)
        color_dist = np.linalg.norm(rgb[src] - rgb[dst], axis=1)
        normal_dot = np.abs(np.sum(normal[src] * normal[dst], axis=1))
        same_bucket = buckets[src] == buckets[dst]
        flexible_bucket = (
            np.isin(buckets[src], [0, 3, 4])
            & np.isin(buckets[dst], [0, 3, 4])
        )
        compatible = (
            (color_dist <= compatible_color_distance)
            & ((normal_dot >= compatible_normal_dot) | flexible_bucket)
            & (same_bucket | flexible_bucket)
        )
        tiny_edges &= compatible
    for a, b in zip(edge_a[tiny_edges].tolist(), edge_b[tiny_edges].tolist(), strict=True):
        union(int(a), int(b))

    roots = np.fromiter((find(int(label)) for label in tiny_ids.tolist()), dtype=np.int32, count=len(tiny_ids))
    component_voxels: Counter[int] = Counter()
    for label, root in zip(tiny_ids.tolist(), roots.tolist(), strict=True):
        component_voxels[int(root)] += int(counts[int(label)])

    next_id = max_label + 1
    root_to_new: dict[int, int] = {}
    table = np.arange(max_label + 1, dtype=np.int32)
    small_tiny_label_mask = np.zeros(max_label + 1, dtype=bool)
    connected_component_count = 0
    small_component_label_count = 0
    small_component_voxel_count = 0

    for label, root in zip(tiny_ids.tolist(), roots.tolist(), strict=True):
        root = int(root)
        comp_voxels = int(component_voxels[root])
        if comp_voxels >= min_component_voxels:
            if root not in root_to_new:
                root_to_new[root] = next_id
                next_id += 1
                connected_component_count += 1
            table[int(label)] = root_to_new[root]
        else:
            small_component_label_count += 1
            small_component_voxel_count += int(counts[int(label)])
            small_tiny_label_mask[int(label)] = True

    small_label_mask = small_tiny_label_mask[labels]
    grid_residual_count = 0
    if np.any(small_label_mask):
        if mode == "connected":
            # Preserve exact connected components. This can be very fragmented
            # but is useful for diagnostics.
            for label, root in zip(tiny_ids.tolist(), roots.tolist(), strict=True):
                if component_voxels[int(root)] >= min_component_voxels:
                    continue
                if int(root) not in root_to_new:
                    root_to_new[int(root)] = next_id
                    next_id += 1
                    connected_component_count += 1
                table[int(label)] = root_to_new[int(root)]
        elif mode in {"connected-grid", "connected-compatible"}:
            small_indices = np.flatnonzero(small_label_mask)
            xyz = arrays["xyz"]
            small_xyz = xyz[small_indices].astype(np.float64, copy=False)
            origin = np.floor(xyz.min(axis=0) / grid_size).astype(np.int64)
            grid = np.floor(small_xyz / max(grid_size, 1e-6)).astype(np.int64) - origin
            shape = grid.max(axis=0).astype(np.int64) + 1
            ny = int(max(shape[1], 1))
            nz = int(max(shape[2], 1))
            grid_keys = (grid[:, 0] * ny + grid[:, 1]) * nz + grid[:, 2]
            unique_keys = np.unique(grid_keys)
            key_to_new = {int(key): int(next_id + i) for i, key in enumerate(unique_keys.tolist())}
            next_id += len(unique_keys)
            grid_residual_count = int(len(unique_keys))
            small_labels = labels[small_indices]
            # Assign all tiny labels that appear in a local grid cell to that
            # local residual id.  A tiny label may span multiple cells; the
            # first observed local bucket is sufficient because these labels
            # are already below the tiny threshold.
            label_to_grid: dict[int, int] = {}
            for label, key in zip(small_labels.tolist(), grid_keys.tolist(), strict=True):
                label_to_grid.setdefault(int(label), key_to_new[int(key)])
            for label, new_id in label_to_grid.items():
                table[int(label)] = int(new_id)
        else:
            raise ValueError(f"unsupported --precollapse-mode: {mode}")

    out = table[labels]
    residual_ids = np.unique(table[tiny_ids])
    return out, {
        "enabled": True,
        "mode": mode,
        "threshold": int(threshold),
        "tiny_patch_count": int(len(tiny_ids)),
        "tiny_voxel_count": int(counts[tiny_ids].sum()),
        "noise_patch_id": None,
        "residual_component_count": int(len(residual_ids)),
        "connected_component_count": int(connected_component_count),
        "grid_residual_count": int(grid_residual_count),
        "small_component_label_count": int(small_component_label_count),
        "small_component_voxel_count": int(small_component_voxel_count),
        "min_component_voxels": int(min_component_voxels),
        "grid_size": float(grid_size),
        "compatible_color_distance": float(compatible_color_distance),
        "compatible_normal_dot": float(compatible_normal_dot),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-patches", type=int, default=1000)
    parser.add_argument("--noise-patch-voxels", type=int, default=0, help="Collapse source patches at or below this voxel count into one noise patch before coarsening")
    parser.add_argument("--precollapse-noise", action=argparse.BooleanOptionalAction, default=True, help="Vector-collapse tiny source labels before expensive per-patch statistics")
    parser.add_argument("--precollapse-mode", choices=["global", "connected", "connected-grid", "connected-compatible"], default="global")
    parser.add_argument("--precollapse-grid-size", type=float, default=0.50)
    parser.add_argument("--precollapse-min-component-voxels", type=int, default=24)
    parser.add_argument("--precollapse-compatible-color-distance", type=float, default=95.0)
    parser.add_argument("--precollapse-compatible-normal-dot", type=float, default=0.30)
    parser.add_argument("--grid-cell-size", type=float, default=2.0, help="Deprecated; kept for CLI compatibility")
    parser.add_argument("--neighbors-per-patch", type=int, default=12)
    parser.add_argument("--max-centroid-distance", type=float, default=3.0)
    parser.add_argument("--max-bbox-gap", type=float, default=0.35)
    parser.add_argument("--min-merge-score", type=float, default=0.48)
    parser.add_argument("--max-color-distance", type=float, default=180.0)
    parser.add_argument("--hard-color-distance", type=float, default=210.0)
    parser.add_argument("--max-merged-extent", type=float, default=28.0)
    parser.add_argument("--max-merged-aspect", type=float, default=18.0)
    parser.add_argument("--max-component-voxels", type=int, default=800000)
    parser.add_argument("--color-weight", type=float, default=0.24)
    parser.add_argument("--bucket-weight", type=float, default=0.22)
    parser.add_argument("--normal-weight", type=float, default=0.12)
    parser.add_argument("--distance-weight", type=float, default=0.18)
    parser.add_argument("--gap-weight", type=float, default=0.18)
    parser.add_argument("--balance-weight", type=float, default=0.06)
    parser.add_argument("--stable-mismatch-penalty", type=float, default=0.20)
    parser.add_argument("--hard-color-penalty", type=float, default=0.16)
    parser.add_argument("--over-extent-penalty", type=float, default=0.18)
    parser.add_argument("--preview-stride", type=int, default=10)
    parser.add_argument("--max-log-rows", type=int, default=50000)
    parser.add_argument("--max-source-patch-ids", type=int, default=256)
    parser.add_argument("--overlap-voxel-size", type=float, default=0.0)
    parser.add_argument("--overlap-merge-passes", type=int, default=1)
    parser.add_argument("--overlap-max-labels-per-cell", type=int, default=8)
    parser.add_argument("--overlap-min-cells", type=int, default=4)
    parser.add_argument("--overlap-min-ratio", type=float, default=0.18)
    parser.add_argument("--overlap-ratio-weight", type=float, default=0.24)
    parser.add_argument("--overlap-min-score", type=float, default=0.48)
    parser.add_argument("--overlap-max-component-voxels", type=int, default=8000000)
    parser.add_argument("--overlap-hard-max-extent", type=float, default=1e9)
    parser.add_argument("--overlap-block-stable-mismatch", action="store_true")
    parser.add_argument("--overlap-block-stable-rough", action="store_true")
    parser.add_argument("--overlap-same-geometry-only", action="store_true")
    parser.add_argument("--split-large-patches", action="store_true")
    parser.add_argument("--split-large-min-voxels", type=int, default=500000)
    parser.add_argument("--split-large-min-extent", type=float, default=45.0)
    parser.add_argument("--split-large-cell-size", type=float, default=12.0)
    parser.add_argument("--split-large-min-child-voxels", type=int, default=2000)
    parser.add_argument("--split-large-by-bucket", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_patches <= 0:
        raise ValueError("--target-patches must be positive")
    log(f"read region input: {args.region_input}")
    arrays, src, dst = read_region_input(args.region_input)
    log(f"read labels: {args.labels}")
    labels = read_labels(args.labels)
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"label count mismatch: labels={len(labels)} voxels={len(arrays['xyz'])}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    precollapse_report = {"enabled": False}
    if args.precollapse_noise and args.noise_patch_voxels > 0:
        log(f"precollapse tiny patches <= {args.noise_patch_voxels} voxels")
        labels, precollapse_report = precollapse_tiny_patches(
            labels=labels,
            src=src,
            dst=dst,
            arrays=arrays,
            threshold=args.noise_patch_voxels,
            mode=args.precollapse_mode,
            grid_size=args.precollapse_grid_size,
            min_component_voxels=args.precollapse_min_component_voxels,
            compatible_color_distance=args.precollapse_compatible_color_distance,
            compatible_normal_dot=args.precollapse_compatible_normal_dot,
        )
        # The labels now contain explicit residual components, so the slower
        # stats-level tiny-patch collapse is no longer needed unless the mode
        # intentionally produced one global residual.
        args.noise_patch_voxels = 0
        args.precollapsed_noise_id = precollapse_report.get("noise_patch_id")
        log(
            "precollapse done: "
            f"tiny_patches={precollapse_report.get('tiny_patch_count')} "
            f"tiny_voxels={precollapse_report.get('tiny_voxel_count')}"
        )
    log("compute patch stats")
    stats = compute_patch_stats(arrays, labels)
    log(f"coarsen active stats: {len(stats)} patches")
    out, component_stats, report, merge_log, overlap_log = coarsen(arrays, labels, stats, args)
    report["precollapse_noise"] = precollapse_report
    log(f"coarsen done: output_patches={len(component_stats)}")
    report["output_ply"] = str(args.output_dir / f"geo_patches_coarse_stride{args.preview_stride}.ply")
    report["output_jsonl"] = str(args.output_dir / "geo_patches_coarse.jsonl")
    log(f"write preview ply: {report['output_ply']}")
    report["preview_points"] = write_ply(Path(report["output_ply"]), arrays, out, args.preview_stride)
    log(f"write jsonl: {report['output_jsonl']}")
    report["jsonl_patch_count"] = write_component_jsonl(Path(report["output_jsonl"]), component_stats, args)
    (args.output_dir / "coarse_merge_log.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merge_log),
        encoding="utf-8",
    )
    (args.output_dir / "overlap_suppression_log.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in overlap_log),
        encoding="utf-8",
    )
    (args.output_dir / "coarse_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
