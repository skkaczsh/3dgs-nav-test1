#!/usr/bin/env python3
"""Resolve patch boundary conflicts by assigning one owner per fine voxel.

This optimizer adjusts boundary ownership directly.  It starts from conservative
C++ superpatch labels, finds fine spatial cells occupied by multiple patches,
and transfers the cell to the patch model with the best local evidence.  It is
intentionally limited to conflict fine cells for v1; it does not yet do global
large-large merging.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from optimize_geo_patch_merges import (
    BUCKET_NAMES,
    PatchStats,
    compatible_bucket_score,
    compute_patch_stats,
    normal_score,
    read_labels,
    read_region_input,
    write_jsonl,
    write_ply,
)


def dominant_bucket_name(buckets: np.ndarray) -> str:
    if len(buckets) == 0:
        return "unknown"
    bucket, count = Counter(int(v) for v in buckets.tolist()).most_common(1)[0]
    if count / max(len(buckets), 1) < 0.55:
        return "mixed"
    return BUCKET_NAMES.get(bucket, "unknown")


def normalized_mean_normal(normals: np.ndarray) -> np.ndarray:
    if len(normals) == 0:
        return np.zeros(3, dtype=np.float64)
    normal = normals.astype(np.float64, copy=False).mean(axis=0)
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 1e-9 else normal


def cell_patch_score(
    stats: PatchStats,
    cell_rgb: np.ndarray,
    cell_normal: np.ndarray,
    cell_geometry: str,
    cell_count_for_patch: int,
    cell_total: int,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    color_dist = float(np.linalg.norm(stats.mean_rgb - cell_rgb))
    color = max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
    normal = normal_score(stats.mean_normal, cell_normal)
    bucket = compatible_bucket_score(stats.geometry_type, cell_geometry)
    occupancy = cell_count_for_patch / max(float(cell_total), 1.0)
    size = min(1.0, np.log1p(stats.count) / np.log1p(args.size_prior_voxels))
    stable_guard = 0.0
    if stats.geometry_type in {"horizontal", "vertical"} and cell_geometry not in {stats.geometry_type, "unknown", "mixed"}:
        stable_guard = 0.12
    score = (
        0.34 * color
        + 0.24 * bucket
        + 0.18 * normal
        + 0.17 * occupancy
        + 0.07 * size
        - stable_guard
    )
    return score, {
        "score": score,
        "color": color,
        "bucket": bucket,
        "normal": normal,
        "occupancy": occupancy,
        "size": size,
        "stable_guard": stable_guard,
    }


def sorted_fine_groups(xyz: np.ndarray, fine_voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.rint(xyz.astype(np.float64, copy=False) / fine_voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[starts[1:], len(order)]
    return order, starts, ends


def conflict_summary(labels: np.ndarray, xyz: np.ndarray, voxel_size: float) -> dict[str, Any]:
    order, starts, ends = sorted_fine_groups(xyz, voxel_size)
    sorted_labels = labels[order]
    conflict_cells = 0
    conflict_rows = 0
    conflict_extra = 0
    for start, end in zip(starts, ends, strict=True):
        group = sorted_labels[start:end]
        if len(group) <= 1:
            continue
        counts = Counter(int(v) for v in group.tolist())
        if len(counts) <= 1:
            continue
        total = int(end - start)
        conflict_cells += 1
        conflict_rows += total
        conflict_extra += total - counts.most_common(1)[0][1]
    return {
        "voxel_size": voxel_size,
        "conflict_cell_count": conflict_cells,
        "conflict_rows": conflict_rows,
        "conflict_extra_rows": conflict_extra,
        "conflict_row_ratio": conflict_rows / max(float(len(labels)), 1.0),
        "conflict_extra_ratio": conflict_extra / max(float(len(labels)), 1.0),
    }


def optimize_boundaries(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    stats = compute_patch_stats(arrays, labels)
    order, starts, ends = sorted_fine_groups(arrays["xyz"], args.fine_voxel_size)
    sorted_labels = labels[order]
    optimized = labels.copy()
    transfer_log: list[dict[str, Any]] = []
    conflict_cells = 0
    resolved_cells = 0
    transferred_points = 0

    for start, end in zip(starts, ends, strict=True):
        idx = order[start:end]
        group_labels = sorted_labels[start:end]
        if len(group_labels) <= 1:
            continue
        counts = Counter(int(v) for v in group_labels.tolist())
        if len(counts) <= 1:
            continue
        conflict_cells += 1
        cell_rgb = arrays["rgb"][idx].astype(np.float64, copy=False).mean(axis=0)
        cell_normal = normalized_mean_normal(arrays["normal"][idx])
        cell_geometry = dominant_bucket_name(arrays["buckets"][idx])
        candidates = []
        for patch_id, count in counts.items():
            if patch_id not in stats:
                continue
            score, details = cell_patch_score(stats[patch_id], cell_rgb, cell_normal, cell_geometry, count, len(idx), args)
            candidates.append((score, patch_id, details))
        if len(candidates) <= 1:
            continue
        candidates.sort(reverse=True, key=lambda row: row[0])
        best_score, owner, best_details = candidates[0]
        second_score = candidates[1][0]
        if best_score < args.min_owner_score or best_score - second_score < args.owner_margin:
            continue
        changed_mask = group_labels != owner
        changed = int(np.count_nonzero(changed_mask))
        if changed == 0:
            continue
        changed_indices = idx[changed_mask]
        optimized[changed_indices] = owner
        resolved_cells += 1
        transferred_points += changed
        if len(transfer_log) < args.max_log_rows:
            transfer_log.append(
                {
                    "owner_patch": int(owner),
                    "transferred_points": changed,
                    "cell_total": int(len(idx)),
                    "cell_geometry": cell_geometry,
                    "best_score": best_score,
                    "second_score": second_score,
                    "winner_details": best_details,
                    "candidates": [
                        {"patch_id": int(pid), "score": float(score), **details}
                        for score, pid, details in candidates[:8]
                    ],
                }
            )

    report = {
        "schema": "geo-patch-boundary-transfer/v1",
        "fine_voxel_size": args.fine_voxel_size,
        "input_patch_count": int(len(np.unique(labels))),
        "output_patch_count": int(len(np.unique(optimized))),
        "conflict_cells_seen": conflict_cells,
        "resolved_cells": resolved_cells,
        "transferred_points": transferred_points,
        "before_fine_conflict": conflict_summary(labels, arrays["xyz"], args.fine_voxel_size),
        "after_fine_conflict": conflict_summary(optimized, arrays["xyz"], args.fine_voxel_size),
        "before_coarse_conflict": conflict_summary(labels, arrays["xyz"], args.coarse_voxel_size),
        "after_coarse_conflict": conflict_summary(optimized, arrays["xyz"], args.coarse_voxel_size),
        "params": vars(args),
    }
    return optimized, report, transfer_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fine-voxel-size", type=float, default=0.05)
    parser.add_argument("--coarse-voxel-size", type=float, default=0.10)
    parser.add_argument("--small-patch-voxels", type=int, default=8)
    parser.add_argument("--min-owner-score", type=float, default=0.46)
    parser.add_argument("--owner-margin", type=float, default=0.025)
    parser.add_argument("--max-color-distance", type=float, default=130.0)
    parser.add_argument("--size-prior-voxels", type=float, default=100000.0)
    parser.add_argument("--preview-stride", type=int, default=5)
    parser.add_argument("--max-log-rows", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, _src, _dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"label count mismatch: labels={len(labels)} voxels={len(arrays['xyz'])}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    optimized, report, transfer_log = optimize_boundaries(arrays, labels, args)
    report["output_ply"] = str(output_dir / f"geo_patches_boundary_transfer_stride{args.preview_stride}.ply")
    report["output_jsonl"] = str(output_dir / "geo_patches_boundary_transfer.jsonl")
    report["preview_points"] = write_ply(Path(report["output_ply"]), arrays, optimized, args.preview_stride)
    report["jsonl_patch_count"] = write_jsonl(Path(report["output_jsonl"]), arrays, optimized, args)
    (output_dir / "transfer_log.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transfer_log),
        encoding="utf-8",
    )
    (output_dir / "boundary_transfer_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
