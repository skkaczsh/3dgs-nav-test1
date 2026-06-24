#!/usr/bin/env python3
"""Graph-energy style patch optimizer (single-stage).

This optimizer is designed for the 0.01~0.1m region-model pipeline.
It performs all operations in a unified loop:

1) optional internal split on dirty patches
2) boundary reassignment at voxel-cell level
3) adjacent patch merge/split candidate optimization with optional
   simulated-annealing acceptance

Unlike previous staged scripts, boundary transfer and merge decisions are
re-evaluated in the same optimization loop so small, local moves can be
reversed by later moves when evidence changes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BUCKET_NAMES = {
    0: "unknown",
    1: "horizontal",
    2: "vertical",
    3: "thin_linear",
    4: "rough_mixed",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@dataclass
class PatchStats:
    patch_id: int
    count: int
    centroid: np.ndarray
    mean_rgb: np.ndarray
    mean_normal: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bucket_counts: Counter[int]
    geometry_type: str
    source_patch_ids: set[int]
    status: str = "geo_patch"
    source_patch_count: int = 1
    merge_step: int = 0
    conflict_flags: list[str] | None = None


def read_region_input(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        if f.read(len(b"GPRGv1\n")) != b"GPRGv1\n":
            raise ValueError(f"invalid region input magic: {path}")
        n = int(np.fromfile(f, dtype="<i8", count=1)[0])
        m = int(np.fromfile(f, dtype="<i8", count=1)[0])
        arrays = {
            "xyz": np.fromfile(f, dtype="<f4", count=n * 3).reshape(n, 3),
            "rgb": np.fromfile(f, dtype="<f4", count=n * 3).reshape(n, 3),
            "normal": np.fromfile(f, dtype="<f4", count=n * 3).reshape(n, 3),
            "roughness": np.fromfile(f, dtype="<f4", count=n),
            "planarity": np.fromfile(f, dtype="<f4", count=n),
            "linearity": np.fromfile(f, dtype="<f4", count=n),
            "local_color_std": np.fromfile(f, dtype="<f4", count=n),
            "height_range": np.fromfile(f, dtype="<f4", count=n),
            "buckets": np.fromfile(f, dtype="<i2", count=n),
        }
        src = np.fromfile(f, dtype="<i4", count=m)
        dst = np.fromfile(f, dtype="<i4", count=m)
    return arrays, src, dst


def read_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        if f.read(len(b"GPRGlabels1\n")) != b"GPRGlabels1\n":
            raise ValueError(f"invalid labels magic: {path}")
        n = int(np.fromfile(f, dtype="<i8", count=1)[0])
        return np.fromfile(f, dtype="<i4", count=n).astype(np.int32, copy=False)


def write_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GPRGlabels1\n")
        np.array([len(labels)], dtype="<i8").tofile(f)
        labels.astype("<i4", copy=False).tofile(f)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1)
    out = np.zeros_like(value, dtype=np.float64)
    ok = norm > 1e-9
    out[ok] = value[ok] / norm[ok, None]
    return out


def normalized_mean_vector(value: np.ndarray) -> np.ndarray:
    if len(value) == 0:
        return np.zeros(3, dtype=np.float64)
    normal = normalize_rows(value.astype(np.float64, copy=False))
    mean = normal.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return mean / norm


def entropy(values: Counter[int]) -> float:
    if not values:
        return 0.0
    total = sum(values.values())
    p = np.array([v / total for v in values.values()], dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def dominant_geometry(bucket_counts: Counter[int]) -> str:
    if not bucket_counts:
        return "unknown"
    bucket, count = bucket_counts.most_common(1)[0]
    ratio = count / max(sum(bucket_counts.values()), 1)
    if ratio >= 0.6:
        return BUCKET_NAMES.get(int(bucket), "unknown")
    return "mixed"


def compute_patch_stats(arrays: dict[str, np.ndarray], labels: np.ndarray) -> dict[int, PatchStats]:
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
    ends = np.r_[starts[1:], len(sorted_labels)]
    patch_ids = sorted_labels[starts].astype(np.int32, copy=False)

    xyz = arrays["xyz"][order].astype(np.float64, copy=False)
    rgb = arrays["rgb"][order].astype(np.float64, copy=False)
    normal = normalize_rows(arrays["normal"][order].astype(np.float64, copy=False))
    buckets = arrays["buckets"][order].astype(np.int16, copy=False)

    xyz_sum = np.add.reduceat(xyz, starts, axis=0)
    rgb_sum = np.add.reduceat(rgb, starts, axis=0)
    normal_sum = np.add.reduceat(normal, starts, axis=0)
    bbox_min = np.minimum.reduceat(xyz, starts, axis=0)
    bbox_max = np.maximum.reduceat(xyz, starts, axis=0)

    stats: dict[int, PatchStats] = {}
    for i, patch_id in enumerate(patch_ids.tolist()):
        start = int(starts[i])
        end = int(ends[i])
        count = int(end - start)
        bucket_counts = Counter(int(v) for v in buckets[start:end].tolist())
        normal_sum_vec = normal_sum[i]
        nrm = float(np.linalg.norm(normal_sum_vec))
        if nrm > 1e-9:
            normal_sum_vec /= nrm

        stats[int(patch_id)] = PatchStats(
            patch_id=int(patch_id),
            count=count,
            centroid=xyz_sum[i] / max(float(count), 1.0),
            mean_rgb=rgb_sum[i] / max(float(count), 1.0),
            mean_normal=normal_sum_vec,
            bbox_min=bbox_min[i],
            bbox_max=bbox_max[i],
            bucket_counts=bucket_counts,
            geometry_type=dominant_geometry(bucket_counts),
            source_patch_ids={int(patch_id)},
            conflict_flags=["high_entropy"] if entropy(bucket_counts) > 1.1 else [],
        )
    return stats


def merge_patch_stats(a: PatchStats, b: PatchStats) -> PatchStats:
    total = a.count + b.count
    centroid = (a.centroid * a.count + b.centroid * b.count) / max(float(total), 1.0)
    mean_rgb = (a.mean_rgb * a.count + b.mean_rgb * b.count) / max(float(total), 1.0)
    mean_normal = a.mean_normal * a.count + b.mean_normal * b.count
    norm = float(np.linalg.norm(mean_normal))
    if norm > 1e-9:
        mean_normal /= norm
    bucket_counts = Counter(a.bucket_counts)
    bucket_counts.update(b.bucket_counts)
    return PatchStats(
        patch_id=a.patch_id,
        count=total,
        centroid=centroid,
        mean_rgb=mean_rgb,
        mean_normal=mean_normal,
        bbox_min=np.minimum(a.bbox_min, b.bbox_min),
        bbox_max=np.maximum(a.bbox_max, b.bbox_max),
        bucket_counts=bucket_counts,
        geometry_type=dominant_geometry(bucket_counts),
        source_patch_ids=set(a.source_patch_ids) | set(b.source_patch_ids),
        source_patch_count=a.source_patch_count + b.source_patch_count,
        merge_step=max(a.merge_step, b.merge_step) + 1,
        conflict_flags=sorted(set((a.conflict_flags or []) + (b.conflict_flags or []))),
        status="merged",
    )


def compatible_bucket_score(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if "unknown" in {a, b}:
        return 0.62
    if {a, b} <= {"rough_mixed", "thin_linear", "mixed"}:
        return 0.78
    if {a, b} in [{"horizontal", "rough_mixed"}, {"vertical", "rough_mixed"}]:
        return 0.56
    return 0.09


def normal_score(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-9 or bn < 1e-9:
        return 0.5
    return max(0.0, min(1.0, float(np.dot(a, b) / (an * bn))))


def bbox_gap(a: PatchStats, b: PatchStats) -> float:
    gap = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    return float(np.linalg.norm(gap))


def bbox_volume(stats: PatchStats) -> float:
    extent = np.maximum(stats.bbox_max - stats.bbox_min, 1e-3)
    return float(np.prod(extent))


def bbox_overlap_features(a: PatchStats, b: PatchStats) -> dict[str, float]:
    dims = np.maximum(0.0, np.minimum(a.bbox_max, b.bbox_max) - np.maximum(a.bbox_min, b.bbox_min))
    overlap = float(np.prod(dims))
    va = bbox_volume(a)
    vb = bbox_volume(b)
    union = va + vb - overlap
    return {
        "bbox_overlap_volume": overlap,
        "bbox_ratio_min": overlap / max(min(va, vb), 1e-9),
        "bbox_ratio_max": overlap / max(max(va, vb), 1e-9),
        "bbox_iou": overlap / max(union, 1e-9),
        "bbox_centroid_distance": float(np.linalg.norm(a.centroid - b.centroid)),
    }


def geometry_size_penalty(stats: PatchStats) -> float:
    # Encourage compact, coherent geometry model, but avoid over-penalizing very large surfaces.
    extent = np.maximum(stats.bbox_max - stats.bbox_min, 1e-6)
    span = float(np.sqrt((extent * extent).sum()))
    return max(0.0, 1.0 - 0.08 * span)


def cell_bucket_signature(buckets: np.ndarray) -> str:
    if len(buckets) == 0:
        return "unknown"
    cnt = Counter(int(v) for v in buckets.tolist())
    return dominant_geometry(cnt)


def boundary_score(
    patch: PatchStats,
    cell_rgb: np.ndarray,
    cell_normal: np.ndarray,
    cell_bucket: str,
    share_ratio: float,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    color_dist = float(np.linalg.norm(patch.mean_rgb - cell_rgb))
    color = max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
    bucket = compatible_bucket_score(patch.geometry_type, cell_bucket)
    normal = normal_score(patch.mean_normal, cell_normal)
    share = float(share_ratio)
    size_prior = min(1.0, math.log1p(max(patch.count, 1)) / math.log1p(args.patch_size_prior))

    guard = 0.0
    if patch.geometry_type in {"horizontal", "vertical"} and cell_bucket not in {patch.geometry_type, "unknown", "mixed", "rough_mixed"}:
        guard += args.surface_guard

    score = (
        0.34 * color
        + 0.20 * bucket
        + 0.18 * normal
        + 0.18 * share
        + 0.10 * size_prior
        - guard
    )
    return score, {
        "score": score,
        "color": color,
        "bucket": bucket,
        "normal": normal,
        "share": share,
        "size_prior": size_prior,
        "guard": guard,
    }


def patch_pair_quality(
    patch: PatchStats,
    neighbor_share: int,
    args: argparse.Namespace,
) -> float:
    # Patch-internal quality for annealing objective.
    color = max(0.0, 1.0 - float(np.linalg.norm(patch.mean_rgb - np.array([120.0, 120.0, 120.0]))) / 220.0)
    normal = normal_score(patch.mean_normal, np.array([0.0, 0.0, 1.0]))
    geom = 1.0 if patch.geometry_type != "mixed" else 0.4
    size = min(1.0, math.log1p(patch.count) / math.log1p(max(args.patch_target_size, 1)))
    neigh = min(1.0, neighbor_share / max(args.pair_share_norm, 1.0))
    return 0.28 * color + 0.20 * normal + 0.24 * geom + 0.15 * size + 0.13 * neigh


def merge_pair_gain(
    a: PatchStats,
    b: PatchStats,
    shared_edges: int,
    neighbor_share: float,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    color_dist = float(np.linalg.norm(a.mean_rgb - b.mean_rgb))
    color = max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
    bucket = compatible_bucket_score(a.geometry_type, b.geometry_type)
    normal = normal_score(a.mean_normal, b.mean_normal)
    gap_score = max(0.0, min(1.0, 1.0 - bbox_gap(a, b) / max(args.max_bbox_gap, 1e-6)))
    edge = min(1.0, float(shared_edges) / max(float(min(a.count, b.count)), 1.0))
    entropy_gain = max(0.0, (1.2 - max(entropy(a.bucket_counts), entropy(b.bucket_counts))) / 1.2)

    merged = merge_patch_stats(a, b)
    size_penalty = max(0.0, 0.5 - len(merged.source_patch_ids) / max(1, args.max_patch_sources))
    geometry_guard = 0.0
    if a.geometry_type != b.geometry_type and "horizontal" in {a.geometry_type, b.geometry_type} and "vertical" in {a.geometry_type, b.geometry_type}:
        geometry_guard = args.surface_merge_penalty

    merge_score = (
        0.36 * color
        + 0.18 * bucket
        + 0.16 * normal
        + 0.12 * gap_score
        + 0.10 * edge
        + 0.05 * neighbor_share
        + 0.03 * entropy_gain
    )
    separate = (
        0.24 * (color + bucket + normal) / 3.0
        + 0.15 * geometry_size_penalty(a)
        + 0.15 * geometry_size_penalty(b)
        + geometry_size_penalty(a) * geometry_size_penalty(b)
    )
    gain = merge_score - separate - geometry_guard - size_penalty

    return gain, {
        "gain": gain,
        "color": color,
        "bucket": bucket,
        "normal": normal,
        "gap": gap_score,
        "edge": edge,
        "entropy_gain": entropy_gain,
        "size_penalty": size_penalty,
        "geometry_guard": geometry_guard,
        "separate": separate,
        "merge_score": merge_score,
    }


def attachment_merge_decision(
    anchor: PatchStats,
    fragment: PatchStats,
    shared_edges: int,
    candidate_support: float,
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, float]]:
    """Decide whether a tiny fragment should become part of a large patch.

    This is deliberately separate from general patch merging.  General merging
    optimizes two comparable patch models.  Attachment merging handles the
    common over-fragmentation case where a few voxels are strongly glued to a
    large rough/mixed surface but would be filtered out by min-anchor rules.
    """
    if not args.enable_attachment_merge:
        return False, "attachment_disabled", {}
    if anchor.count < args.attachment_min_anchor_voxels:
        return False, "attachment_anchor_too_small", {}
    if fragment.count > args.attachment_max_fragment_voxels:
        return False, "attachment_fragment_too_large", {}
    size_ratio = float(anchor.count) / max(float(fragment.count), 1.0)
    if size_ratio < args.attachment_min_size_ratio:
        return False, "attachment_size_ratio", {"size_ratio": size_ratio}
    if "mixed" not in {anchor.geometry_type, fragment.geometry_type} and "rough_mixed" not in {anchor.geometry_type, fragment.geometry_type}:
        return False, "attachment_bucket", {"size_ratio": size_ratio}

    contact_ratio = float(shared_edges) / max(float(fragment.count), 1.0)
    patch_color_dist = float(np.linalg.norm(anchor.mean_rgb - fragment.mean_rgb))
    contact_color_dist = float(candidate.get("contact_color_distance", -1.0))
    if args.attachment_use_contact_evidence and contact_color_dist >= 0:
        color_dist = contact_color_dist
    else:
        color_dist = patch_color_dist
    color = max(0.0, min(1.0, 1.0 - color_dist / max(args.attachment_max_color_distance, 1e-6)))
    patch_normal = normal_score(anchor.mean_normal, fragment.mean_normal)
    contact_normal = float(candidate.get("contact_normal_score", -1.0))
    if args.attachment_use_contact_evidence and contact_normal >= 0:
        normal = contact_normal
    else:
        normal = patch_normal
    bucket = compatible_bucket_score(anchor.geometry_type, fragment.geometry_type)
    gap = bbox_gap(anchor, fragment)
    gap_score = max(0.0, min(1.0, 1.0 - gap / max(args.attachment_max_bbox_gap, 1e-6)))
    contact = max(0.0, min(1.0, contact_ratio / max(args.attachment_contact_norm, 1e-6)))
    support = max(float(candidate_support), contact_ratio)
    score = (
        args.attachment_color_weight * color
        + args.attachment_normal_weight * normal
        + args.attachment_bucket_weight * bucket
        + args.attachment_contact_weight * contact
        + args.attachment_gap_weight * gap_score
    ) / max(
        args.attachment_color_weight
        + args.attachment_normal_weight
        + args.attachment_bucket_weight
        + args.attachment_contact_weight
        + args.attachment_gap_weight,
        1e-9,
    )
    detail = {
        "attachment_score": float(score),
        "attachment_color": float(color),
        "attachment_normal": float(normal),
        "attachment_bucket": float(bucket),
        "attachment_contact": float(contact),
        "attachment_gap": float(gap_score),
        "attachment_color_distance": float(color_dist),
        "attachment_patch_color_distance": float(patch_color_dist),
        "attachment_contact_color_distance": float(contact_color_dist),
        "attachment_patch_normal": float(patch_normal),
        "attachment_contact_normal": float(contact_normal),
        "attachment_bbox_gap": float(gap),
        "attachment_contact_ratio": float(contact_ratio),
        "attachment_size_ratio": float(size_ratio),
        "attachment_support": float(support),
    }
    if shared_edges < args.attachment_min_shared_edges:
        return False, "attachment_shared_edges", detail
    if contact_ratio < args.attachment_min_contact_ratio:
        return False, "attachment_contact_ratio", detail
    if color_dist > args.attachment_max_color_distance:
        return False, "attachment_color_distance", detail
    if normal < args.attachment_min_normal_score:
        return False, "attachment_normal", detail
    if gap > args.attachment_max_bbox_gap:
        return False, "attachment_bbox_gap", detail
    if score < args.attachment_min_score:
        return False, "attachment_score", detail
    return True, "accepted_attachment", detail


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
    keys = a * (int(labels.max()) + 1) + b
    uk, uc = np.unique(keys, return_counts=True)
    max_label = int(labels.max())
    return {
        (int(k // (max_label + 1)), int(k % (max_label + 1))): int(c)
        for k, c in zip(uk.tolist(), uc.tolist())
    }


def build_edge_features(
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[tuple[int, int], dict[str, float]]:
    if len(src) == 0:
        return {}
    la = labels[src]
    lb = labels[dst]
    mask = la != lb
    if not np.any(mask):
        return {}

    src_m = src[mask]
    dst_m = dst[mask]
    la = la[mask].astype(np.int64, copy=False)
    lb = lb[mask].astype(np.int64, copy=False)
    swap = la > lb
    a = np.where(swap, lb, la)
    b = np.where(swap, la, lb)
    idx_a = np.where(swap, dst_m, src_m)
    idx_b = np.where(swap, src_m, dst_m)

    max_label = int(labels.max())
    keys = a * (max_label + 1) + b
    uk, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)

    rgb = arrays["rgb"].astype(np.float64, copy=False)
    normal = normalize_rows(arrays["normal"].astype(np.float64, copy=False))
    rgb_a_sum = np.zeros((len(uk), 3), dtype=np.float64)
    rgb_b_sum = np.zeros((len(uk), 3), dtype=np.float64)
    normal_a_sum = np.zeros((len(uk), 3), dtype=np.float64)
    normal_b_sum = np.zeros((len(uk), 3), dtype=np.float64)
    np.add.at(rgb_a_sum, inv, rgb[idx_a])
    np.add.at(rgb_b_sum, inv, rgb[idx_b])
    np.add.at(normal_a_sum, inv, normal[idx_a])
    np.add.at(normal_b_sum, inv, normal[idx_b])

    rgb_a_mean = rgb_a_sum / counts[:, None]
    rgb_b_mean = rgb_b_sum / counts[:, None]
    color_distance = np.linalg.norm(rgb_a_mean - rgb_b_mean, axis=1)
    nrm_a = np.linalg.norm(normal_a_sum, axis=1)
    nrm_b = np.linalg.norm(normal_b_sum, axis=1)
    dot = np.sum(normal_a_sum * normal_b_sum, axis=1)
    normal_sim = np.full(len(uk), 0.5, dtype=np.float64)
    ok = (nrm_a > 1e-9) & (nrm_b > 1e-9)
    normal_sim[ok] = np.clip(dot[ok] / (nrm_a[ok] * nrm_b[ok]), 0.0, 1.0)

    out: dict[tuple[int, int], dict[str, float]] = {}
    for i, key in enumerate(uk.tolist()):
        pair = (int(key // (max_label + 1)), int(key % (max_label + 1)))
        out[pair] = {
            "shared_edges": int(counts[i]),
            "contact_color_distance": float(color_distance[i]),
            "contact_normal_score": float(normal_sim[i]),
        }
    return out


def build_overlap_candidate_scores(stats: dict[int, PatchStats], args: argparse.Namespace) -> dict[tuple[int, int], float]:
    if not args.enable_overlap_merge_candidates or args.overlap_candidate_top_n <= 1:
        return {}
    selected = sorted(stats.values(), key=lambda row: row.count, reverse=True)[: args.overlap_candidate_top_n]
    out: dict[tuple[int, int], float] = {}
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            f = bbox_overlap_features(a, b)
            if f["bbox_overlap_volume"] <= 0:
                continue
            if f["bbox_ratio_min"] < args.overlap_candidate_min_ratio and f["bbox_iou"] < args.overlap_candidate_min_iou:
                continue
            if f["bbox_centroid_distance"] > args.overlap_candidate_max_centroid and f["bbox_ratio_min"] < args.overlap_candidate_long_min_ratio:
                continue
            pa, pb = sorted((int(a.patch_id), int(b.patch_id)))
            out[(pa, pb)] = max(float(f["bbox_ratio_min"]), float(f["bbox_iou"]))
            if len(out) >= args.overlap_candidate_max_pairs:
                return out
    return out


def linearize_cells(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    grid = np.floor(xyz / voxel_size).astype(np.int64, copy=False)
    grid -= grid.min(axis=0)
    dims = grid.max(axis=0) + 1
    return (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]


def build_fine_overlap_candidate_scores(
    xyz: np.ndarray,
    labels: np.ndarray,
    stats: dict[int, PatchStats],
    args: argparse.Namespace,
) -> dict[tuple[int, int], float]:
    if not args.enable_fine_overlap_merge_candidates or args.fine_overlap_voxel_size <= 0:
        return {}
    cell_ids = linearize_cells(xyz, args.fine_overlap_voxel_size)
    order = np.lexsort((labels, cell_ids))
    cell_ids = cell_ids[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    starts = np.r_[0, np.flatnonzero(np.diff(cell_ids)) + 1]
    ends = np.r_[starts[1:], len(cell_ids)]

    patch_cell_counts: Counter[int] = Counter()
    pair_cell_counts: Counter[tuple[int, int]] = Counter()
    for start, end in zip(starts.tolist(), ends.tolist()):
        cell_labels = np.unique(sorted_labels[start:end])
        if len(cell_labels) < 2 or len(cell_labels) > args.fine_overlap_max_labels_per_cell:
            if len(cell_labels) == 1:
                patch_cell_counts[int(cell_labels[0])] += 1
            continue
        labels_list = [int(v) for v in cell_labels.tolist() if int(v) in stats]
        if len(labels_list) < 2:
            continue
        for label in labels_list:
            patch_cell_counts[label] += 1
        for i, a in enumerate(labels_list):
            for b in labels_list[i + 1 :]:
                pair_cell_counts[(a, b)] += 1

    out: dict[tuple[int, int], float] = {}
    for (a, b), shared in pair_cell_counts.items():
        if shared < args.fine_overlap_min_cells:
            continue
        min_cells = min(patch_cell_counts.get(a, 0), patch_cell_counts.get(b, 0))
        if min_cells <= 0:
            continue
        ratio = float(shared) / max(float(min_cells), 1.0)
        if ratio < args.fine_overlap_min_ratio:
            continue
        out[(a, b)] = ratio
        if len(out) >= args.fine_overlap_max_pairs:
            break
    return out


def build_merge_candidates(
    edge_counts: dict[tuple[int, int], int],
    stats: dict[int, PatchStats],
    args: argparse.Namespace,
    fine_scores: dict[tuple[int, int], float] | None = None,
    edge_features: dict[tuple[int, int], dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    for pair, shared in edge_counts.items():
        a, b = pair
        if a not in stats or b not in stats:
            continue
        features = (edge_features or {}).get(pair, {})
        support = float(shared) / max(float(min(stats[a].count, stats[b].count)), 1.0)
        candidates[pair] = {
            "pair": pair,
            "shared_edges": int(shared),
            "support": support,
            "source": "adjacency",
            "overlap_support": 0.0,
            "fine_overlap_support": 0.0,
            "contact_color_distance": float(features.get("contact_color_distance", -1.0)),
            "contact_normal_score": float(features.get("contact_normal_score", -1.0)),
        }

    overlap_scores = build_overlap_candidate_scores(stats, args)
    for pair, score in overlap_scores.items():
        if pair in candidates:
            candidates[pair]["support"] = max(float(candidates[pair]["support"]), float(score))
            candidates[pair]["overlap_support"] = float(score)
            candidates[pair]["source"] = "adjacency+overlap"
            continue
        candidates[pair] = {
            "pair": pair,
            "shared_edges": 0,
            "support": float(score),
            "source": "overlap",
            "overlap_support": float(score),
            "fine_overlap_support": 0.0,
            "contact_color_distance": -1.0,
            "contact_normal_score": -1.0,
        }

    for pair, score in (fine_scores or {}).items():
        if pair in candidates:
            candidates[pair]["support"] = max(float(candidates[pair]["support"]), float(score))
            candidates[pair]["fine_overlap_support"] = float(score)
            source = str(candidates[pair]["source"])
            if "fine_overlap" not in source:
                candidates[pair]["source"] = f"{source}+fine_overlap"
            continue
        candidates[pair] = {
            "pair": pair,
            "shared_edges": 0,
            "support": float(score),
            "source": "fine_overlap",
            "overlap_support": 0.0,
            "fine_overlap_support": float(score),
            "contact_color_distance": -1.0,
            "contact_normal_score": -1.0,
        }

    out = list(candidates.values())
    out.sort(key=lambda row: (float(row["support"]), int(row["shared_edges"])), reverse=True)
    return out[: args.max_merge_candidates]


def build_conflict_cells(
    xyz: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.rint(xyz / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[starts[1:], len(order)]
    return order, starts, ends


def boundary_transfer(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    stats: dict[int, PatchStats],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[np.ndarray, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    order, starts, ends = build_conflict_cells(arrays["xyz"], labels, args.fine_voxel_size)
    s_labels = labels[order]

    rejected = 0
    accepted = 0
    accept_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    out = labels.copy()

    for start, end in zip(starts, ends, strict=True):
        if end - start <= 1:
            continue
        idx = order[start:end]
        cell_labels = s_labels[start:end]
        count = Counter(int(v) for v in cell_labels.tolist())
        if len(count) <= 1:
            continue

        patch_ids = sorted(count.items(), key=lambda x: x[1], reverse=True)
        current = patch_ids[0][0]
        cell_rgb = arrays["rgb"][idx].astype(np.float64).mean(axis=0)
        cell_normal = normalized_mean_vector(arrays["normal"][idx])
        cell_bucket = cell_bucket_signature(arrays["buckets"][idx])

        candidates: list[tuple[float, int, dict[str, float]]] = []
        for pid, c in count.items():
            if pid not in stats:
                continue
            score, details = boundary_score(stats[pid], cell_rgb, cell_normal, cell_bucket, c / max(float(end - start), 1.0), args)
            candidates.append((score, pid, details))

        if len(candidates) <= 1:
            continue
        candidates.sort(reverse=True, key=lambda x: x[0])
        best_score, best, best_details = candidates[0]
        second_score = candidates[1][0] if len(candidates) >= 2 else float("-inf")
        margin = best_score - second_score
        if best == current or margin < args.boundary_margin:
            rejected += 1
            if len(reject_rows) < args.max_log_rows:
                reject_rows.append(
                    {
                        "cell_size": int(end - start),
                        "current_patch": int(current),
                        "best_patch": int(best),
                        "best_score": float(best_score),
                        "second_score": float(second_score),
                        "reason": "margin_too_small_or_same_owner",
                    }
                )
            continue
        if best_score < args.min_boundary_owner_score:
            rejected += 1
            if len(reject_rows) < args.max_log_rows:
                reject_rows.append(
                    {
                        "cell_size": int(end - start),
                        "current_patch": int(current),
                        "best_patch": int(best),
                        "best_score": float(best_score),
                        "second_score": float(second_score),
                        "reason": "score_below_threshold",
                    }
                )
            continue

        changed_mask = cell_labels != best
        if not np.any(changed_mask):
            continue
        changed_points = int(np.count_nonzero(changed_mask))
        out[idx[changed_mask]] = int(best)
        accepted += changed_points
        if len(accept_rows) < args.max_log_rows:
            accept_rows.append(
                {
                    "cell_size": int(end - start),
                    "from_patch": int(current),
                    "to_patch": int(best),
                    "changed_points": changed_points,
                    "best_score": float(best_score),
                    "second_score": float(second_score),
                    "margin": float(margin),
                    "cell_bucket": cell_bucket,
                    "winner_details": best_details,
                }
            )

    # tiny bit of randomness to avoid lock-step oscillation
    if accepted > 0 and args.seed and args.boundary_shuffle_ratio > 0:
        if rng.random() < args.boundary_shuffle_ratio:
            shuffled = out.copy()
            n = min(int(0.0005 * len(shuffled)), args.max_shuffle_swaps)
            for _ in range(n):
                i = rng.randrange(len(shuffled))
                j = rng.randrange(len(shuffled))
                if shuffled[i] != shuffled[j] and rng.random() < 0.5:
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            out = shuffled

    return out, accepted, rejected, accept_rows, reject_rows


def split_component(
    patch_id: int,
    point_ids: np.ndarray,
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    next_id: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    if len(point_ids) < args.split_min_component_voxels:
        return labels, next_id, []

    # Build internal edges for this patch.
    in_patch = set(point_ids.tolist())
    patch_mask = np.isin(src, point_ids) & np.isin(dst, point_ids)
    if not np.any(patch_mask):
        return labels, next_id, []
    s = src[patch_mask]
    t = dst[patch_mask]
    if len(s) == 0:
        return labels, next_id, []

    sub_ids = np.asarray(sorted(point_ids), dtype=np.int64)
    loc = {int(pid): int(i) for i, pid in enumerate(sub_ids)}
    order = np.arange(len(sub_ids), dtype=np.int32)

    parent = np.arange(len(sub_ids), dtype=np.int32)
    rank = np.zeros(len(sub_ids), dtype=np.uint8)

    def fnd(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def uni(a: int, b: int) -> None:
        ra = fnd(a)
        rb = fnd(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Feature continuity for this patch
    geoidx = arrays["buckets"][point_ids]
    geom = dominant_geometry(Counter(int(v) for v in geoidx.tolist()))
    for a, b in zip(s.tolist(), t.tolist(), strict=True):
        la = loc.get(int(a))
        lb = loc.get(int(b))
        if la is None or lb is None:
            continue
        if a == b:
            continue
        if np.linalg.norm(arrays["rgb"][a] - arrays["rgb"][b]) > args.internal_color_distance:
            continue
        normal_a = normalize_rows(arrays["normal"][[a]])[0]
        normal_b = normalize_rows(arrays["normal"][[b]])[0]
        if np.linalg.norm(normal_a) < 1e-9 or np.linalg.norm(normal_b) < 1e-9:
            continue
        dot = float(np.dot(normal_a, normal_b))
        if dot < args.internal_normal_dot and geom in {"horizontal", "vertical", "rough_mixed", "unknown"}:
            continue
        uni(int(la), int(lb))

    roots = np.fromiter((fnd(i) for i in range(len(sub_ids))), dtype=np.int32, count=len(sub_ids))
    comps, counts = np.unique(roots, return_counts=True)
    if len(comps) <= 1:
        return labels, next_id, []

    # keep largest as parent, split others to new labels
    comp_counts = sorted(zip(comps.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)
    largest_root = int(comp_counts[0][0])
    largest_count = int(comp_counts[0][1])

    component_groups: dict[int, list[int]] = defaultdict(list)
    for local, root in enumerate(roots.tolist()):
        component_groups[int(root)].append(sub_ids[local])

    logs: list[dict[str, Any]] = []
    for comp_root, comp_size in comp_counts:
        if comp_root == largest_root:
            continue
        if comp_size >= args.residual_component_voxels:
            new_id = next_id
            next_id += 1
            labels[np.array(component_groups[comp_root], dtype=np.int64)] = new_id
            logs.append({
                "patch_id": int(patch_id),
                "new_patch_id": int(new_id),
                "component_size": int(comp_size),
                "reason": "split",
            })
        elif comp_size >= args.split_min_component_voxels:
            new_id = next_id
            next_id += 1
            labels[np.array(component_groups[comp_root], dtype=np.int64)] = new_id
            logs.append({
                "patch_id": int(patch_id),
                "new_patch_id": int(new_id),
                "component_size": int(comp_size),
                "reason": "residual",
            })
        else:
            logs.append({
                "patch_id": int(patch_id),
                "new_patch_id": int(patch_id),
                "component_size": int(comp_size),
                "reason": "kept",
            })

    return labels, next_id, logs


def propose_splits(
    labels: np.ndarray,
    arrays: np.ndarray | dict[str, np.ndarray],
    src: np.ndarray,
    dst: np.ndarray,
    stats: dict[int, PatchStats],
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, list[dict[str, Any]], int]:
    if not args.enable_split:
        return labels, max(stats.keys(), default=0) + 1, [], 0

    if isinstance(arrays, np.ndarray):
        return labels, 0, [], 0
    next_id = int(labels.max()) + 1
    split_logs: list[dict[str, Any]] = []
    split_count = 0
    candidates = [
        (pid, st)
        for pid, st in stats.items()
        if (st.count >= args.dirty_min_voxels)
        or (
            st.count >= args.dirty_entropy_min_voxels
            and entropy(st.bucket_counts) >= args.entropy_split_threshold
            and (st.bbox_max[2] - st.bbox_min[2]) > args.dirty_min_height
        )
    ]
    for pid, _st in sorted(candidates, key=lambda x: x[1].count, reverse=True):
        mask = np.nonzero(labels == pid)[0]
        if len(mask) < args.split_min_component_voxels:
            continue
        labels, next_id, rows = split_component(pid, mask, arrays, labels, src, dst, next_id, args)
        split_logs.extend(rows)
        split_count += len(rows)
    return labels, next_id, split_logs, split_count


def merge_step(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    stats: dict[int, PatchStats],
    args: argparse.Namespace,
    rng: random.Random,
    temp: float,
) -> tuple[np.ndarray, int, list[dict[str, Any]], int]:
    edge_counts = build_edge_counts(labels, arrays["src"] if "src" in arrays else np.array([], dtype=np.int32), arrays["dst"] if "dst" in arrays else np.array([], dtype=np.int32))
    edge_features = build_edge_features(
        labels,
        arrays["src"] if "src" in arrays else np.array([], dtype=np.int32),
        arrays["dst"] if "dst" in arrays else np.array([], dtype=np.int32),
        arrays,
    )
    fine_scores = build_fine_overlap_candidate_scores(arrays["xyz"], labels, stats, args)
    candidates = build_merge_candidates(edge_counts, stats, args, fine_scores=fine_scores, edge_features=edge_features)
    if not candidates:
        return labels, 0, [], 0

    accepted = 0
    rejects = 0
    logs: list[dict[str, Any]] = []

    for candidate in candidates:
        a, b = candidate["pair"]
        shared = int(candidate["shared_edges"])
        candidate_source = str(candidate["source"])
        candidate_support = float(candidate["support"])
        overlap_support = float(candidate.get("overlap_support", 0.0))
        fine_overlap_support = float(candidate.get("fine_overlap_support", 0.0))
        if a not in stats or b not in stats:
            continue
        if stats[a].count > stats[b].count:
            anchor_id = a
            src_id = b
        else:
            anchor_id = b
            src_id = a

        anchor = stats[anchor_id]
        src_stats = stats[src_id]
        if src_id == anchor_id:
            continue

        adjacency_share = float(shared) / max(float(min(anchor.count, src_stats.count)), 1.0)
        neighbor_share = max(adjacency_share, candidate_support)
        attachment_ok = False
        attachment_reason = ""
        attachment_detail: dict[str, float] = {}
        if min(anchor.count, src_stats.count) < args.min_anchor_voxels:
            attachment_ok, attachment_reason, attachment_detail = attachment_merge_decision(
                anchor,
                src_stats,
                shared,
                candidate_support,
                candidate,
                args,
            )
            if attachment_ok:
                labels[labels == src_id] = anchor_id
                stats[anchor_id] = merge_patch_stats(anchor, src_stats)
                del stats[src_id]
                accepted += 1
                if len(logs) < args.max_log_rows:
                    logs.append(
                        {
                            "status": "accept",
                            "reason": attachment_reason,
                            "anchor_patch_id": int(anchor_id),
                            "merge_patch_id": int(src_id),
                            "shared_edges": int(shared),
                            "neighbor_share": float(neighbor_share),
                            "adjacency_share": float(adjacency_share),
                            "overlap_support": float(overlap_support),
                            "fine_overlap_support": float(fine_overlap_support),
                            "candidate_source": f"{candidate_source}+attachment",
                            **attachment_detail,
                        }
                    )
                continue
            rejects += 1
            if len(logs) < args.max_log_rows:
                logs.append(
                    {
                        "status": "reject",
                        "a": int(anchor_id),
                        "b": int(src_id),
                        "shared_edges": int(shared),
                        "neighbor_share": float(neighbor_share),
                        "adjacency_share": float(adjacency_share),
                        "overlap_support": float(overlap_support),
                        "fine_overlap_support": float(fine_overlap_support),
                        "candidate_source": f"{candidate_source}+attachment",
                        "reason": attachment_reason or "small_patch_no_attachment",
                        **attachment_detail,
                    }
                )
            continue
        if (
            args.overlap_only_require_fine_overlap
            and candidate_source == "overlap"
            and fine_overlap_support < args.fine_overlap_min_ratio
        ):
            rejects += 1
            if len(logs) < args.max_log_rows:
                logs.append(
                    {
                        "status": "reject",
                        "a": int(anchor_id),
                        "b": int(src_id),
                        "shared_edges": int(shared),
                        "neighbor_share": float(neighbor_share),
                        "adjacency_share": float(adjacency_share),
                        "overlap_support": float(overlap_support),
                        "fine_overlap_support": float(fine_overlap_support),
                        "candidate_source": candidate_source,
                        "reason": "overlap_only_without_fine_overlap",
                    }
                )
            continue
        if candidate_source == "overlap" and overlap_support < args.overlap_candidate_min_ratio:
            continue
        if candidate_source == "fine_overlap" and fine_overlap_support < args.fine_overlap_min_ratio:
            continue
        if "adjacency" in candidate_source:
            if (
                adjacency_share < args.merge_min_neighbor_support
                and overlap_support < args.overlap_candidate_min_ratio
                and fine_overlap_support < args.fine_overlap_min_ratio
            ):
                continue
        gain, detail = merge_pair_gain(anchor, src_stats, shared, neighbor_share, args)
        accepted_decision = False
        if gain >= args.min_merge_gain:
            accepted_decision = True
        else:
            if args.enable_annealing and temp > args.anneal_temp_min and np.exp(gain / max(temp, 1e-6)) > rng.random():
                accepted_decision = True

        if not accepted_decision:
            rejects += 1
            if len(logs) < args.max_log_rows:
                logs.append(
                    {
                        "status": "reject",
                        "a": int(anchor_id),
                        "b": int(src_id),
                        "shared_edges": int(shared),
                        "gain": float(gain),
                        "neighbor_share": float(neighbor_share),
                        "adjacency_share": float(adjacency_share),
                        "overlap_support": float(overlap_support),
                        "fine_overlap_support": float(fine_overlap_support),
                        "candidate_source": candidate_source,
                        "reason": "min_gain_or_anneal_fail",
                        **detail,
                    }
                )
            continue

        # Accept merge.
        labels[labels == src_id] = anchor_id
        stats[anchor_id] = merge_patch_stats(anchor, src_stats)
        del stats[src_id]
        accepted += 1
        if len(logs) < args.max_log_rows:
            logs.append(
                {
                    "status": "accept",
                    "anchor_patch_id": int(anchor_id),
                    "merge_patch_id": int(src_id),
                    "shared_edges": int(shared),
                    "neighbor_share": float(neighbor_share),
                    "adjacency_share": float(adjacency_share),
                    "overlap_support": float(overlap_support),
                    "fine_overlap_support": float(fine_overlap_support),
                    "candidate_source": candidate_source,
                    "gain": float(gain),
                    **detail,
                }
            )

    return labels, accepted, logs, rejects


def write_ply(path: Path, arrays: dict[str, np.ndarray], labels: np.ndarray, stride: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(0, len(labels), stride, dtype=np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(idx)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for i in idx.tolist():
            x, y, z = arrays["xyz"][i]
            # deterministic pseudo-color to avoid relying on label remap
            hid = int(labels[i])
            seed = (hid * 1103515245 + 12345) & 0xFFFFFFFF
            r = 48 + (seed & 0xBF)
            g = 48 + ((seed >> 8) & 0xBF)
            b = 48 + ((seed >> 16) & 0xBF)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {hid} 1\n")
    return int(len(idx))


def summarize_log_reasons(rows: list[dict[str, Any]], key: str = "reason") -> dict[str, int]:
    return dict(Counter(str(row.get(key, "unknown")) for row in rows))


def summarize_merge_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("status", "unknown")) for row in rows))


def write_jsonl(path: Path, stats: dict[int, PatchStats], args: argparse.Namespace) -> int:
    with path.open("w", encoding="utf-8") as f:
        for patch_id in sorted(stats):
            s = stats[patch_id]
            row = {
                "patch_id": patch_id,
                "object": patch_id,
                "voxel_count": s.count,
                "status": s.status,
                "geometry_type": s.geometry_type,
                "semantic_label": s.geometry_type,
                "description": f"energy-graph: {s.geometry_type}",
                "bucket_counts": {BUCKET_NAMES[k]: int(v) for k, v in s.bucket_counts.items()},
                "bucket_entropy": entropy(s.bucket_counts),
                "centroid": s.centroid.tolist(),
                "bbox_3d": {"min": s.bbox_min.tolist(), "max": s.bbox_max.tolist()},
                "extent": (s.bbox_max - s.bbox_min).tolist(),
                "mean_rgb": s.mean_rgb.tolist(),
                "mean_normal": s.mean_normal.tolist(),
                "source_patch_count": s.source_patch_count,
                "source_patch_ids": sorted(int(v) for v in s.source_patch_ids)[: args.max_source_patch_ids],
                "source_patch_ids_truncated": len(s.source_patch_ids) > args.max_source_patch_ids,
                "conflict_flags": s.conflict_flags or [],
                "merge_step": s.merge_step,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(stats)


def optimize(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"label count mismatch: labels={len(labels)} voxels={len(arrays['xyz'])}")

    rng = random.Random(args.seed)
    report: dict[str, Any] = {}

    log("compute initial patch stats")
    stats = compute_patch_stats(arrays, labels)
    report["input_patch_count"] = int(len(stats))
    report["input_point_count"] = int(len(labels))
    report["input_high_entropy_count"] = int(sum(1 for s in stats.values() if entropy(s.bucket_counts) > args.entropy_split_threshold))

    # stash connectivity graph for merge step without changing signature
    working = arrays.copy()
    working["src"] = src
    working["dst"] = dst

    total_split = 0
    total_boundary_ok = 0
    total_boundary_reject = 0
    total_merge = 0
    total_merge_reject = 0
    split_log: list[dict[str, Any]] = []
    boundary_accept_log: list[dict[str, Any]] = []
    boundary_reject_log: list[dict[str, Any]] = []
    merge_log: list[dict[str, Any]] = []

    # optional split first to prevent huge mixed patches blocking objective
    if args.enable_split:
        log("propose initial splits")
        labels, _next, split_rows, split_cnt = propose_splits(labels, working, src, dst, stats, args)
        split_log.extend(split_rows)
        total_split = split_cnt
        stats = compute_patch_stats(working, labels)
        report["after_split_patch_count"] = int(len(stats))

    temp = args.anneal_temp_start
    temp_decay = (args.anneal_temp_start - args.anneal_temp_min) / max(args.max_iters, 1)

    for step in range(max(args.max_iters, 1)):
        changed = False
        log(f"iteration {step + 1}/{max(args.max_iters, 1)} start: patch_count={len(stats)} temp={temp:.4f}")

        # 1) boundary reassignment in current geometry
        if args.enable_boundary:
            log("boundary transfer")
            next_labels, moved, rejected, bal, brl = boundary_transfer(working, labels, stats, args, rng)
            total_boundary_ok += moved
            total_boundary_reject += rejected
            boundary_accept_log.extend(bal)
            boundary_reject_log.extend(brl)
            if np.any(next_labels != labels):
                changed = True
                labels = next_labels

        # 2) recompute then merge
        log("compute stats before merge")
        stats = compute_patch_stats(working, labels)
        log("merge step")
        next_labels, merged, mlog, mrej = merge_step(working, labels, stats, args, rng, temp)
        if merged > 0:
            changed = True
        total_merge += merged
        total_merge_reject += mrej
        merge_log.extend(mlog)

        labels = next_labels
        stats = compute_patch_stats(working, labels)
        log(f"iteration {step + 1} done: patch_count={len(stats)} boundary_moved={moved if args.enable_boundary else 0} merge_accept={merged}")

        if not changed:
            # convergence
            break

        if args.enable_annealing:
            temp = max(args.anneal_temp_min, temp - temp_decay)

    report.update(
        {
            "output_patch_count": int(len(stats)),
            "split_count": int(total_split),
            "boundary_moved_points": int(total_boundary_ok),
            "boundary_rejected_cells": int(total_boundary_reject),
            "merge_accept": int(total_merge),
            "merge_reject": int(total_merge_reject),
            "split_log_count": int(len(split_log)),
            "boundary_log_accept_count": int(len(boundary_accept_log)),
            "boundary_log_reject_count": int(len(boundary_reject_log)),
            "merge_log_count": int(len(merge_log)),
            "split_reason_counts": summarize_log_reasons(split_log),
            "boundary_reason_counts": summarize_log_reasons(boundary_reject_log),
            "merge_status_counts": summarize_merge_status(merge_log),
            "merge_reject_reason_counts": summarize_log_reasons([row for row in merge_log if row.get("status") == "reject"]),
            "params": vars(args),
        }
    )

    # final purity markers
    high_entropy = []
    for pid, s in stats.items():
        e = entropy(s.bucket_counts)
        if e > args.entropy_split_threshold:
            s.conflict_flags = sorted(set((s.conflict_flags or []) + ["high_entropy_post" ]))
            high_entropy.append(pid)
    report["output_high_entropy_count"] = int(len(high_entropy))

    return labels, report, split_log, boundary_accept_log + boundary_reject_log, merge_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iters", type=int, default=6)

    parser.add_argument("--enable-split", action="store_true")
    parser.add_argument("--enable-boundary", action="store_true")
    parser.add_argument("--enable-annealing", action="store_true")

    parser.add_argument("--dirty-min-voxels", type=int, default=40000)
    parser.add_argument("--dirty-entropy-min-voxels", type=int, default=70000)
    parser.add_argument("--dirty-min-height", type=float, default=1.6)
    parser.add_argument("--entropy-split-threshold", type=float, default=1.1)
    parser.add_argument("--split-min-component-voxels", type=int, default=1400)
    parser.add_argument("--residual-component-voxels", type=int, default=420)

    parser.add_argument("--internal-normal-dot", type=float, default=0.52)
    parser.add_argument("--internal-color-distance", type=float, default=55.0)

    parser.add_argument("--fine-voxel-size", type=float, default=0.05)
    parser.add_argument("--min-boundary-owner-score", type=float, default=0.50)
    parser.add_argument("--boundary-margin", type=float, default=0.02)
    parser.add_argument("--surface-guard", type=float, default=0.10)
    parser.add_argument("--patch-size-prior", type=float, default=2600.0)
    parser.add_argument("--boundary-shuffle-ratio", type=float, default=0.02)
    parser.add_argument("--max-shuffle-swaps", type=int, default=200)

    parser.add_argument("--min-merge-gain", type=float, default=0.35)
    parser.add_argument("--min-anchor-voxels", type=int, default=900)
    parser.add_argument("--merge-min-neighbor-support", type=float, default=0.08)
    parser.add_argument("--max-merge-candidates", type=int, default=180000)
    parser.add_argument("--surface-merge-penalty", type=float, default=0.06)
    parser.add_argument("--max-bbox-gap", type=float, default=0.55)
    parser.add_argument("--max-color-distance", type=float, default=130.0)
    parser.add_argument("--pair-share-norm", type=float, default=5000.0)
    parser.add_argument("--patch-target-size", type=float, default=3200.0)
    parser.add_argument("--max-patch-sources", type=float, default=18)
    parser.add_argument("--enable-attachment-merge", action="store_true")
    parser.add_argument("--attachment-min-score", type=float, default=0.76)
    parser.add_argument("--attachment-min-contact-ratio", type=float, default=0.10)
    parser.add_argument("--attachment-min-shared-edges", type=int, default=1)
    parser.add_argument("--attachment-max-color-distance", type=float, default=65.0)
    parser.add_argument("--attachment-min-normal-score", type=float, default=0.45)
    parser.add_argument("--attachment-max-bbox-gap", type=float, default=0.05)
    parser.add_argument("--attachment-max-fragment-voxels", type=int, default=1200)
    parser.add_argument("--attachment-min-anchor-voxels", type=int, default=100000)
    parser.add_argument("--attachment-min-size-ratio", type=float, default=500.0)
    parser.add_argument("--attachment-contact-norm", type=float, default=0.20)
    parser.add_argument("--attachment-color-weight", type=float, default=0.30)
    parser.add_argument("--attachment-normal-weight", type=float, default=0.18)
    parser.add_argument("--attachment-bucket-weight", type=float, default=0.14)
    parser.add_argument("--attachment-contact-weight", type=float, default=0.28)
    parser.add_argument("--attachment-gap-weight", type=float, default=0.10)
    parser.add_argument("--attachment-use-contact-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-overlap-merge-candidates", action="store_true")
    parser.add_argument("--overlap-candidate-top-n", type=int, default=2000)
    parser.add_argument("--overlap-candidate-min-ratio", type=float, default=0.65)
    parser.add_argument("--overlap-candidate-min-iou", type=float, default=0.12)
    parser.add_argument("--overlap-candidate-max-centroid", type=float, default=20.0)
    parser.add_argument("--overlap-candidate-long-min-ratio", type=float, default=0.90)
    parser.add_argument("--overlap-candidate-max-pairs", type=int, default=60000)
    parser.add_argument("--enable-fine-overlap-merge-candidates", action="store_true")
    parser.add_argument("--fine-overlap-voxel-size", type=float, default=0.05)
    parser.add_argument("--fine-overlap-min-cells", type=int, default=4)
    parser.add_argument("--fine-overlap-min-ratio", type=float, default=0.50)
    parser.add_argument("--fine-overlap-max-labels-per-cell", type=int, default=8)
    parser.add_argument("--fine-overlap-max-pairs", type=int, default=80000)
    parser.add_argument("--overlap-only-require-fine-overlap", action="store_true")

    parser.add_argument("--anneal-temp-start", type=float, default=0.33)
    parser.add_argument("--anneal-temp-min", type=float, default=0.02)

    parser.add_argument("--preview-stride", type=int, default=5)
    parser.add_argument("--output-stem", default="geo_patches_energy_v3")
    parser.add_argument("--max-source-patch-ids", type=int, default=24)
    parser.add_argument("--max-log-rows", type=int, default=30000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log(f"read region input: {args.region_input}")
    arrays, src, dst = read_region_input(args.region_input)
    log(f"read labels: {args.labels}")
    labels = read_labels(args.labels)

    log("optimize")
    optimized, report, split_log, boundary_log, merge_log = optimize(
        arrays,
        labels,
        src,
        dst,
        args,
    )

    stats = compute_patch_stats(arrays, optimized)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.output_stem
    if args.overlap_only_require_fine_overlap:
        report["schema"] = "geo-patch-energy-graph-v6"
    elif args.enable_fine_overlap_merge_candidates:
        report["schema"] = "geo-patch-energy-graph-v5"
    else:
        report["schema"] = "geo-patch-energy-graph-v4"
    report["region_input"] = str(args.region_input)
    report["labels_in"] = str(args.labels)
    log("write preview ply")
    report["preview_points"] = write_ply(out_dir / f"{stem}_stride{args.preview_stride}.ply", arrays, optimized, args.preview_stride)
    report["output_labels"] = str(out_dir / f"{stem}_labels.bin")
    report["output_jsonl"] = str(out_dir / f"{stem}.jsonl")
    report["output_report"] = str(out_dir / f"{stem}_report.json")
    report["output_split_log"] = str(out_dir / "split_log.jsonl")
    report["output_boundary_log"] = str(out_dir / "boundary_log.jsonl")
    report["output_merge_log"] = str(out_dir / "merge_log.jsonl")
    log("write labels")
    write_labels(Path(report["output_labels"]), optimized)
    log("write jsonl")
    report["jsonl_patch_count"] = write_jsonl(out_dir / f"{stem}.jsonl", stats, args)

    log("write logs and report")
    (out_dir / "split_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split_log), encoding="utf-8")
    (out_dir / "boundary_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in boundary_log), encoding="utf-8")
    (out_dir / "merge_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merge_log), encoding="utf-8")
    (out_dir / f"{stem}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
