#!/usr/bin/env python3
"""Optimize geometry patch labels by absorbing small patches into anchors.

This is a post-region-grow graph optimization stage.  It does not change the
region-growing core; it treats the C++ output as conservative superpatches and
only allows small patches to merge into nearby large anchors when the merge has
positive evidence from spatial adjacency, color, geometry bucket compatibility,
and normal/charts.
"""

from __future__ import annotations

import argparse
import json
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


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1)
    out = np.zeros_like(value, dtype=np.float64)
    ok = norm > 1e-9
    out[ok] = value[ok] / norm[ok, None]
    return out


def dominant_geometry(bucket_counts: Counter[int]) -> str:
    if not bucket_counts:
        return "unknown"
    bucket, count = bucket_counts.most_common(1)[0]
    ratio = count / max(sum(bucket_counts.values()), 1)
    return BUCKET_NAMES[int(bucket)] if ratio >= 0.65 else "mixed"


def compute_patch_stats(arrays: dict[str, np.ndarray], labels: np.ndarray) -> dict[int, PatchStats]:
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
    ends = np.r_[starts[1:], len(sorted_labels)]
    patch_ids = sorted_labels[starts].astype(np.int32, copy=False)
    counts = (ends - starts).astype(np.int64, copy=False)
    xyz = arrays["xyz"][order].astype(np.float64, copy=False)
    rgb = arrays["rgb"][order].astype(np.float64, copy=False)
    normal = arrays["normal"][order].astype(np.float64, copy=False)
    buckets = arrays["buckets"][order]
    xyz_sum = np.add.reduceat(xyz, starts, axis=0)
    rgb_sum = np.add.reduceat(rgb, starts, axis=0)
    normal_sum = np.add.reduceat(normal, starts, axis=0)
    bbox_min = np.minimum.reduceat(xyz, starts, axis=0)
    bbox_max = np.maximum.reduceat(xyz, starts, axis=0)
    normals = normalize_rows(normal_sum)

    stats: dict[int, PatchStats] = {}
    for i, patch_id in enumerate(patch_ids.tolist()):
        bucket_counts = Counter(int(v) for v in buckets[starts[i] : ends[i]].tolist())
        stats[int(patch_id)] = PatchStats(
            patch_id=int(patch_id),
            count=int(counts[i]),
            centroid=xyz_sum[i] / max(float(counts[i]), 1.0),
            mean_rgb=rgb_sum[i] / max(float(counts[i]), 1.0),
            mean_normal=normals[i],
            bbox_min=bbox_min[i],
            bbox_max=bbox_max[i],
            bucket_counts=bucket_counts,
            geometry_type=dominant_geometry(bucket_counts),
            source_patch_ids={int(patch_id)},
        )
    return stats


def compatible_bucket_score(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if "unknown" in {a, b}:
        return 0.65
    if {a, b} <= {"rough_mixed", "thin_linear", "unknown"}:
        return 0.75
    if {a, b} in [{"horizontal", "rough_mixed"}, {"horizontal", "thin_linear"}]:
        return 0.45
    if {a, b} in [{"vertical", "rough_mixed"}, {"vertical", "thin_linear"}]:
        return 0.35
    return 0.05


def normal_score(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-9 or bn < 1e-9:
        return 0.5
    cos = abs(float(np.dot(a, b) / (an * bn)))
    return max(0.0, min(1.0, cos))


def bbox_gap(a: PatchStats, b: PatchStats) -> float:
    gap = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    return float(np.linalg.norm(gap))


def merge_gain(small: PatchStats, anchor: PatchStats, shared_edges: int, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    color_dist = float(np.linalg.norm(small.mean_rgb - anchor.mean_rgb))
    color_score = max(0.0, min(1.0, 1.0 - color_dist / max(args.max_color_distance, 1e-6)))
    bucket = compatible_bucket_score(small.geometry_type, anchor.geometry_type)
    normal = normal_score(small.mean_normal, anchor.mean_normal)
    size_score = min(1.0, shared_edges / max(float(small.count), 1.0))
    gap = bbox_gap(small, anchor)
    gap_score = max(0.0, min(1.0, 1.0 - gap / max(args.max_bbox_gap, 1e-6)))
    # Stable surfaces should not absorb a normal-incompatible small patch just
    # because it is adjacent in sparse LiDAR space.
    stable_guard = 0.0
    if anchor.geometry_type in {"horizontal", "vertical"} and small.geometry_type not in {anchor.geometry_type, "unknown"}:
        stable_guard = 0.18
    gain = (
        0.34 * color_score
        + 0.24 * bucket
        + 0.16 * normal
        + 0.18 * size_score
        + 0.08 * gap_score
        - stable_guard
    )
    details = {
        "gain": gain,
        "color_score": color_score,
        "bucket_score": bucket,
        "normal_score": normal,
        "edge_support": size_score,
        "gap_score": gap_score,
        "stable_guard": stable_guard,
        "shared_edges": float(shared_edges),
    }
    return gain, details


def merge_stats(anchor: PatchStats, small: PatchStats) -> None:
    total = anchor.count + small.count
    anchor.centroid = (anchor.centroid * anchor.count + small.centroid * small.count) / total
    anchor.mean_rgb = (anchor.mean_rgb * anchor.count + small.mean_rgb * small.count) / total
    normal = anchor.mean_normal * anchor.count + small.mean_normal * small.count
    norm = np.linalg.norm(normal)
    anchor.mean_normal = normal / norm if norm > 1e-9 else normal
    anchor.bbox_min = np.minimum(anchor.bbox_min, small.bbox_min)
    anchor.bbox_max = np.maximum(anchor.bbox_max, small.bbox_max)
    anchor.bucket_counts.update(small.bucket_counts)
    anchor.count = total
    anchor.geometry_type = dominant_geometry(anchor.bucket_counts)
    anchor.source_patch_ids.update(small.source_patch_ids)


def build_patch_edges(labels: np.ndarray, src: np.ndarray, dst: np.ndarray) -> Counter[tuple[int, int]]:
    a = labels[src]
    b = labels[dst]
    mask = a != b
    edges: Counter[tuple[int, int]] = Counter()
    for pa, pb in zip(a[mask].tolist(), b[mask].tolist(), strict=True):
        if pa > pb:
            pa, pb = pb, pa
        edges[(int(pa), int(pb))] += 1
    return edges


def optimize(arrays: dict[str, np.ndarray], labels: np.ndarray, src: np.ndarray, dst: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    stats = compute_patch_stats(arrays, labels)
    patch_edges = build_patch_edges(labels, src, dst)
    parent = {patch_id: patch_id for patch_id in stats}
    small_ids = {pid for pid, s in stats.items() if s.count < args.small_patch_voxels}
    merge_log: list[dict[str, Any]] = []

    # One conservative pass: each original small patch may be absorbed by one
    # current anchor. This avoids unstable cascading large-large merges.
    for small_id in sorted(small_ids, key=lambda pid: stats[pid].count):
        if parent.get(small_id) != small_id:
            continue
        small = stats[small_id]
        candidates: list[tuple[float, int, dict[str, float]]] = []
        for (a, b), shared in patch_edges.items():
            if small_id not in {a, b}:
                continue
            other = b if a == small_id else a
            anchor_id = parent.get(other, other)
            if anchor_id == small_id or anchor_id not in stats:
                continue
            anchor = stats[anchor_id]
            if anchor.count < args.anchor_min_voxels:
                continue
            gain, details = merge_gain(small, anchor, shared, args)
            if gain >= args.min_gain:
                candidates.append((gain, anchor_id, details))
        if not candidates:
            continue
        candidates.sort(key=lambda row: row[0], reverse=True)
        gain, anchor_id, details = candidates[0]
        merge_stats(stats[anchor_id], small)
        parent[small_id] = anchor_id
        del stats[small_id]
        merge_log.append(
            {
                "small_patch_id": small_id,
                "anchor_patch_id": anchor_id,
                "small_voxels": small.count,
                "anchor_voxels_after": stats[anchor_id].count,
                **details,
            }
        )

    remapped = labels.copy()
    if merge_log:
        table = np.arange(int(labels.max()) + 1, dtype=np.int32)
        for old, new in parent.items():
            table[int(old)] = int(new)
        remapped = table[labels]
    report = {
        "schema": "geo-patch-merge-optimizer/v1",
        "initial_patch_count": int(len(parent)),
        "final_patch_count": int(len(set(parent.values()))),
        "merged_small_patch_count": int(len(merge_log)),
        "remaining_small_patch_count": int(sum(1 for s in stats.values() if s.count < args.small_patch_voxels)),
        "params": vars(args),
    }
    return remapped, report, merge_log


def patch_color(patch_id: int) -> tuple[int, int, int]:
    rng = (int(patch_id) * 1103515245 + 12345) & 0x7FFFFFFF
    return (40 + rng % 206, 40 + (rng // 257) % 206, 40 + (rng // 65537) % 206)


def write_ply(path: Path, arrays: dict[str, np.ndarray], labels: np.ndarray, stride: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = np.arange(0, len(labels), stride, dtype=np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(selected)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for i in selected.tolist():
            x, y, z = arrays["xyz"][i]
            r, g, b = patch_color(int(labels[i]))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {int(labels[i])} 1\n")
    return int(len(selected))


def write_jsonl(path: Path, arrays: dict[str, np.ndarray], labels: np.ndarray, args: argparse.Namespace) -> int:
    stats = compute_patch_stats(arrays, labels)
    with path.open("w", encoding="utf-8") as f:
        for patch_id in sorted(stats):
            s = stats[patch_id]
            row = {
                "patch_id": patch_id,
                "object": patch_id,
                "voxel_count": s.count,
                "status": "small_patch" if s.count < args.small_patch_voxels else "geo_patch",
                "geometry_type": s.geometry_type,
                "semantic_label": s.geometry_type,
                "description": f"optimized geometry patch: {s.geometry_type}",
                "bucket_counts": {BUCKET_NAMES[k]: int(v) for k, v in s.bucket_counts.items()},
                "centroid": s.centroid.astype(float).tolist(),
                "bbox_3d": {"min": s.bbox_min.astype(float).tolist(), "max": s.bbox_max.astype(float).tolist()},
                "extent": (s.bbox_max - s.bbox_min).astype(float).tolist(),
                "mean_rgb": s.mean_rgb.astype(float).tolist(),
                "mean_normal": s.mean_normal.astype(float).tolist(),
                "source_patch_count": len(s.source_patch_ids),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--small-patch-voxels", type=int, default=8)
    parser.add_argument("--anchor-min-voxels", type=int, default=64)
    parser.add_argument("--min-gain", type=float, default=0.66)
    parser.add_argument("--max-color-distance", type=float, default=120.0)
    parser.add_argument("--max-bbox-gap", type=float, default=0.35)
    parser.add_argument("--preview-stride", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"label count mismatch: labels={len(labels)} voxels={len(arrays['xyz'])}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    optimized, report, merge_log = optimize(arrays, labels, src, dst, args)
    report["output_ply"] = str(output_dir / f"geo_patches_optimized_stride{args.preview_stride}.ply")
    report["output_jsonl"] = str(output_dir / "geo_patches_optimized.jsonl")
    report["preview_points"] = write_ply(Path(report["output_ply"]), arrays, optimized, args.preview_stride)
    report["jsonl_patch_count"] = write_jsonl(Path(report["output_jsonl"]), arrays, optimized, args)
    (output_dir / "merge_log.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merge_log),
        encoding="utf-8",
    )
    (output_dir / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
