#!/usr/bin/env python3
"""Build geometry patches with a vectorized similarity graph.

This is a dense-data counterpart to build_geo_patch_demo.py.  It keeps the
same voxel/PCA feature inputs, but replaces Python FIFO region growing with:

1. torch/scipy-friendly array conversion
2. batched neighbor edge scoring
3. sparse connected components

The graph route is intentionally pairwise.  It is a speed/scaling baseline, not
a replacement for future model-aware region growing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_geo_patch_demo import (
    compute_local_features,
    compute_local_features_torch,
    geometry_bucket,
    neighbor_offsets,
    read_voxels,
)


BUCKET_IDS = {
    "unknown": 0,
    "horizontal": 1,
    "vertical": 2,
    "thin_linear": 3,
    "rough_mixed": 4,
}
ID_BUCKETS = {value: key for key, value in BUCKET_IDS.items()}


def clamp01_np(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0.0, 1.0)


def voxel_arrays(voxels: dict[tuple[int, int, int], dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = np.asarray(list(voxels), dtype=np.int64)
    xyz = np.vstack([voxels[key]["xyz"] for key in voxels]).astype(np.float32)
    rgb = np.vstack([voxels[key]["rgb"] for key in voxels]).astype(np.float32)
    normal = np.vstack([voxels[key]["normal"] for key in voxels]).astype(np.float32)
    roughness = np.asarray([voxels[key]["roughness"] for key in voxels], dtype=np.float32)
    planarity = np.asarray([voxels[key]["planarity"] for key in voxels], dtype=np.float32)
    linearity = np.asarray([voxels[key]["linearity"] for key in voxels], dtype=np.float32)
    height_range = np.asarray([voxels[key]["height_range"] for key in voxels], dtype=np.float32)
    local_color_std = np.asarray([voxels[key]["local_color_std"] for key in voxels], dtype=np.float32)
    buckets = np.asarray([BUCKET_IDS[str(voxels[key]["bucket"])] for key in voxels], dtype=np.int16)
    return {
        "keys": keys,
        "xyz": xyz,
        "rgb": rgb,
        "normal": normal,
        "roughness": roughness,
        "planarity": planarity,
        "linearity": linearity,
        "height_range": height_range,
        "local_color_std": local_color_std,
        "buckets": buckets,
    }


def sorted_linear_index(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    mins = keys.min(axis=0)
    shifted = keys - mins[None, :]
    spans = shifted.max(axis=0) + 1
    stride_y = int(spans[2])
    stride_x = int(spans[1] * spans[2])
    linear = shifted[:, 0] * stride_x + shifted[:, 1] * stride_y + shifted[:, 2]
    order = np.argsort(linear)
    return linear, order, linear[order], (stride_x, stride_y)


def positive_offsets(radius: int) -> list[tuple[int, int, int]]:
    offsets = []
    for dx, dy, dz in neighbor_offsets(radius):
        if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
            offsets.append((dx, dy, dz))
    return offsets


def bucket_score_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same = a == b
    unknown = (a == BUCKET_IDS["unknown"]) | (b == BUCKET_IDS["unknown"])
    rough_horizontal = ((a == BUCKET_IDS["rough_mixed"]) & (b == BUCKET_IDS["horizontal"])) | (
        (b == BUCKET_IDS["rough_mixed"]) & (a == BUCKET_IDS["horizontal"])
    )
    rough_vertical = ((a == BUCKET_IDS["rough_mixed"]) & (b == BUCKET_IDS["vertical"])) | (
        (b == BUCKET_IDS["rough_mixed"]) & (a == BUCKET_IDS["vertical"])
    )
    out = np.full(a.shape, 0.15, dtype=np.float32)
    out[unknown] = 0.72
    out[rough_horizontal | rough_vertical] = 0.55
    out[same] = 1.0
    return out


def collect_edges(
    arrays: dict[str, np.ndarray],
    connect_radius_voxels: int,
    min_edge_score: float,
    max_color_distance: float,
    max_height_delta: float,
    max_normal_angle: float,
    max_plane_residual: float,
    bucket_guard: str,
    weights: dict[str, float],
    color_bridge_distance_factor: float,
    color_bridge_texture_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    keys = arrays["keys"]
    linear, order, sorted_linear, (stride_x, stride_y) = sorted_linear_index(keys)
    n = len(keys)
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []

    for dx, dy, dz in positive_offsets(connect_radius_voxels):
        offset_linear = dx * stride_x + dy * stride_y + dz
        query = linear + offset_linear
        pos = np.searchsorted(sorted_linear, query)
        in_bounds = pos < n
        safe_pos = np.minimum(pos, n - 1)
        found = in_bounds & (sorted_linear[safe_pos] == query)
        if not np.any(found):
            continue
        src = np.nonzero(found)[0].astype(np.int64)
        dst = order[safe_pos[found]].astype(np.int64)
        keep = edge_keep(
            arrays,
            src,
            dst,
            min_edge_score,
            max_color_distance,
            max_height_delta,
            max_normal_angle,
            max_plane_residual,
            bucket_guard,
            weights,
            color_bridge_distance_factor,
            color_bridge_texture_delta,
        )
        if np.any(keep):
            src_chunks.append(src[keep].astype(np.int32))
            dst_chunks.append(dst[keep].astype(np.int32))
    if not src_chunks:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(src_chunks), np.concatenate(dst_chunks)


def edge_keep(
    arrays: dict[str, np.ndarray],
    src: np.ndarray,
    dst: np.ndarray,
    min_edge_score: float,
    max_color_distance: float,
    max_height_delta: float,
    max_normal_angle: float,
    max_plane_residual: float,
    bucket_guard: str,
    weights: dict[str, float],
    color_bridge_distance_factor: float,
    color_bridge_texture_delta: float,
) -> np.ndarray:
    xyz_a = arrays["xyz"][src]
    xyz_b = arrays["xyz"][dst]
    rgb_a = arrays["rgb"][src]
    rgb_b = arrays["rgb"][dst]
    n_a = arrays["normal"][src]
    n_b = arrays["normal"][dst]
    dot = np.abs(np.sum(n_a * n_b, axis=1))
    dot = np.clip(dot, 0.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    rgb_dist = np.linalg.norm(rgb_a - rgb_b, axis=1)
    dz = np.abs(xyz_a[:, 2] - xyz_b[:, 2])
    rough_delta = np.abs(arrays["roughness"][src] - arrays["roughness"][dst])
    planarity_delta = np.abs(arrays["planarity"][src] - arrays["planarity"][dst])
    linearity_delta = np.abs(arrays["linearity"][src] - arrays["linearity"][dst])
    color_std_delta = np.abs(arrays["local_color_std"][src] - arrays["local_color_std"][dst])
    height_range_delta = np.abs(arrays["height_range"][src] - arrays["height_range"][dst])
    plane_residual = np.abs(np.sum((xyz_b - xyz_a) * n_a, axis=1))

    veto = bucket_guard_veto(arrays["buckets"][src], arrays["buckets"][dst], bucket_guard)
    veto |= (rgb_dist > max_color_distance * 1.75) & (color_std_delta > 55.0)
    veto |= rough_delta > 0.36
    veto |= dz > max_height_delta * 3.0
    veto |= plane_residual > max_plane_residual * 3.5
    if bucket_guard in {"same-bucket-or-color", "same-bucket-or-fine-color"}:
        if bucket_guard == "same-bucket-or-fine-color":
            bridge_buckets = np.isin(
                arrays["buckets"][src],
                [BUCKET_IDS["rough_mixed"], BUCKET_IDS["thin_linear"], BUCKET_IDS["unknown"]],
            ) & np.isin(
                arrays["buckets"][dst],
                [BUCKET_IDS["rough_mixed"], BUCKET_IDS["thin_linear"], BUCKET_IDS["unknown"]],
            )
        else:
            bridge_buckets = np.ones_like(veto, dtype=bool)
        color_bridge = (
            bridge_buckets
            & (rgb_dist <= max_color_distance * color_bridge_distance_factor)
            & (color_std_delta <= color_bridge_texture_delta)
            & (rough_delta <= 0.22)
            & (dz <= max_height_delta * 1.6)
            & (plane_residual <= max_plane_residual * 2.0)
        )
        veto &= ~color_bridge

    scores = {
        "color": clamp01_np(1.0 - rgb_dist / max(max_color_distance, 1e-6)),
        "color_texture": clamp01_np(1.0 - color_std_delta / 85.0),
        "roughness": clamp01_np(1.0 - rough_delta / 0.24),
        "planarity": clamp01_np(1.0 - planarity_delta / 0.50),
        "linearity": clamp01_np(1.0 - linearity_delta / 0.50),
        "height_range": clamp01_np(1.0 - height_range_delta / 0.28),
        "height": clamp01_np(1.0 - dz / max(max_height_delta * 2.8, 1e-6)),
        "bucket": bucket_score_np(arrays["buckets"][src], arrays["buckets"][dst]),
        "normal": clamp01_np(1.0 - angle / max(max_normal_angle * 1.8, 1e-6)),
        "plane": clamp01_np(1.0 - plane_residual / max(max_plane_residual * 2.5, 1e-6)),
    }
    shape_score = (
        0.38 * scores["roughness"]
        + 0.24 * scores["linearity"]
        + 0.24 * scores["planarity"]
        + 0.14 * scores["height_range"]
    )
    texture_score = 0.62 * scores["color"] + 0.38 * scores["color_texture"]
    total_weight = max(float(sum(weights.values())), 1e-6)
    total = (
        weights["texture"] * texture_score
        + weights["shape"] * shape_score
        + weights["height"] * scores["height"]
        + weights["bucket"] * scores["bucket"]
        + weights["normal"] * scores["normal"]
        + weights["plane"] * scores["plane"]
    ) / total_weight
    return (~veto) & (total >= min_edge_score)


def bucket_guard_veto(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "loose":
        return np.zeros(a.shape, dtype=bool)
    unknown = BUCKET_IDS["unknown"]
    horizontal = BUCKET_IDS["horizontal"]
    vertical = BUCKET_IDS["vertical"]
    thin = BUCKET_IDS["thin_linear"]
    rough = BUCKET_IDS["rough_mixed"]
    same = a == b
    has_unknown = (a == unknown) | (b == unknown)
    veto = np.zeros(a.shape, dtype=bool)
    if mode == "same-or-unknown":
        veto |= ~(same | has_unknown)
    elif mode in {"same-bucket", "same-bucket-or-color", "same-bucket-or-fine-color"}:
        veto |= ~same
    elif mode == "no-rough-bridge":
        has_rough = (a == rough) | (b == rough)
        has_thin = (a == thin) | (b == thin)
        horizontal_vertical = ((a == horizontal) & (b == vertical)) | ((a == vertical) & (b == horizontal))
        # Rough/mixed voxels are the main bridge source. They may merge with
        # themselves, but not connect surfaces to unrelated structures.
        veto |= has_rough & ~same
        # Thin structures should not glue onto walls/floors in the graph pass.
        veto |= has_thin & ~(same | has_unknown)
        veto |= horizontal_vertical
    else:
        raise ValueError(f"unknown bucket guard: {mode}")
    return veto


def connected_labels(n: int, src: np.ndarray, dst: np.ndarray) -> tuple[int, np.ndarray]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(src) == 0:
        return n, np.arange(n, dtype=np.int32)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.uint8)
    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    count, labels = connected_components(graph, directed=False, return_labels=True)
    return int(count), labels.astype(np.int32, copy=False)


def patch_color(patch_id: int) -> tuple[int, int, int]:
    rng = random.Random(int(patch_id) * 1000003)
    return (rng.randint(40, 245), rng.randint(40, 245), rng.randint(40, 245))


def write_outputs(output_dir: Path, arrays: dict[str, np.ndarray], labels: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "geo_patches_graph_random_color.ply"
    jsonl_path = output_dir / "geo_patches_graph.jsonl"
    report_path = output_dir / "geo_patch_graph_report.json"
    n = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_to_patch = {int(label): i + 1 for i, label in enumerate(unique_labels)}
    patch_ids = np.asarray([label_to_patch[int(label)] for label in labels], dtype=np.int32)

    bucket_counts_by_patch: dict[int, Counter[str]] = defaultdict(Counter)
    for patch_id, bucket_id in zip(patch_ids, arrays["buckets"], strict=True):
        bucket_counts_by_patch[int(patch_id)][ID_BUCKETS[int(bucket_id)]] += 1

    patches: list[dict[str, Any]] = []
    for label, count in zip(unique_labels, counts, strict=True):
        patch_id = label_to_patch[int(label)]
        buckets = bucket_counts_by_patch[patch_id]
        dominant_bucket, dominant_count = buckets.most_common(1)[0]
        patches.append(
            {
                "patch_id": patch_id,
                "voxel_count": int(count),
                "status": "small_patch" if int(count) < args.small_patch_voxels else "geo_patch",
                "geometry_type": dominant_bucket if dominant_count / max(int(count), 1) >= 0.65 else "mixed",
                "bucket_counts": dict(buckets),
                "semantic_label": dominant_bucket,
                "description": f"graph geometry patch: {dominant_bucket}",
                "voxel_size": args.voxel_size,
            }
        )

    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for xyz, patch_id in zip(arrays["xyz"], patch_ids, strict=True):
            color = patch_color(int(patch_id))
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {color[0]} {color[1]} {color[2]} {int(patch_id)} 1\n")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in patches:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    geometry_counts = Counter(str(row["geometry_type"]) for row in patches)
    bucket_counts = Counter(ID_BUCKETS[int(bucket)] for bucket in arrays["buckets"])
    report = {
        "schema": "geo-patch-graph/v1",
        "input_ply": str(args.input_ply),
        "output_ply": str(ply_path),
        "output_jsonl": str(jsonl_path),
        "voxel_size": args.voxel_size,
        "voxel_count": n,
        "patch_count": len(patches),
        "small_patch_count": sum(1 for row in patches if row["status"] == "small_patch"),
        "bucket_voxel_counts": dict(bucket_counts),
        "patch_geometry_counts": dict(geometry_counts),
        "params": vars(args),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--voxel-backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--binary-voxel-input", action="store_true")
    parser.add_argument("--feature-backend", choices=("cpu", "torch"), default="torch")
    parser.add_argument("--feature-radius-voxels", type=int, default=3)
    parser.add_argument("--feature-batch-size", type=int, default=8192)
    parser.add_argument("--torch-device", default="cuda:0")
    parser.add_argument("--connect-radius-voxels", type=int, default=1)
    parser.add_argument("--min-edge-score", type=float, default=0.50)
    parser.add_argument("--max-color-distance", type=float, default=135.0)
    parser.add_argument("--max-height-delta", type=float, default=0.22)
    parser.add_argument("--max-normal-angle", type=float, default=58.0)
    parser.add_argument("--max-plane-residual", type=float, default=0.12)
    parser.add_argument(
        "--bucket-guard",
        choices=(
            "loose",
            "same-or-unknown",
            "no-rough-bridge",
            "same-bucket",
            "same-bucket-or-color",
            "same-bucket-or-fine-color",
        ),
        default="loose",
    )
    parser.add_argument("--texture-weight", type=float, default=0.36)
    parser.add_argument("--shape-weight", type=float, default=0.28)
    parser.add_argument("--height-weight", type=float, default=0.12)
    parser.add_argument("--bucket-weight", type=float, default=0.10)
    parser.add_argument("--normal-weight", type=float, default=0.07)
    parser.add_argument("--plane-weight", type=float, default=0.07)
    parser.add_argument("--color-bridge-distance-factor", type=float, default=0.45)
    parser.add_argument("--color-bridge-texture-delta", type=float, default=28.0)
    parser.add_argument("--small-patch-voxels", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    voxels = read_voxels(
        args.input_ply,
        args.voxel_size,
        args.max_points,
        args.voxel_backend,
        args.torch_device,
        args.binary_voxel_input,
    )
    if args.feature_backend == "torch":
        compute_local_features_torch(voxels, args.feature_radius_voxels, args.torch_device, args.feature_batch_size)
    else:
        compute_local_features(voxels, args.feature_radius_voxels)
    for item in voxels.values():
        item["bucket"] = geometry_bucket(item)
    arrays = voxel_arrays(voxels)
    src, dst = collect_edges(
        arrays,
        args.connect_radius_voxels,
        args.min_edge_score,
        args.max_color_distance,
        args.max_height_delta,
        args.max_normal_angle,
        args.max_plane_residual,
        args.bucket_guard,
        {
            "texture": args.texture_weight,
            "shape": args.shape_weight,
            "height": args.height_weight,
            "bucket": args.bucket_weight,
            "normal": args.normal_weight,
            "plane": args.plane_weight,
        },
        args.color_bridge_distance_factor,
        args.color_bridge_texture_delta,
    )
    _count, labels = connected_labels(len(arrays["keys"]), src, dst)
    write_outputs(args.output_dir, arrays, labels, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
