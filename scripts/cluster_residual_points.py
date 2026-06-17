#!/usr/bin/env python3
"""Cluster residual RGB points into geometry/color objects.

This is the second stage after project_priority_masks_to_lx.py:

1. priority classes remove sky, ground, wall, grass, car, railing
2. remaining residual points are clustered by 3D voxel connectivity
3. neighboring voxels only merge when their mean RGB distance is acceptable

The output intentionally does not invent semantic labels. It creates object
identities and object statistics for a later VLM/LLM naming or review stage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_binary_xyzrgb_priority_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header.append(line.decode("ascii", errors="ignore").strip())
            if line.strip() == b"end_header":
                break
        vertex_count = 0
        for line in header:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Only binary_little_endian PLY is supported.")
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("priority", "u1"),
        ])
        data = np.frombuffer(f.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
    points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    colors = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.uint8)
    priority = data["priority"].astype(np.uint8)
    return points, colors, priority


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.uint8)

    def find(self, x: int) -> int:
        p = int(self.parent[x])
        if p != x:
            self.parent[x] = self.find(p)
        return int(self.parent[x])

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def object_color(object_id: int) -> tuple[int, int, int]:
    x = (object_id * 1103515245 + 12345) & 0xFFFFFFFF
    r = 64 + ((x >> 0) & 127)
    g = 64 + ((x >> 8) & 127)
    b = 64 + ((x >> 16) & 127)
    return int(r), int(g), int(b)


def write_object_ply(path: Path, points: np.ndarray, object_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for oid in np.unique(object_ids):
        colors[object_ids == oid] = (40, 40, 40) if oid == 0 else object_color(int(oid))
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("object", "<u4"),
    ])
    data = np.empty(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]
    data["object"] = object_ids.astype(np.uint32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uint object\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(data.tobytes())


def pca_stats(points: np.ndarray) -> dict:
    centered = points.astype(np.float64) - points.mean(axis=0, keepdims=True)
    if len(points) < 3:
        eigvals = np.zeros(3, dtype=np.float64)
        eigvecs = np.eye(3, dtype=np.float64)
    else:
        cov = (centered.T @ centered) / max(len(points) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    extents = points.max(axis=0) - points.min(axis=0)
    thickness_rms = float(np.sqrt(max(eigvals[-1], 0.0)))
    spread_rms = float(np.sqrt(max(eigvals[0], 0.0)))
    planarity = 1.0 - (thickness_rms / max(spread_rms, 1e-6))
    normal = eigvecs[:, -1].astype(float).tolist()
    return {
        "pca_eigenvalues": eigvals.astype(float).tolist(),
        "pca_normal": normal,
        "thickness_rms": thickness_rms,
        "spread_rms": spread_rms,
        "planarity": float(planarity),
        "extent": extents.astype(float).tolist(),
        "max_extent": float(extents.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument("--color-threshold", type=float, default=55.0)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--surface-min-points", type=int, default=5000)
    parser.add_argument("--surface-min-extent", type=float, default=3.0)
    parser.add_argument("--surface-thickness-rms", type=float, default=0.18)
    parser.add_argument("--max-points", type=int, default=0, help="Optional deterministic downsample for smoke tests.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    points, colors, priority = read_binary_xyzrgb_priority_ply(args.input_ply)
    if args.max_points and len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(points), args.max_points, replace=False))
        points, colors, priority = points[idx], colors[idx], priority[idx]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(points) == 0:
        raise SystemExit("No residual points to cluster.")

    voxels = np.floor(points / float(args.voxel_size)).astype(np.int32)
    unique_voxels, inv, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    voxel_count = len(unique_voxels)
    color_sum = np.zeros((voxel_count, 3), dtype=np.float64)
    np.add.at(color_sum, inv, colors.astype(np.float64))
    mean_color = color_sum / np.maximum(counts[:, None], 1)

    voxel_lookup = {tuple(v.tolist()): i for i, v in enumerate(unique_voxels)}
    uf = UnionFind(voxel_count)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0)
    ]
    edges_tested = 0
    edges_merged = 0
    for i, voxel in enumerate(unique_voxels):
        base = tuple(voxel.tolist())
        for off in offsets:
            j = voxel_lookup.get((base[0] + off[0], base[1] + off[1], base[2] + off[2]))
            if j is None:
                continue
            edges_tested += 1
            if np.linalg.norm(mean_color[i] - mean_color[j]) <= args.color_threshold:
                uf.union(i, j)
                edges_merged += 1

    roots = np.array([uf.find(i) for i in range(voxel_count)], dtype=np.int32)
    root_for_point = roots[inv]
    root_point_counts = Counter(int(x) for x in root_for_point.tolist())
    kept_roots = [root for root, count in root_point_counts.items() if count >= args.min_points]
    kept_roots.sort(key=lambda r: (-root_point_counts[r], r))
    root_to_object = {root: i + 1 for i, root in enumerate(kept_roots)}

    object_ids = np.zeros(len(points), dtype=np.uint32)
    for root, oid in root_to_object.items():
        object_ids[root_for_point == root] = oid

    object_path = args.output_dir / "residual_objects.ply"
    write_object_ply(object_path, points, object_ids)

    jsonl_path = args.output_dir / "residual_objects.jsonl"
    objects = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for root, oid in root_to_object.items():
            mask = root_for_point == root
            pts = points[mask]
            cols = colors[mask].astype(np.float32)
            geom = pca_stats(pts)
            is_surface_candidate = (
                int(mask.sum()) >= args.surface_min_points
                and geom["max_extent"] >= args.surface_min_extent
                and geom["thickness_rms"] <= args.surface_thickness_rms
            )
            obj = {
                "object_id": int(oid),
                "point_count": int(mask.sum()),
                "voxel_count": int((roots == root).sum()),
                "centroid": pts.mean(axis=0).astype(float).tolist(),
                "bbox_min": pts.min(axis=0).astype(float).tolist(),
                "bbox_max": pts.max(axis=0).astype(float).tolist(),
                "mean_color": cols.mean(axis=0).astype(float).tolist(),
                **geom,
                "source": str(args.input_ply),
                "semantic_label": "residual_surface_candidate" if is_surface_candidate else "unlabeled_residual",
                "description": "",
                "status": "hold_as_surface_residual" if is_surface_candidate else "needs_semantic_review",
            }
            objects.append(obj)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    report = {
        "input_ply": str(args.input_ply),
        "output_dir": str(args.output_dir),
        "voxel_size": args.voxel_size,
        "color_threshold": args.color_threshold,
        "min_points": args.min_points,
        "point_count": int(len(points)),
        "voxel_count": int(voxel_count),
        "edges_tested": int(edges_tested),
        "edges_merged": int(edges_merged),
        "object_count": int(len(objects)),
        "assigned_points": int((object_ids > 0).sum()),
        "noise_points": int((object_ids == 0).sum()),
        "object_ply": str(object_path),
        "objects_jsonl": str(jsonl_path),
    }
    (args.output_dir / "residual_cluster_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
