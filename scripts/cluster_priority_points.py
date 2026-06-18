#!/usr/bin/env python3
"""Cluster priority-layer points into object records.

`project_priority_masks_to_lx.py` removes known classes before residual
clustering. That is correct for processing, but a class-only priority layer is
too coarse for QA and downstream object reasoning: a car or railing should have
its own object id. This script clusters each priority class independently by
3D voxel connectivity and writes a viewer-friendly ASCII PLY plus JSONL object
metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PRIORITY_CLASSES = {
    1: ("floor", 3, "priority_ground_object"),
    2: ("wall", 2, "priority_wall_object"),
    3: ("grass", 5, "priority_grass_object"),
    4: ("car", 8, "priority_car_object"),
    5: ("railing", 9, "priority_railing_object"),
}

PLY_TYPE_MAP = {
    "float": "<f4", "float32": "<f4", "double": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.uint8)

    def find(self, x: int) -> int:
        parent = int(self.parent[x])
        if parent != x:
            self.parent[x] = self.find(parent)
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


def read_ply(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open("rb") as f:
        fmt = "ascii"
        props: list[str] = []
        prop_types: list[str] = []
        vertex_count = 0
        in_vertex = False
        header_lines = 0
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Invalid PLY header: {path}")
            header_lines += 1
            line = raw.decode("ascii", errors="ignore").strip()
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "format":
                fmt = parts[1]
            elif len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                prop_types.append(parts[1])
                props.append(parts[-1])
            elif line == "end_header":
                break

    if fmt == "ascii":
        data = np.loadtxt(path, skiprows=header_lines, dtype=np.float64, max_rows=vertex_count)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data, props

    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {fmt}")
    dtype = np.dtype([(name, PLY_TYPE_MAP[ptype]) for ptype, name in zip(prop_types, props)])
    with path.open("rb") as f:
        while f.readline().strip() != b"end_header":
            pass
        arr = np.frombuffer(f.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
    data = np.column_stack([arr[name] for name in props]).astype(np.float64)
    return data, props


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
    thickness = float(np.sqrt(max(eigvals[-1], 0.0)))
    spread = float(np.sqrt(max(eigvals[0], 0.0)))
    return {
        "pca_eigenvalues": eigvals.astype(float).tolist(),
        "pca_normal": eigvecs[:, -1].astype(float).tolist(),
        "thickness_rms": thickness,
        "spread_rms": spread,
        "planarity": float(1.0 - thickness / max(spread, 1e-6)),
        "extent": extents.astype(float).tolist(),
        "max_extent": float(extents.max()),
    }


def component_roots(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voxels = np.floor(points / float(voxel_size)).astype(np.int32)
    unique_voxels, inv = np.unique(voxels, axis=0, return_inverse=True)
    lookup = {tuple(v.tolist()): i for i, v in enumerate(unique_voxels)}
    uf = UnionFind(len(unique_voxels))
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0)
    ]
    for i, voxel in enumerate(unique_voxels):
        base = tuple(voxel.tolist())
        for dx, dy, dz in offsets:
            j = lookup.get((base[0] + dx, base[1] + dy, base[2] + dz))
            if j is not None:
                uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(len(unique_voxels))], dtype=np.int32)
    return roots[inv], unique_voxels, roots


def write_ascii_ply(path: Path, rows: list[tuple[float, float, float, int, int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for x, y, z, r, g, b, obj, sem in rows:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {obj} {sem}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--min-points", type=int, default=40)
    parser.add_argument("--min-points-by-class", default="floor:200,wall:200,grass:100,car:30,railing:20")
    parser.add_argument("--object-id-base", type=int, default=1_000_000)
    args = parser.parse_args()

    min_by_class = {}
    for item in args.min_points_by_class.split(","):
        if not item.strip():
            continue
        name, value = item.split(":", 1)
        min_by_class[name.strip()] = int(value)

    data, props = read_ply(args.input_ply)
    idx = {name: i for i, name in enumerate(props)}
    for name in ("x", "y", "z", "red", "green", "blue", "priority"):
        if name not in idx:
            raise ValueError(f"Input PLY missing {name}: {args.input_ply}")

    points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    colors = data[:, [idx["red"], idx["green"], idx["blue"]]].astype(np.uint8)
    priority = data[:, idx["priority"]].astype(np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[tuple[float, float, float, int, int, int, int, int]] = []
    objects = []
    class_counts = Counter()
    object_counts = Counter()
    next_seq = 1

    for priority_id in sorted(PRIORITY_CLASSES):
        label, semantic, status = PRIORITY_CLASSES[priority_id]
        class_mask = priority == priority_id
        class_counts[label] = int(class_mask.sum())
        if not np.any(class_mask):
            continue
        class_points = points[class_mask]
        class_colors = colors[class_mask]
        class_indices = np.where(class_mask)[0]
        roots_for_point, voxel_roots, roots = component_roots(class_points, args.voxel_size)
        root_counts = Counter(int(x) for x in roots_for_point.tolist())
        min_points = min_by_class.get(label, args.min_points)
        kept_roots = [root for root, count in root_counts.items() if count >= min_points]
        kept_roots.sort(key=lambda root: (-root_counts[root], root))
        for root in kept_roots:
            local_mask = roots_for_point == root
            pts = class_points[local_mask]
            cols = class_colors[local_mask]
            object_id = args.object_id_base + priority_id * 100_000 + next_seq
            next_seq += 1
            object_counts[label] += 1
            for p, c in zip(pts, cols):
                out_rows.append((
                    float(p[0]), float(p[1]), float(p[2]),
                    int(c[0]), int(c[1]), int(c[2]),
                    int(object_id), int(semantic),
                ))
            obj = {
                "object_id": int(object_id),
                "priority_id": int(priority_id),
                "semantic_label": label,
                "description": f"priority-layer {label} component",
                "status": status,
                "point_count": int(local_mask.sum()),
                "voxel_count": int((voxel_roots == root).sum()),
                "centroid": pts.mean(axis=0).astype(float).tolist(),
                "bbox_min": pts.min(axis=0).astype(float).tolist(),
                "bbox_max": pts.max(axis=0).astype(float).tolist(),
                "bbox_3d": {
                    "min": pts.min(axis=0).astype(float).tolist(),
                    "max": pts.max(axis=0).astype(float).tolist(),
                },
                "mean_color": cols.astype(np.float32).mean(axis=0).astype(float).tolist(),
                **pca_stats(pts),
                "source": str(args.input_ply),
            }
            objects.append(obj)

    out_ply = args.output_dir / "priority_objects_ascii.ply"
    out_jsonl = args.output_dir / "priority_objects.jsonl"
    write_ascii_ply(out_ply, out_rows)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    report = {
        "input_ply": str(args.input_ply),
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "voxel_size": args.voxel_size,
        "class_point_counts": dict(class_counts),
        "object_counts": dict(object_counts),
        "object_count": int(len(objects)),
        "output_points": int(len(out_rows)),
        "dropped_small_points": int(sum(class_counts.values()) - len(out_rows)),
    }
    (args.output_dir / "priority_object_cluster_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
