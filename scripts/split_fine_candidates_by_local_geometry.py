#!/usr/bin/env python3
"""Split fine-object candidate objects into local 3D components.

The full-scene object layer is good for review, but fine objects such as
railings and cars can still be too fragmented or locally mixed with surface
points.  This script takes a candidate JSONL plus a full-density object PLY,
extracts only candidate points, and reclusters each original object with a
smaller voxel size.

The output is intentionally a candidate dataset, not a final semantic result.
It preserves the parent candidate label/prompt metadata and adds local geometry
features so image-evidence and DINO stages can work on tighter 3D seeds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PLY_TYPE_MAP = {
    "float": "<f4", "float32": "<f4", "double": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}

SEMANTIC_IDS = {
    "unknown": 0,
    "wall": 2,
    "floor": 3,
    "grass": 5,
    "fine_candidate": 7,
    "car": 8,
    "railing": 9,
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_ply_header(path: Path) -> tuple[str, int, list[str], list[str], int]:
    with path.open("rb") as f:
        fmt = "ascii"
        vertex_count = 0
        props: list[str] = []
        prop_types: list[str] = []
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
    return fmt, vertex_count, props, prop_types, header_lines


def read_candidate_points(
    ply_path: Path,
    candidate_ids: set[int],
) -> dict[int, dict[str, list[list[float]]]]:
    fmt, vertex_count, props, prop_types, header_lines = parse_ply_header(ply_path)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    if object_col is None:
        raise ValueError(f"PLY missing object/object_id field: {ply_path}")
    for name in ("x", "y", "z"):
        if name not in idx:
            raise ValueError(f"PLY missing {name}: {ply_path}")

    buckets: dict[int, dict[str, list[list[float]]]] = {
        oid: {"xyz": [], "rgb": []} for oid in candidate_ids
    }
    if fmt == "ascii":
        with ply_path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(header_lines):
                next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) <= object_col:
                    continue
                try:
                    oid = int(round(float(parts[object_col])))
                except ValueError:
                    continue
                bucket = buckets.get(oid)
                if bucket is None:
                    continue
                bucket["xyz"].append([float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])])
                if {"red", "green", "blue"}.issubset(idx):
                    bucket["rgb"].append([float(parts[idx["red"]]), float(parts[idx["green"]]), float(parts[idx["blue"]])])
                else:
                    bucket["rgb"].append([180.0, 180.0, 180.0])
        return {oid: b for oid, b in buckets.items() if b["xyz"]}

    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {fmt}")
    dtype = np.dtype([(name, PLY_TYPE_MAP[ptype]) for ptype, name in zip(prop_types, props)])
    with ply_path.open("rb") as f:
        while f.readline().strip() != b"end_header":
            pass
        arr = np.frombuffer(f.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
    object_values = arr[props[object_col]].astype(np.uint32)
    for oid in candidate_ids:
        mask = object_values == oid
        if not np.any(mask):
            continue
        buckets[oid]["xyz"] = np.column_stack([arr["x"][mask], arr["y"][mask], arr["z"][mask]]).astype(float).tolist()
        if {"red", "green", "blue"}.issubset(set(props)):
            buckets[oid]["rgb"] = np.column_stack([arr["red"][mask], arr["green"][mask], arr["blue"][mask]]).astype(float).tolist()
        else:
            buckets[oid]["rgb"] = [[180.0, 180.0, 180.0]] * int(mask.sum())
    return {oid: b for oid, b in buckets.items() if b["xyz"]}


def component_roots(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, int]:
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
    return roots[inv], len(unique_voxels)


def pca_stats(points: np.ndarray) -> dict[str, Any]:
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
    eps = 1e-9
    l1, l2, l3 = [float(max(v, 0.0)) for v in eigvals]
    extents = points.max(axis=0) - points.min(axis=0)
    thickness = float(np.sqrt(l3))
    spread = float(np.sqrt(l1))
    linearity = float((l1 - l2) / max(l1, eps))
    planarity = float((l2 - l3) / max(l1, eps))
    scattering = float(l3 / max(l1, eps))
    return {
        "pca_eigenvalues": [l1, l2, l3],
        "pca_major_axis": eigvecs[:, 0].astype(float).tolist(),
        "pca_normal": eigvecs[:, -1].astype(float).tolist(),
        "linearity": linearity,
        "planarity_pca": planarity,
        "scattering": scattering,
        "thickness_rms": thickness,
        "spread_rms": spread,
        "extent": extents.astype(float).tolist(),
        "extent_max": float(extents.max()),
        "extent_mid": float(np.sort(extents)[1]),
        "extent_min": float(extents.min()),
    }


def geometry_class(stats: dict[str, Any], point_count: int, args: argparse.Namespace) -> str:
    if (
        stats["linearity"] >= args.linear_min_linearity
        and stats["extent_max"] >= args.linear_min_extent
        and stats["extent_min"] <= args.linear_max_thickness
    ):
        return "linear_candidate"
    if (
        stats["planarity_pca"] >= args.plane_min_planarity
        and stats["extent_max"] >= args.plane_min_extent
        and stats["thickness_rms"] <= args.plane_max_thickness
        and point_count >= args.plane_min_points
    ):
        return "planar_surface_fragment"
    if stats["extent_max"] <= args.compact_max_extent:
        return "compact_candidate"
    return "irregular_candidate"


def color_for_subobject(object_id: int, geometry: str) -> tuple[int, int, int]:
    if geometry == "linear_candidate":
        return 245, 190, 35
    if geometry == "planar_surface_fragment":
        return 170, 170, 170
    if geometry == "compact_candidate":
        return 90, 190, 245
    x = (object_id * 1103515245 + 12345) & 0xFFFFFFFF
    return 80 + (x & 127), 80 + ((x >> 8) & 127), 80 + ((x >> 16) & 127)


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


def subobject_id(parent_id: int, split_index: int) -> int:
    value = parent_id * 1000 + split_index
    if value > 4_294_000_000:
        value = (parent_id % 3_000_000) * 1000 + split_index
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--object-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--min-points", type=int, default=24)
    parser.add_argument("--max-components-per-object", type=int, default=30)
    parser.add_argument("--linear-min-linearity", type=float, default=0.62)
    parser.add_argument("--linear-min-extent", type=float, default=0.8)
    parser.add_argument("--linear-max-thickness", type=float, default=0.45)
    parser.add_argument("--plane-min-planarity", type=float, default=0.45)
    parser.add_argument("--plane-min-extent", type=float, default=1.2)
    parser.add_argument("--plane-max-thickness", type=float, default=0.20)
    parser.add_argument("--plane-min-points", type=int, default=80)
    parser.add_argument("--compact-max-extent", type=float, default=1.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(args.objects_jsonl)
    candidate_by_id = {int(row["object_id"]): row for row in candidates}
    point_buckets = read_candidate_points(args.object_ply, set(candidate_by_id))

    output_objects: list[dict[str, Any]] = []
    ply_rows: list[tuple[float, float, float, int, int, int, int, int]] = []
    missing_objects = sorted(set(candidate_by_id) - set(point_buckets))
    split_counts = Counter()
    geometry_counts = Counter()
    parent_component_counts: dict[int, int] = {}

    for parent_id in sorted(point_buckets):
        parent = candidate_by_id[parent_id]
        pts = np.asarray(point_buckets[parent_id]["xyz"], dtype=np.float32)
        rgb = np.asarray(point_buckets[parent_id]["rgb"], dtype=np.uint8)
        roots_for_point, voxel_count = component_roots(pts, args.voxel_size)
        root_counts = Counter(int(x) for x in roots_for_point.tolist())
        kept_roots = [root for root, count in root_counts.items() if count >= args.min_points]
        kept_roots.sort(key=lambda root: (-root_counts[root], root))
        if args.max_components_per_object > 0:
            kept_roots = kept_roots[: args.max_components_per_object]
        parent_component_counts[parent_id] = len(kept_roots)
        split_counts[len(kept_roots)] += 1

        for split_index, root in enumerate(kept_roots, 1):
            mask = roots_for_point == root
            sub_pts = pts[mask]
            sub_rgb = rgb[mask]
            stats = pca_stats(sub_pts)
            geom = geometry_class(stats, int(mask.sum()), args)
            geometry_counts[geom] += 1
            oid = subobject_id(parent_id, split_index)
            candidate_label = str(parent.get("candidate_label") or parent.get("semantic_label_original") or parent.get("semantic_label") or "fine_candidate")
            semantic_label = str(parent.get("semantic_label") or "fine_candidate")
            dino_group = str(parent.get("dino_prompt_group") or (candidate_label if candidate_label in {"car", "railing"} else "unknown"))
            out = dict(parent)
            out.update({
                "object_id": oid,
                "parent_object_id": parent_id,
                "split_index": split_index,
                "split_source": str(args.object_ply),
                "split_voxel_size": args.voxel_size,
                "semantic_label": semantic_label,
                "candidate_label": candidate_label,
                "dino_prompt_group": dino_group,
                "downstream_stage": "dino_fine_object_review",
                "review_priority": "high",
                "geometry_class": geom,
                "point_count": int(mask.sum()),
                "parent_point_count": int(len(pts)),
                "parent_component_count": len(kept_roots),
                "voxel_count": int(voxel_count),
                "centroid": sub_pts.mean(axis=0).astype(float).tolist(),
                "bbox_min": sub_pts.min(axis=0).astype(float).tolist(),
                "bbox_max": sub_pts.max(axis=0).astype(float).tolist(),
                "bbox_3d": {
                    "min": sub_pts.min(axis=0).astype(float).tolist(),
                    "max": sub_pts.max(axis=0).astype(float).tolist(),
                },
                "mean_color": sub_rgb.astype(np.float32).mean(axis=0).astype(float).tolist(),
                **stats,
            })
            output_objects.append(out)
            r, g, b = color_for_subobject(oid, geom)
            sem = SEMANTIC_IDS.get(candidate_label, SEMANTIC_IDS.get(semantic_label, 7))
            for p in sub_pts:
                ply_rows.append((float(p[0]), float(p[1]), float(p[2]), r, g, b, oid, sem))

    out_jsonl = args.output_dir / "fine_candidate_splits.jsonl"
    out_ply = args.output_dir / "fine_candidate_splits.ply"
    write_jsonl(out_jsonl, output_objects)
    write_ascii_ply(out_ply, ply_rows)

    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "object_ply": str(args.object_ply),
        "output_dir": str(args.output_dir),
        "output_jsonl": str(out_jsonl),
        "output_ply": str(out_ply),
        "voxel_size": args.voxel_size,
        "min_points": args.min_points,
        "candidate_objects": len(candidates),
        "objects_with_points": len(point_buckets),
        "missing_objects": missing_objects,
        "split_object_count": len(output_objects),
        "split_point_count": len(ply_rows),
        "parent_component_count_histogram": {str(k): int(v) for k, v in sorted(split_counts.items())},
        "geometry_class_counts": dict(geometry_counts),
        "candidate_label_counts": dict(Counter(str(row.get("candidate_label") or row.get("semantic_label")) for row in output_objects)),
        "top_split_parents": [
            {"parent_object_id": int(pid), "component_count": int(count)}
            for pid, count in sorted(parent_component_counts.items(), key=lambda item: (-item[1], item[0]))[:30]
        ],
    }
    (args.output_dir / "fine_candidate_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
