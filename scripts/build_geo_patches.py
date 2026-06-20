#!/usr/bin/env python3
"""Build geometry-first GeoPatch records from a point-cloud PLY.

This is the first stage of the geometry-first semantic route.  It does not
trust semantic labels as object boundaries.  If a source PLY already contains an
``object`` property, that object id is only used as a seed; mixed seeds are
split into local PCA/connectivity patches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PLY_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}

GEOMETRY_COLORS = {
    "horizontal_surface": (196, 168, 112),
    "vertical_surface": (120, 150, 180),
    "upper_surface": (165, 145, 210),
    "linear_thin": (240, 210, 60),
    "bulky_object": (235, 90, 80),
    "vegetation_like": (80, 160, 80),
    "mixed": (245, 150, 40),
    "unknown": (150, 150, 150),
}

GEOMETRY_SEMANTIC = {
    "horizontal_surface": 3,
    "vertical_surface": 2,
    "upper_surface": 20,
    "linear_thin": 9,
    "bulky_object": 17,
    "vegetation_like": 5,
    "mixed": 0,
    "unknown": 0,
}


def parse_ply_header(path: Path) -> tuple[str, list[tuple[str, str]], int, int]:
    fmt = "ascii"
    props: list[tuple[str, str]] = []
    vertex_count = 0
    header_bytes = 0
    in_vertex = False
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            header_bytes += len(raw)
            line = raw.decode("utf-8", errors="replace").strip()
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "format":
                fmt = parts[1]
            elif len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append((parts[-2], parts[-1]))
            elif line == "end_header":
                break
    if vertex_count <= 0:
        raise ValueError(f"No vertex count found: {path}")
    return fmt, props, vertex_count, header_bytes


def read_ply_numeric(
    path: Path,
    point_stride: int = 1,
    max_points: int = 0,
) -> tuple[list[str], np.ndarray]:
    fmt, typed_props, vertex_count, header_bytes = parse_ply_header(path)
    names = [name for _ptype, name in typed_props]
    stride = max(int(point_stride), 1)
    if fmt == "ascii":
        kept = (vertex_count + stride - 1) // stride
        if max_points:
            kept = min(kept, int(max_points))
        data = np.empty((kept, len(names)), dtype=np.float64)
        row = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip() == "end_header":
                    break
            for i, line in enumerate(f):
                if i % stride:
                    continue
                parts = line.strip().split()
                if len(parts) < len(names):
                    continue
                data[row, :] = [float(x) for x in parts[: len(names)]]
                row += 1
                if row >= kept:
                    break
        return names, data[:row]
    if fmt == "binary_little_endian":
        dtype = np.dtype([(name, PLY_DTYPE.get(ptype, "<f4")) for ptype, name in typed_props])
        with path.open("rb") as f:
            f.seek(header_bytes)
            table = np.fromfile(f, dtype=dtype, count=vertex_count)
        table = table[::stride]
        if max_points:
            table = table[: int(max_points)]
        data = np.column_stack([table[name].astype(np.float64) for name in names])
        return names, data
    raise ValueError(f"Unsupported PLY format {fmt}: {path}")


def pca_stats(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 3:
        return {
            "normal": [0.0, 0.0, 1.0],
            "eigenvalues": [0.0, 0.0, 0.0],
            "linearity": 0.0,
            "planarity": 0.0,
            "scattering": 0.0,
            "thickness": 0.0,
        }
    centered = points.astype(np.float64) - points.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(len(points) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    denom = max(float(vals[0]), 1e-12)
    normal = vecs[:, -1]
    if normal[2] < 0:
        normal = -normal
    return {
        "normal": [float(x) for x in normal.tolist()],
        "eigenvalues": [float(x) for x in vals.tolist()],
        "linearity": float((vals[0] - vals[1]) / denom),
        "planarity": float((vals[1] - vals[2]) / denom),
        "scattering": float(vals[2] / denom),
        "thickness": float(math.sqrt(max(float(vals[-1]), 0.0))),
    }


def bbox(points: np.ndarray) -> dict[str, Any]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {"min": [float(x) for x in lo.tolist()], "max": [float(x) for x in hi.tolist()]}


def geometry_type_from_stats(stats: dict[str, Any], extent: np.ndarray, args: argparse.Namespace) -> str:
    nz = abs(float(stats["normal"][2]))
    linearity = float(stats["linearity"])
    planarity = float(stats["planarity"])
    scattering = float(stats["scattering"])
    thickness = float(stats["thickness"])
    z_extent = float(extent[2])
    xy_extent = float(max(extent[0], extent[1]))

    if linearity >= args.linear_thin_min_linearity and max(extent) >= args.linear_thin_min_extent:
        return "linear_thin"
    if nz >= args.horizontal_normal_z and planarity >= args.surface_min_planarity and thickness <= args.surface_max_thickness:
        if z_extent >= args.upper_surface_min_z_extent and xy_extent <= args.upper_surface_max_xy_extent:
            return "upper_surface"
        return "horizontal_surface"
    if nz <= args.vertical_normal_z and planarity >= args.surface_min_planarity and thickness <= args.wall_max_thickness:
        return "vertical_surface"
    if nz >= args.horizontal_normal_z and scattering >= args.vegetation_min_scattering and z_extent <= args.vegetation_max_z_extent:
        return "vegetation_like"
    if max(extent) >= args.bulky_min_extent and z_extent >= args.bulky_min_z_extent:
        return "bulky_object"
    if planarity < args.mixed_planarity_max and max(extent) >= args.mixed_min_extent:
        return "mixed"
    return "unknown"


def voxelize(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int, int], np.ndarray]]:
    coords = np.floor(points / float(voxel_size)).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, coord in enumerate(coords):
        buckets[(int(coord[0]), int(coord[1]), int(coord[2]))].append(idx)
    out = {key: np.asarray(value, dtype=np.int64) for key, value in buckets.items()}
    return coords, np.asarray(list(out.keys()), dtype=np.int32), out


def voxel_orientation_groups(points: np.ndarray, voxel_size: float, args: argparse.Namespace) -> dict[str, list[tuple[int, int, int]]]:
    _coords, _keys, buckets = voxelize(points, voxel_size)
    groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for key, indices in buckets.items():
        if len(indices) < args.local_pca_min_points:
            groups["unknown"].append(key)
            continue
        pts = points[indices]
        stats = pca_stats(pts)
        ext = pts.max(axis=0) - pts.min(axis=0)
        groups[geometry_type_from_stats(stats, ext, args)].append(key)
    return groups


def connected_voxel_components(
    voxel_keys: list[tuple[int, int, int]],
    buckets: dict[tuple[int, int, int], np.ndarray],
    min_points: int,
) -> list[np.ndarray]:
    key_set = set(voxel_keys)
    visited: set[tuple[int, int, int]] = set()
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    components: list[np.ndarray] = []
    for start in voxel_keys:
        if start in visited:
            continue
        queue: deque[tuple[int, int, int]] = deque([start])
        visited.add(start)
        comp_keys = []
        while queue:
            key = queue.popleft()
            comp_keys.append(key)
            for dx, dy, dz in offsets:
                nxt = (key[0] + dx, key[1] + dy, key[2] + dz)
                if nxt in key_set and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        indices = np.concatenate([buckets[key] for key in comp_keys])
        if len(indices) >= int(min_points):
            components.append(indices)
    components.sort(key=len, reverse=True)
    return components


def split_axis_aligned_planes(points: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    """Fallback split for obvious floor/wall style slabs.

    Local voxel PCA can be unstable when a sparse wall and floor touch along an
    edge.  This fallback extracts high-support x/y/z slabs, leaving any
    remaining points as residual.  It is deliberately conservative and is used
    only when the local-PCA split does not produce multiple components.
    """

    remaining = np.ones(len(points), dtype=bool)
    components: list[np.ndarray] = []
    for _ in range(args.axis_plane_max_planes):
        remaining_idx = np.where(remaining)[0]
        if len(remaining_idx) < args.min_patch_points:
            break
        best: np.ndarray | None = None
        best_score = 0.0
        for axis in (0, 1, 2):
            values = points[remaining_idx, axis]
            bins = np.floor(values / float(args.axis_plane_bin_size)).astype(np.int32)
            for bucket in np.unique(bins):
                seed_idx = remaining_idx[bins == bucket]
                if len(seed_idx) < args.min_patch_points:
                    continue
                center = float(np.median(points[seed_idx, axis]))
                comp = remaining_idx[np.abs(points[remaining_idx, axis] - center) <= args.axis_plane_distance]
                if len(comp) < args.min_patch_points:
                    continue
                stats = pca_stats(points[comp])
                extent = points[comp].max(axis=0) - points[comp].min(axis=0)
                gtype = geometry_type_from_stats(stats, extent, args)
                if gtype not in {"horizontal_surface", "vertical_surface", "upper_surface"}:
                    continue
                score = float(len(comp)) * max(float(stats["planarity"]), 0.05)
                if score > best_score:
                    best_score = score
                    best = comp
        if best is None:
            break
        components.append(best)
        remaining[best] = False
    residual = np.where(remaining)[0]
    if len(residual) > 0:
        components.append(residual)
    components.sort(key=len, reverse=True)
    return components


def split_seed_points(points: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    if len(points) < args.min_patch_points:
        return [np.arange(len(points), dtype=np.int64)]
    stats = pca_stats(points)
    ext = points.max(axis=0) - points.min(axis=0)
    gtype = geometry_type_from_stats(stats, ext, args)
    clean = gtype not in {"mixed", "unknown"} and float(stats["planarity"]) >= args.clean_planarity_min
    if clean and len(points) <= args.max_clean_seed_points:
        return [np.arange(len(points), dtype=np.int64)]

    _coords, _keys, buckets = voxelize(points, args.patch_voxel_size)
    orientation_groups = voxel_orientation_groups(points, args.patch_voxel_size, args)
    components: list[np.ndarray] = []
    for group_keys in orientation_groups.values():
        components.extend(connected_voxel_components(group_keys, buckets, args.min_patch_points))

    covered = np.zeros(len(points), dtype=bool)
    for comp in components:
        covered[comp] = True
    residual = np.where(~covered)[0]
    if len(residual) > 0:
        components.append(residual)
    clean_types = set()
    for comp in components:
        pts = points[comp]
        if len(pts) < 3:
            continue
        clean_types.add(geometry_type_from_stats(pca_stats(pts), pts.max(axis=0) - pts.min(axis=0), args))
    if len(components) <= 1 or not (clean_types - {"mixed", "unknown"}):
        fallback = split_axis_aligned_planes(points, args)
        fallback_types = set()
        for comp in fallback:
            pts = points[comp]
            if len(pts) >= 3:
                fallback_types.add(geometry_type_from_stats(pca_stats(pts), pts.max(axis=0) - pts.min(axis=0), args))
        if len(fallback_types - {"mixed", "unknown"}) > len(clean_types - {"mixed", "unknown"}) or len(fallback) > len(components):
            components = fallback
    if not components:
        components = [np.arange(len(points), dtype=np.int64)]
    components.sort(key=len, reverse=True)
    return components


def counter_from_values(values: np.ndarray | None) -> dict[str, int]:
    if values is None or len(values) == 0:
        return {}
    counts = Counter(str(int(round(float(x)))) for x in values.tolist())
    return dict(counts)


def region_votes(points: np.ndarray, structural_field: dict[str, Any] | None, sample_points: int) -> tuple[dict[str, int], float]:
    if structural_field is None or len(points) == 0:
        return {}, 0.0
    from classify_surface_attachment import vote_structural_regions
    from build_structural_region_field import REGION_NAMES

    sample = points
    if sample_points and len(points) > sample_points:
        indices = np.linspace(0, len(points) - 1, sample_points).astype(np.int64)
        sample = points[indices]
    counts, confidence = vote_structural_regions(sample, structural_field, neighbor_radius=1)
    return {REGION_NAMES.get(int(k), str(k)): int(v) for k, v in counts.items()}, float(confidence)


def load_structural_field_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    from classify_surface_attachment import load_structural_field

    return load_structural_field(path)


def build_geo_patches(args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    names, data = read_ply_numeric(args.input_ply, args.point_stride, args.max_points)
    idx = {name: i for i, name in enumerate(names)}
    for required in ("x", "y", "z"):
        if required not in idx:
            raise ValueError(f"Input PLY missing {required}: {args.input_ply}")
    points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    if all(name in idx for name in ("red", "green", "blue")):
        colors = np.clip(data[:, [idx["red"], idx["green"], idx["blue"]]], 0, 255).astype(np.uint8)

    seed_prop = args.seed_property
    if seed_prop == "auto":
        seed_prop = "object" if "object" in idx else "none"
    if seed_prop != "none" and seed_prop not in idx:
        raise ValueError(f"Seed property {seed_prop!r} not present in {args.input_ply}")
    if seed_prop == "none":
        seed_values = np.zeros(len(points), dtype=np.int64)
    else:
        seed_values = np.rint(data[:, idx[seed_prop]]).astype(np.int64)

    source_props = {
        name: data[:, idx[name]]
        for name in ("object", "semantic", "priority", "frame", "camera", "target")
        if name in idx
    }
    structural_field = load_structural_field_optional(args.structural_field)
    patches: list[dict[str, Any]] = []
    patch_ids = np.zeros(len(points), dtype=np.int32)
    patch_index = 1
    seed_counts = Counter(seed_values.tolist())
    for seed, _count in seed_counts.most_common():
        seed_indices = np.where(seed_values == seed)[0]
        local_points = points[seed_indices]
        components = split_seed_points(local_points, args)
        seed_split = len(components) > 1
        for local_component_index, comp_local in enumerate(components):
            global_indices = seed_indices[comp_local]
            pts = points[global_indices]
            if len(pts) == 0:
                continue
            stats = pca_stats(pts)
            ext = pts.max(axis=0) - pts.min(axis=0)
            gtype = geometry_type_from_stats(stats, ext, args)
            if gtype == "mixed" and len(pts) < args.mixed_min_points:
                gtype = "unknown"
            structural_votes, structural_conf = region_votes(pts, structural_field, args.structural_sample_points)
            patch_id = f"patch_{patch_index:06d}"
            patch_ids[global_indices] = patch_index
            frame_values = source_props.get("frame")
            frame_span = None
            if frame_values is not None:
                frames = frame_values[global_indices]
                frame_span = {
                    "min": int(np.min(frames)),
                    "max": int(np.max(frames)),
                    "mean": float(np.mean(frames)),
                    "count": int(len(set(int(x) for x in frames.tolist()))),
                }
            color_values = colors[global_indices].astype(np.float64)
            patch = {
                "patch_id": patch_id,
                "patch_index": patch_index,
                "source_seed_property": seed_prop,
                "source_seed_value": int(seed),
                "connectivity_component_id": f"{int(seed)}:{local_component_index}",
                "point_count": int(len(global_indices)),
                "bbox_3d": bbox(pts),
                "centroid": [float(x) for x in pts.mean(axis=0).tolist()],
                "extent": [float(x) for x in ext.tolist()],
                "normal": stats["normal"],
                "pca": stats,
                "thickness": float(stats["thickness"]),
                "linearity": float(stats["linearity"]),
                "planarity": float(stats["planarity"]),
                "roughness": float(stats["scattering"]),
                "geometry_type": gtype,
                "split_status": "clean" if gtype not in {"mixed", "unknown"} else ("needs_split" if len(pts) >= args.min_patch_points else "residual"),
                "seed_was_split": bool(seed_split),
                "structural_region_votes": structural_votes,
                "structural_confidence_mean": structural_conf,
                "source_frame_span": frame_span,
                "color_stats": {
                    "mean_rgb": [float(x) for x in color_values.mean(axis=0).tolist()],
                    "std_rgb": [float(x) for x in color_values.std(axis=0).tolist()],
                },
                "source_votes": {
                    name: counter_from_values(values[global_indices])
                    for name, values in source_props.items()
                },
            }
            patches.append(patch)
            patch_index += 1

    report = {
        "schema": "geo-patches/v1",
        "input_ply": str(args.input_ply),
        "input_points": int(len(points)),
        "patch_count": len(patches),
        "seed_property": seed_prop,
        "seed_count": len(seed_counts),
        "parameters": {
            "patch_voxel_size": float(args.patch_voxel_size),
            "min_patch_points": int(args.min_patch_points),
            "point_stride": int(args.point_stride),
        },
        "geometry_type_counts": dict(Counter(p["geometry_type"] for p in patches)),
        "split_status_counts": dict(Counter(p["split_status"] for p in patches)),
    }
    return patches, patch_ids, points, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_patch_ply(path: Path, points: np.ndarray, patch_ids: np.ndarray, patches: list[dict[str, Any]]) -> None:
    patch_by_id = {int(p["patch_index"]): p for p in patches}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int object\nproperty uchar semantic\n")
        f.write("property int patch\n")
        f.write("end_header\n")
        for point, patch_idx in zip(points, patch_ids):
            patch = patch_by_id.get(int(patch_idx), {})
            gtype = str(patch.get("geometry_type") or "unknown")
            color = GEOMETRY_COLORS.get(gtype, GEOMETRY_COLORS["unknown"])
            semantic = GEOMETRY_SEMANTIC.get(gtype, 0)
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]} {color[1]} {color[2]} {int(patch_idx)} {semantic} {int(patch_idx)}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--structural-field", type=Path, default=None)
    parser.add_argument("--seed-property", default="auto", help="auto, none, or a numeric PLY property such as object")
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--patch-voxel-size", type=float, default=0.18)
    parser.add_argument("--min-patch-points", type=int, default=120)
    parser.add_argument("--mixed-min-points", type=int, default=300)
    parser.add_argument("--max-clean-seed-points", type=int, default=20_000)
    parser.add_argument("--local-pca-min-points", type=int, default=8)
    parser.add_argument("--clean-planarity-min", type=float, default=0.70)
    parser.add_argument("--surface-min-planarity", type=float, default=0.58)
    parser.add_argument("--surface-max-thickness", type=float, default=0.30)
    parser.add_argument("--wall-max-thickness", type=float, default=0.45)
    parser.add_argument("--horizontal-normal-z", type=float, default=0.86)
    parser.add_argument("--vertical-normal-z", type=float, default=0.42)
    parser.add_argument("--linear-thin-min-linearity", type=float, default=0.72)
    parser.add_argument("--linear-thin-min-extent", type=float, default=0.80)
    parser.add_argument("--vegetation-min-scattering", type=float, default=0.05)
    parser.add_argument("--vegetation-max-z-extent", type=float, default=2.50)
    parser.add_argument("--bulky-min-extent", type=float, default=0.80)
    parser.add_argument("--bulky-min-z-extent", type=float, default=0.45)
    parser.add_argument("--mixed-planarity-max", type=float, default=0.50)
    parser.add_argument("--mixed-min-extent", type=float, default=1.20)
    parser.add_argument("--upper-surface-min-z-extent", type=float, default=0.15)
    parser.add_argument("--upper-surface-max-xy-extent", type=float, default=3.00)
    parser.add_argument("--axis-plane-bin-size", type=float, default=0.10)
    parser.add_argument("--axis-plane-distance", type=float, default=0.05)
    parser.add_argument("--axis-plane-max-planes", type=int, default=12)
    parser.add_argument("--structural-sample-points", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patches, patch_ids, points, report = build_geo_patches(args)
    write_jsonl(args.output_dir / "geo_patches.jsonl", patches)
    write_patch_ply(args.output_dir / "geo_patch_points.ply", points, patch_ids, patches)
    (args.output_dir / "geo_patch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
