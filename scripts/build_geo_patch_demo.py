#!/usr/bin/env python3
"""Build geometry-first point-cloud patches for visual QA.

This demo intentionally ignores existing semantic/object ids.  It builds voxel
patches from geometry and color continuity only, so we can inspect whether the
structural boundaries are clean before adding MASK/VLM evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


def parse_header(path: Path) -> tuple[list[str], int, int]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            if line.strip() == "end_header":
                break
    return props, vertex_count, header_lines


def read_voxels(path: Path, voxel_size: float, max_points: int | None = None) -> dict[tuple[int, int, int], dict[str, Any]]:
    props, vertex_count, header_lines = parse_header(path)
    idx = {name: i for i, name in enumerate(props)}
    for required in ("x", "y", "z", "red", "green", "blue"):
        if required not in idx:
            raise ValueError(f"PLY missing {required}: {path}")
    voxels: dict[tuple[int, int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for i, line in enumerate(f):
            if i >= vertex_count:
                break
            if max_points is not None and i >= max_points:
                break
            if not line.strip():
                continue
            row = line.split()
            xyz = np.array([float(row[idx["x"]]), float(row[idx["y"]]), float(row[idx["z"]])], dtype=np.float64)
            rgb = np.array([float(row[idx["red"]]), float(row[idx["green"]]), float(row[idx["blue"]])], dtype=np.float64)
            key = tuple(math.floor(float(v) / voxel_size) for v in xyz)
            item = voxels.get(key)
            if item is None:
                voxels[key] = {"count": 1, "xyz_sum": xyz, "rgb_sum": rgb}
            else:
                item["count"] += 1
                item["xyz_sum"] += xyz
                item["rgb_sum"] += rgb
    for item in voxels.values():
        count = max(float(item["count"]), 1.0)
        item["xyz"] = item["xyz_sum"] / count
        item["rgb"] = item["rgb_sum"] / count
    return voxels


def neighbor_offsets(radius: int) -> list[tuple[int, int, int]]:
    offsets: list[tuple[int, int, int]] = []
    r2 = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                d2 = dx * dx + dy * dy + dz * dz
                if d2 <= r2:
                    offsets.append((dx, dy, dz))
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1] + item[2] * item[2])
    return offsets


def compute_local_features(
    voxels: dict[tuple[int, int, int], dict[str, Any]],
    feature_radius_voxels: int,
) -> None:
    offsets = [(0, 0, 0)] + neighbor_offsets(feature_radius_voxels)
    for key, item in voxels.items():
        pts = []
        rgbs = []
        for dx, dy, dz in offsets:
            nbr = voxels.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if nbr is None:
                continue
            pts.append(nbr["xyz"])
            rgbs.append(nbr["rgb"])
        arr = np.vstack(pts).astype(np.float64)
        item["local_neighbor_count"] = int(len(arr))
        item["local_color_std"] = float(np.linalg.norm(np.std(np.vstack(rgbs), axis=0))) if rgbs else 0.0
        if len(arr) < 4:
            item["normal"] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            item["planarity"] = 0.0
            item["linearity"] = 0.0
            item["roughness"] = 1.0
            item["height_range"] = 0.0
            continue
        centered = arr - arr.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(len(arr) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 1e-12)
        eigvecs = eigvecs[:, order]
        normal = eigvecs[:, -1]
        if normal[2] < 0:
            normal = -normal
        l1, l2, l3 = [float(x) for x in eigvals]
        item["normal"] = normal.astype(np.float64)
        item["linearity"] = float((l1 - l2) / l1)
        item["planarity"] = float((l2 - l3) / l1)
        item["roughness"] = float(l3 / l1)
        item["height_range"] = float(arr[:, 2].max() - arr[:, 2].min())


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    cos = abs(float(np.dot(a, b) / (an * bn)))
    return math.degrees(math.acos(max(min(cos, 1.0), -1.0)))


def geometry_bucket(item: dict[str, Any]) -> str:
    normal = item["normal"]
    nz = abs(float(normal[2]))
    if item["linearity"] >= 0.72 and item["planarity"] < 0.22:
        return "thin_linear"
    if item["planarity"] >= 0.35 and nz >= 0.82:
        return "horizontal"
    if item["planarity"] >= 0.30 and nz <= 0.35:
        return "vertical"
    if item["roughness"] >= 0.16 or item["local_color_std"] >= 75:
        return "rough_mixed"
    return "unknown"


def edge_allowed(
    a: dict[str, Any],
    b: dict[str, Any],
    max_normal_angle: float,
    max_color_distance: float,
    max_height_delta: float,
    strict_bucket: bool,
) -> bool:
    dz = abs(float(a["xyz"][2] - b["xyz"][2]))
    if dz > max_height_delta:
        return False
    rgb_dist = float(np.linalg.norm(a["rgb"] - b["rgb"]))
    if rgb_dist > max_color_distance:
        return False
    angle = normal_angle_deg(a["normal"], b["normal"])
    if angle > max_normal_angle:
        return False
    if strict_bucket and a["bucket"] != b["bucket"]:
        if {a["bucket"], b["bucket"]} <= {"horizontal", "unknown"}:
            return True
        if {a["bucket"], b["bucket"]} <= {"vertical", "unknown"}:
            return True
        return False
    return True


def build_patches(
    voxels: dict[tuple[int, int, int], dict[str, Any]],
    connect_radius_voxels: int,
    max_normal_angle: float,
    max_color_distance: float,
    max_height_delta: float,
    strict_bucket: bool,
) -> tuple[dict[tuple[int, int, int], int], list[dict[str, Any]]]:
    for item in voxels.values():
        item["bucket"] = geometry_bucket(item)
    offsets = neighbor_offsets(connect_radius_voxels)
    unvisited = set(voxels)
    patch_for_voxel: dict[tuple[int, int, int], int] = {}
    patches: list[dict[str, Any]] = []
    next_patch_id = 1
    while unvisited:
        start = unvisited.pop()
        queue = deque([start])
        component = [start]
        while queue:
            key = queue.popleft()
            item = voxels[key]
            for dx, dy, dz in offsets:
                nbr_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                if nbr_key not in unvisited:
                    continue
                nbr = voxels[nbr_key]
                if not edge_allowed(
                    item,
                    nbr,
                    max_normal_angle,
                    max_color_distance,
                    max_height_delta,
                    strict_bucket,
                ):
                    continue
                unvisited.remove(nbr_key)
                queue.append(nbr_key)
                component.append(nbr_key)
        patch_id = next_patch_id
        next_patch_id += 1
        for key in component:
            patch_for_voxel[key] = patch_id
        bucket_counts = Counter(str(voxels[key]["bucket"]) for key in component)
        dominant_bucket, dominant_count = bucket_counts.most_common(1)[0]
        patches.append(
            {
                "patch_id": patch_id,
                "voxel_count": int(len(component)),
                "status": "small_patch" if len(component) < 8 else "geo_patch",
                "geometry_type": dominant_bucket if dominant_count / max(len(component), 1) >= 0.65 else "mixed",
                "bucket_counts": dict(bucket_counts),
            }
        )
    return patch_for_voxel, patches


def enrich_patch_stats(
    voxels: dict[tuple[int, int, int], dict[str, Any]],
    patch_for_voxel: dict[tuple[int, int, int], int],
    patches: list[dict[str, Any]],
) -> None:
    pts_by_patch: dict[int, list[np.ndarray]] = defaultdict(list)
    rgb_by_patch: dict[int, list[np.ndarray]] = defaultdict(list)
    normals_by_patch: dict[int, list[np.ndarray]] = defaultdict(list)
    for key, patch_id in patch_for_voxel.items():
        item = voxels[key]
        pts_by_patch[patch_id].append(item["xyz"])
        rgb_by_patch[patch_id].append(item["rgb"])
        normals_by_patch[patch_id].append(item["normal"])
    by_id = {int(row["patch_id"]): row for row in patches}
    for patch_id, pts in pts_by_patch.items():
        arr = np.vstack(pts)
        rgb = np.vstack(rgb_by_patch[patch_id])
        normals = np.vstack(normals_by_patch[patch_id])
        row = by_id[patch_id]
        row["centroid"] = arr.mean(axis=0).astype(float).tolist()
        row["bbox_3d"] = {"min": arr.min(axis=0).astype(float).tolist(), "max": arr.max(axis=0).astype(float).tolist()}
        row["extent"] = (arr.max(axis=0) - arr.min(axis=0)).astype(float).tolist()
        row["mean_rgb"] = rgb.mean(axis=0).astype(float).tolist()
        normal = normals.mean(axis=0)
        norm = np.linalg.norm(normal)
        row["mean_normal"] = (normal / norm if norm > 1e-9 else normal).astype(float).tolist()


def patch_color(patch_id: int) -> tuple[int, int, int]:
    rng = random.Random(patch_id * 1000003)
    return (rng.randint(40, 245), rng.randint(40, 245), rng.randint(40, 245))


def write_outputs(
    output_dir: Path,
    voxels: dict[tuple[int, int, int], dict[str, Any]],
    patch_for_voxel: dict[tuple[int, int, int], int],
    patches: list[dict[str, Any]],
    voxel_size: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "geo_patches_random_color.ply"
    jsonl_path = output_dir / "geo_patches.jsonl"
    report_path = output_dir / "geo_patch_report.json"
    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(patch_for_voxel)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for key, patch_id in sorted(patch_for_voxel.items(), key=lambda item: item[1]):
            xyz = voxels[key]["xyz"]
            color = patch_color(patch_id)
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {color[0]} {color[1]} {color[2]} {patch_id} 1\n")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in patches:
            out = dict(row)
            out["semantic_label"] = out["geometry_type"]
            out["description"] = f"geometry patch: {out['geometry_type']}"
            out["voxel_size"] = voxel_size
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    bucket_counts = Counter(str(item["bucket"]) for item in voxels.values())
    geometry_counts = Counter(str(row["geometry_type"]) for row in patches)
    report = {
        "schema": "geo-patch-demo/v1",
        "input_ply": str(args.input_ply),
        "output_ply": str(ply_path),
        "output_jsonl": str(jsonl_path),
        "voxel_size": voxel_size,
        "voxel_count": len(voxels),
        "patch_count": len(patches),
        "small_patch_count": sum(1 for row in patches if row["status"] == "small_patch"),
        "bucket_voxel_counts": dict(bucket_counts),
        "patch_geometry_counts": dict(geometry_counts),
        "params": {
            "feature_radius_voxels": args.feature_radius_voxels,
            "connect_radius_voxels": args.connect_radius_voxels,
            "max_normal_angle": args.max_normal_angle,
            "max_color_distance": args.max_color_distance,
            "max_height_delta": args.max_height_delta,
            "strict_bucket": args.strict_bucket,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--feature-radius-voxels", type=int, default=3)
    parser.add_argument("--connect-radius-voxels", type=int, default=1)
    parser.add_argument("--max-normal-angle", type=float, default=28.0)
    parser.add_argument("--max-color-distance", type=float, default=58.0)
    parser.add_argument("--max-height-delta", type=float, default=0.18)
    parser.add_argument("--strict-bucket", action="store_true")
    parser.add_argument("--max-points", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    voxels = read_voxels(args.input_ply, args.voxel_size, args.max_points)
    compute_local_features(voxels, args.feature_radius_voxels)
    patch_for_voxel, patches = build_patches(
        voxels,
        args.connect_radius_voxels,
        args.max_normal_angle,
        args.max_color_distance,
        args.max_height_delta,
        args.strict_bucket,
    )
    enrich_patch_stats(voxels, patch_for_voxel, patches)
    write_outputs(args.output_dir, voxels, patch_for_voxel, patches, args.voxel_size, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
