#!/usr/bin/env python3
"""Split mixed priority objects with local voxel PCA geometry.

The priority semantic stage is image-driven and can merge large connected
surfaces under a wrong class, e.g. horizontal roof/ground points inside a huge
`wall` object. This script keeps the point set intact but rewrites object ids
and semantic ids for conflicted large surface objects:

- stream the ASCII PLY by consecutive object id
- for selected conflicted surface objects, classify local voxels by PCA normal
- connect neighboring voxels with the same local class into child objects
- write a replacement priority-object PLY/JSONL pair

It is a deterministic geometry refinement stage, not a visual relabeler.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LABEL_TO_SEMANTIC = {
    "unknown": 0,
    "wall": 2,
    "floor": 3,
    "ground": 3,
    "grass": 5,
    "car": 8,
    "railing": 9,
}

SPLIT_CANDIDATE_LABELS = {"floor", "wall", "grass", "railing"}


@dataclass
class PlyHeader:
    props: list[str]
    vertex_count: int
    header_lines: list[str]


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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_header(path: Path) -> PlyHeader:
    props: list[str] = []
    vertex_count = 0
    header_lines: list[str] = []
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines.append(line)
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "format" and parts[1] != "ascii":
                raise ValueError(f"Only ASCII PLY is supported: {path}")
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
    if vertex_count <= 0:
        raise ValueError(f"Missing vertex count: {path}")
    return PlyHeader(props=props, vertex_count=vertex_count, header_lines=header_lines)


def pca_stats(points: np.ndarray) -> dict[str, Any]:
    if len(points) == 0:
        return {
            "pca_eigenvalues": [0.0, 0.0, 0.0],
            "pca_normal": [0.0, 0.0, 1.0],
            "thickness_rms": 0.0,
            "spread_rms": 0.0,
            "planarity": 0.0,
            "extent": [0.0, 0.0, 0.0],
            "max_extent": 0.0,
        }
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
    thickness = float(np.sqrt(max(float(eigvals[-1]), 0.0)))
    spread = float(np.sqrt(max(float(eigvals[0]), 0.0)))
    return {
        "pca_eigenvalues": eigvals.astype(float).tolist(),
        "pca_normal": eigvecs[:, -1].astype(float).tolist(),
        "thickness_rms": thickness,
        "spread_rms": spread,
        "planarity": float(1.0 - thickness / max(spread, 1e-6)),
        "extent": extents.astype(float).tolist(),
        "max_extent": float(extents.max()),
    }


def linearity_from_eigenvalues(eigvals: list[float]) -> float:
    if len(eigvals) < 2 or float(eigvals[0]) <= 1e-9:
        return 0.0
    return max(0.0, 1.0 - float(eigvals[1]) / max(float(eigvals[0]), 1e-9))


def object_summary(object_id: int, label: str, status: str, points: np.ndarray, colors: np.ndarray,
                   source: str, parent_id: int | None = None, split_reason: str | None = None) -> dict[str, Any]:
    stats = pca_stats(points)
    out = {
        "object_id": int(object_id),
        "semantic_label": label,
        "description": f"priority-layer {label} component",
        "status": status,
        "point_count": int(len(points)),
        "centroid": points.mean(axis=0).astype(float).tolist(),
        "bbox_min": points.min(axis=0).astype(float).tolist(),
        "bbox_max": points.max(axis=0).astype(float).tolist(),
        "bbox_3d": {
            "min": points.min(axis=0).astype(float).tolist(),
            "max": points.max(axis=0).astype(float).tolist(),
        },
        "mean_color": colors.astype(np.float32).mean(axis=0).astype(float).tolist() if len(colors) else [0.0, 0.0, 0.0],
        **stats,
        "source": source,
    }
    if parent_id is not None:
        out["split_parent_object_id"] = int(parent_id)
        out["split_reason"] = split_reason or "local_geometry"
        out["description"] = f"priority-layer {label} child split from object {parent_id}"
    return out


def load_objects(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["object_id"]): row for row in read_jsonl(path)}


def load_split_candidates(path: Path | None, objects: dict[int, dict[str, Any]], args: argparse.Namespace) -> set[int]:
    if not path:
        return {
            oid for oid, obj in objects.items()
            if str(obj.get("semantic_label")) in SPLIT_CANDIDATE_LABELS and int(obj.get("point_count") or 0) >= args.min_split_points
        }
    candidates: set[int] = set()
    for row in read_jsonl(path):
        oid = int(row["object_id"])
        obj = objects.get(oid)
        if not obj:
            continue
        label = str(obj.get("semantic_label") or "unknown")
        if label not in SPLIT_CANDIDATE_LABELS:
            continue
        if int(obj.get("point_count") or 0) < args.min_split_points:
            continue
        action = str(row.get("suggested_action") or "")
        reasons = set(str(r) for r in row.get("reasons") or [])
        if "split" in action or reasons & {
            "wall_has_horizontal_normal",
            "wall_oblique_normal",
            "wall_low_planarity",
            "wall_high_thickness",
            "floor_large_vertical_extent",
            "floor_not_horizontal",
            "grass_large_vertical_extent",
            "grass_low_planarity",
            "railing_surface_like_horizontal",
            "railing_clean_horizontal_surface",
            "railing_overmerged_extent",
        }:
            candidates.add(oid)
    return candidates


def classify_local_voxel(source_label: str, points: np.ndarray, args: argparse.Namespace) -> str:
    if len(points) < args.min_cell_points:
        return "unknown"
    stats = pca_stats(points)
    normal = stats["pca_normal"]
    ext = stats["extent"]
    eigvals = stats["pca_eigenvalues"]
    nz = abs(float(normal[2]))
    planarity = float(stats["planarity"])
    thickness = float(stats["thickness_rms"])
    linearity = linearity_from_eigenvalues(eigvals)
    xy_minor = min(float(ext[0]), float(ext[1]))

    if source_label == "wall":
        if nz >= args.horizontal_normal_z and planarity >= args.min_surface_planarity and thickness <= args.max_horizontal_thickness:
            return args.horizontal_label
        if nz <= args.vertical_normal_z and planarity >= args.min_vertical_planarity:
            return "wall"
        return "unknown"
    if source_label == "floor":
        if nz >= args.floor_keep_normal_z and planarity >= args.min_surface_planarity:
            return args.horizontal_label
        if nz <= args.vertical_normal_z and planarity >= args.min_vertical_planarity:
            return "wall"
        return "unknown"
    if source_label == "grass":
        if planarity < args.grass_min_planarity and thickness <= args.max_grass_thickness:
            return "grass"
        if nz >= args.horizontal_normal_z and planarity >= args.min_surface_planarity:
            return args.horizontal_label
        return "unknown"
    if source_label == "railing":
        if (
            nz >= args.horizontal_normal_z
            and planarity >= args.min_surface_planarity
            and (linearity < args.railing_keep_linearity or xy_minor >= args.railing_max_minor_extent)
        ):
            return args.horizontal_label
        if nz <= args.vertical_normal_z and planarity >= args.min_vertical_planarity and linearity < args.railing_keep_linearity:
            return "wall"
        if linearity >= args.railing_keep_linearity and xy_minor <= args.railing_max_minor_extent:
            return "railing"
        return "unknown"
    return source_label


def classify_from_local_stats(
    source_label: str,
    nz: float,
    planarity: float,
    thickness: float,
    linearity: float,
    args: argparse.Namespace,
) -> str:
    if source_label == "wall":
        if nz >= args.horizontal_normal_z and planarity >= args.min_surface_planarity and thickness <= args.max_horizontal_thickness:
            return args.horizontal_label
        if nz <= args.vertical_normal_z and planarity >= args.min_vertical_planarity:
            return "wall"
        return "unknown"
    if source_label == "floor":
        if nz >= args.floor_keep_normal_z and planarity >= args.min_surface_planarity:
            return args.horizontal_label
        if nz <= args.vertical_normal_z and planarity >= args.min_vertical_planarity:
            return "wall"
        return "unknown"
    if source_label == "grass":
        if planarity < args.grass_min_planarity and thickness <= args.max_grass_thickness:
            return "grass"
        if nz >= args.horizontal_normal_z and planarity >= args.min_surface_planarity:
            return args.horizontal_label
        return "unknown"
    if source_label == "railing":
        if nz >= args.horizontal_normal_z and planarity >= args.min_surface_planarity and linearity < args.railing_keep_linearity:
            return args.horizontal_label
        if nz <= args.vertical_normal_z and planarity >= args.min_vertical_planarity and linearity < args.railing_keep_linearity:
            return "wall"
        if linearity >= args.railing_keep_linearity:
            return "railing"
        return "unknown"
    return source_label


def classify_local_voxels(
    source_label: str,
    points: np.ndarray,
    inverse: np.ndarray,
    cell_count: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, Counter]:
    pts = points.astype(np.float64, copy=False)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    counts = np.bincount(inverse, minlength=cell_count).astype(np.float64)
    sx = np.bincount(inverse, weights=x, minlength=cell_count)
    sy = np.bincount(inverse, weights=y, minlength=cell_count)
    sz = np.bincount(inverse, weights=z, minlength=cell_count)
    sxx = np.bincount(inverse, weights=x * x, minlength=cell_count)
    syy = np.bincount(inverse, weights=y * y, minlength=cell_count)
    szz = np.bincount(inverse, weights=z * z, minlength=cell_count)
    sxy = np.bincount(inverse, weights=x * y, minlength=cell_count)
    sxz = np.bincount(inverse, weights=x * z, minlength=cell_count)
    syz = np.bincount(inverse, weights=y * z, minlength=cell_count)

    labels: list[str] = []
    point_counts = Counter()
    for i in range(cell_count):
        c = counts[i]
        if c < args.min_cell_points:
            label = "unknown"
            labels.append(label)
            point_counts[label] += int(c)
            continue
        mx = sx[i] / c
        my = sy[i] / c
        mz = sz[i] / c
        cov = np.array(
            [
                [sxx[i] / c - mx * mx, sxy[i] / c - mx * my, sxz[i] / c - mx * mz],
                [sxy[i] / c - mx * my, syy[i] / c - my * my, syz[i] / c - my * mz],
                [sxz[i] / c - mx * mz, syz[i] / c - my * mz, szz[i] / c - mz * mz],
            ],
            dtype=np.float64,
        )
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        normal = eigvecs[:, -1]
        nz = abs(float(normal[2]))
        thickness = float(np.sqrt(max(float(eigvals[-1]), 0.0)))
        spread = float(np.sqrt(max(float(eigvals[0]), 0.0)))
        planarity = float(1.0 - thickness / max(spread, 1e-6))
        lin = linearity_from_eigenvalues(eigvals.astype(float).tolist())
        label = classify_from_local_stats(source_label, nz, planarity, thickness, lin, args)
        labels.append(label)
        point_counts[label] += int(c)
    return np.array(labels, dtype=object), point_counts


def connected_cell_components(cell_coords: np.ndarray, cell_labels: np.ndarray, connectivity: int) -> np.ndarray:
    n = len(cell_coords)
    uf = UnionFind(n)
    lookup = {tuple(v.tolist()): i for i, v in enumerate(cell_coords)}
    if connectivity == 6:
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    elif connectivity == 26:
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) > (0, 0, 0)
        ]
    else:
        raise ValueError("--cell-connectivity must be 6 or 26")
    for i, voxel in enumerate(cell_coords):
        base = tuple(voxel.tolist())
        label = cell_labels[i]
        for dx, dy, dz in offsets:
            j = lookup.get((base[0] + dx, base[1] + dy, base[2] + dz))
            if j is not None and cell_labels[j] == label:
                uf.union(i, j)
    return np.array([uf.find(i) for i in range(n)], dtype=np.int32)


def split_object(points: np.ndarray, colors: np.ndarray, source_obj: dict[str, Any], args: argparse.Namespace,
                 next_id: int) -> tuple[np.ndarray, list[dict[str, Any]], int, dict[str, Any]]:
    source_label = str(source_obj.get("semantic_label") or "unknown")
    source_id = int(source_obj["object_id"])
    voxels = np.floor(points / args.local_voxel_size).astype(np.int32)
    cell_coords, inverse, cell_counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    cell_label_arr, cell_stats = classify_local_voxels(source_label, points, inverse, len(cell_coords), args)
    roots = connected_cell_components(cell_coords, cell_label_arr, args.cell_connectivity)
    component_key_for_point = np.array([(str(cell_label_arr[cell]), int(roots[cell])) for cell in inverse], dtype=object)

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for point_idx, key in enumerate(component_key_for_point):
        grouped[(str(key[0]), int(key[1]))].append(point_idx)

    assignments = np.full(len(points), -1, dtype=np.int64)
    child_objects: list[dict[str, Any]] = []
    residual_points: list[int] = []
    kept_components = 0
    small_components = 0
    for (label, _root), ids_list in sorted(grouped.items(), key=lambda item: -len(item[1])):
        ids = np.asarray(ids_list, dtype=np.int64)
        min_points = args.min_unknown_child_points if label == "unknown" else args.min_child_points
        if len(ids) < min_points:
            residual_points.extend(int(i) for i in ids)
            small_components += 1
            continue
        object_id = next_id
        next_id += 1
        assignments[ids] = object_id
        child_objects.append(
            object_summary(
                object_id,
                label,
                f"priority_{label}_local_geometry_child",
                points[ids],
                colors[ids],
                str(source_obj.get("source") or ""),
                parent_id=source_id,
                split_reason=f"{source_label}_local_voxel_pca",
            )
        )
        child_objects[-1]["semantic_label_original"] = source_label
        child_objects[-1]["local_geometry_cell_point_counts"] = dict(cell_stats)
        kept_components += 1

    if residual_points:
        ids = np.asarray(residual_points, dtype=np.int64)
        object_id = next_id
        next_id += 1
        assignments[ids] = object_id
        child_objects.append(
            object_summary(
                object_id,
                "unknown",
                "priority_local_geometry_small_component_residual",
                points[ids],
                colors[ids],
                str(source_obj.get("source") or ""),
                parent_id=source_id,
                split_reason=f"{source_label}_small_local_components",
            )
        )
        child_objects[-1]["semantic_label_original"] = source_label

    if np.any(assignments < 0):
        raise RuntimeError(f"Unassigned split points for object {source_id}: {int(np.count_nonzero(assignments < 0))}")

    split_report = {
        "object_id": source_id,
        "source_label": source_label,
        "source_points": int(len(points)),
        "local_voxel_size": args.local_voxel_size,
        "local_cell_count": int(len(cell_coords)),
        "cell_point_counts_by_label": dict(cell_stats),
        "child_count": len(child_objects),
        "kept_components": kept_components,
        "small_components": small_components,
        "child_point_counts_by_label": dict(Counter(obj["semantic_label"] for obj in child_objects)),
    }
    return assignments, child_objects, next_id, split_report


def parse_group_lines(lines: list[str], idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.array([[float(x) for x in line.strip().split()] for line in lines], dtype=np.float64)
    points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    colors = data[:, [idx["red"], idx["green"], idx["blue"]]].astype(np.uint8)
    objects = data[:, idx["object"]].astype(np.int64)
    return points, colors, objects


def write_group_passthrough(dst, lines: list[str], source_obj: dict[str, Any], output_objects: list[dict[str, Any]],
                            report_counts: Counter) -> None:
    for line in lines:
        dst.write(line)
    output_objects.append(dict(source_obj))
    report_counts["passthrough_objects"] += 1
    report_counts[f"passthrough_label:{source_obj.get('semantic_label', 'unknown')}"] += int(source_obj.get("point_count") or len(lines))


def write_group_split(dst, lines: list[str], idx: dict[str, int], assignments: np.ndarray,
                      child_objects: list[dict[str, Any]], report_counts: Counter) -> None:
    child_label_by_id = {int(obj["object_id"]): str(obj["semantic_label"]) for obj in child_objects}
    for line, object_id in zip(lines, assignments):
        parts = line.strip().split()
        label = child_label_by_id[int(object_id)]
        parts[idx["object"]] = str(int(object_id))
        parts[idx["semantic"]] = str(LABEL_TO_SEMANTIC.get(label, 0))
        dst.write(" ".join(parts) + "\n")
        report_counts[f"output_points:{label}"] += 1


def process_object_group(
    dst,
    lines: list[str],
    object_id: int,
    idx: dict[str, int],
    objects: dict[int, dict[str, Any]],
    split_candidates: set[int],
    output_objects: list[dict[str, Any]],
    split_reports: list[dict[str, Any]],
    report_counts: Counter,
    next_id: int,
    args: argparse.Namespace,
) -> int:
    source_obj = objects.get(object_id)
    if not source_obj:
        for line in lines:
            dst.write(line)
        report_counts["missing_metadata_objects"] += 1
        return next_id
    if object_id not in split_candidates:
        write_group_passthrough(dst, lines, source_obj, output_objects, report_counts)
        return next_id

    print(
        json.dumps(
            {
                "event": "split_object_start",
                "object_id": object_id,
                "label": source_obj.get("semantic_label"),
                "point_count": len(lines),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    points, colors, _objects = parse_group_lines(lines, idx)
    assignments, child_objects, next_id, split_report = split_object(points, colors, source_obj, args, next_id)
    print(json.dumps({"event": "split_object_done", **split_report}, ensure_ascii=False), flush=True)
    write_group_split(dst, lines, idx, assignments, child_objects, report_counts)
    output_objects.extend(child_objects)
    split_reports.append(split_report)
    report_counts["split_source_objects"] += 1
    report_counts[f"split_source_label:{source_obj.get('semantic_label', 'unknown')}"] += int(len(lines))
    return next_id


def process_ply_noncontiguous(
    input_ply: Path,
    output_ply: Path,
    header: PlyHeader,
    idx: dict[str, int],
    objects: dict[int, dict[str, Any]],
    split_candidates: set[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter, int]:
    """Rewrite a viewer PLY without assuming object ids are contiguous.

    Viewer exports are often grouped by frame/target rather than by object. The
    old streaming implementation treated every repeated object run as a separate
    object, duplicating metadata and producing invalid reports.  Candidate
    objects are small enough for this pass, so collect only their lines and
    stream all non-candidates through immediately.
    """
    output_objects: list[dict[str, Any]] = []
    split_reports: list[dict[str, Any]] = []
    report_counts = Counter()
    next_id = args.next_object_id_base
    written_passthrough_objects: set[int] = set()
    candidate_lines: dict[int, list[str]] = {oid: [] for oid in split_candidates}

    with input_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            if line.strip() == "end_header":
                break

        for _line_no in range(header.vertex_count):
            line = src.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) <= idx["object"]:
                continue
            object_id = int(round(float(parts[idx["object"]])))
            if object_id in split_candidates:
                candidate_lines.setdefault(object_id, []).append(line)
                continue
            dst.write(line)
            if object_id not in written_passthrough_objects:
                source_obj = objects.get(object_id)
                if source_obj:
                    output_objects.append(dict(source_obj))
                    report_counts["passthrough_objects"] += 1
                    written_passthrough_objects.add(object_id)
                else:
                    report_counts["missing_metadata_objects"] += 1

        for object_id in sorted(candidate_lines):
            lines = candidate_lines[object_id]
            if not lines:
                continue
            source_obj = objects.get(object_id)
            if not source_obj:
                for line in lines:
                    dst.write(line)
                report_counts["missing_metadata_objects"] += 1
                continue
            print(
                json.dumps(
                    {
                        "event": "split_object_start",
                        "object_id": object_id,
                        "label": source_obj.get("semantic_label"),
                        "point_count": len(lines),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            points, colors, _objects = parse_group_lines(lines, idx)
            assignments, child_objects, next_id, split_report = split_object(points, colors, source_obj, args, next_id)
            print(json.dumps({"event": "split_object_done", **split_report}, ensure_ascii=False), flush=True)
            write_group_split(dst, lines, idx, assignments, child_objects, report_counts)
            output_objects.extend(child_objects)
            split_reports.append(split_report)
            report_counts["split_source_objects"] += 1
            report_counts[f"split_source_label:{source_obj.get('semantic_label', 'unknown')}"] += int(len(lines))

    return output_objects, split_reports, report_counts, next_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--conflicts-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="priority_objects_local_geometry")
    parser.add_argument("--next-object-id-base", type=int, default=3_000_000)
    parser.add_argument("--min-split-points", type=int, default=2_000)
    parser.add_argument("--local-voxel-size", type=float, default=0.60)
    parser.add_argument("--cell-connectivity", type=int, choices=[6, 26], default=6)
    parser.add_argument("--min-cell-points", type=int, default=18)
    parser.add_argument("--min-child-points", type=int, default=120)
    parser.add_argument("--min-unknown-child-points", type=int, default=300)
    parser.add_argument("--horizontal-normal-z", type=float, default=0.85)
    parser.add_argument("--floor-keep-normal-z", type=float, default=0.75)
    parser.add_argument("--vertical-normal-z", type=float, default=0.38)
    parser.add_argument("--min-surface-planarity", type=float, default=0.65)
    parser.add_argument("--min-vertical-planarity", type=float, default=0.45)
    parser.add_argument("--max-horizontal-thickness", type=float, default=0.45)
    parser.add_argument("--grass-min-planarity", type=float, default=0.30)
    parser.add_argument("--max-grass-thickness", type=float, default=0.80)
    parser.add_argument("--railing-keep-linearity", type=float, default=0.82)
    parser.add_argument("--railing-max-minor-extent", type=float, default=1.20)
    parser.add_argument("--horizontal-label", choices=["floor", "ground"], default="floor")
    args = parser.parse_args()

    header = read_header(args.input_ply)
    idx = {name: i for i, name in enumerate(header.props)}
    for name in ("x", "y", "z", "red", "green", "blue", "object", "semantic"):
        if name not in idx:
            raise ValueError(f"Input PLY missing property {name}: {args.input_ply}")

    objects = load_objects(args.objects_jsonl)
    split_candidates = load_split_candidates(args.conflicts_jsonl, objects, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = args.output_dir / f"{args.output_prefix}.ply"
    output_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    output_split_report_jsonl = args.output_dir / f"{args.output_prefix}_split_reports.jsonl"
    output_report = args.output_dir / f"{args.output_prefix}_report.json"

    output_objects, split_reports, report_counts, _next_id = process_ply_noncontiguous(
        args.input_ply,
        output_ply,
        header,
        idx,
        objects,
        split_candidates,
        args,
    )

    output_objects.sort(key=lambda row: int(row["object_id"]))
    write_jsonl(output_jsonl, output_objects)
    write_jsonl(output_split_report_jsonl, split_reports)
    object_label_counts = Counter(str(obj.get("semantic_label") or "unknown") for obj in output_objects)
    point_label_counts = Counter()
    for obj in output_objects:
        point_label_counts[str(obj.get("semantic_label") or "unknown")] += int(obj.get("point_count") or 0)
    report = {
        "input_ply": str(args.input_ply),
        "objects_jsonl": str(args.objects_jsonl),
        "conflicts_jsonl": str(args.conflicts_jsonl) if args.conflicts_jsonl else None,
        "output_ply": str(output_ply),
        "output_jsonl": str(output_jsonl),
        "output_split_report_jsonl": str(output_split_report_jsonl),
        "input_vertex_count": header.vertex_count,
        "input_object_count": len(objects),
        "split_candidate_count": len(split_candidates),
        "output_object_count": len(output_objects),
        "split_source_object_count": len(split_reports),
        "object_label_counts": dict(object_label_counts),
        "point_label_counts": dict(point_label_counts),
        "counts": dict(report_counts),
    }
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
