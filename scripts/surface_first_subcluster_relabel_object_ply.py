#!/usr/bin/env python3
"""Surface-first relabeling at object-internal subcluster granularity.

The first surface-first pass relabels whole Objects. That is too coarse when a
single fused Object contains both a true fine object and contaminated roof/wall
points. This tool splits each object into local 3D chunks and applies the same
geometry-first label rules per chunk.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from surface_first_relabel_object_ply import (
    LABEL_COLORS,
    LABEL_IDS,
    LABEL_NAMES,
    bbox_stats,
    classify_object,
    dominant_semantic,
    pca_shape,
    read_ascii_ply,
    write_ply,
)


def component_offsets() -> list[tuple[int, int, int]]:
    return [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]


OFFSETS = component_offsets()


def connected_voxel_components(points: np.ndarray, voxel_size: float, max_component_points: int) -> list[np.ndarray]:
    """Split points into connected voxel components, then optionally tile large components."""
    if len(points) == 0:
        return []
    voxels = np.floor(points / voxel_size).astype(np.int64)
    by_voxel: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, voxel in enumerate(voxels):
        by_voxel[(int(voxel[0]), int(voxel[1]), int(voxel[2]))].append(i)

    components: list[np.ndarray] = []
    visited: set[tuple[int, int, int]] = set()
    for start in by_voxel:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        cells: list[tuple[int, int, int]] = []
        while queue:
            cell = queue.popleft()
            cells.append(cell)
            for off in OFFSETS:
                nxt = (cell[0] + off[0], cell[1] + off[1], cell[2] + off[2])
                if nxt in by_voxel and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        idx = np.array([i for cell in cells for i in by_voxel[cell]], dtype=np.int64)
        if max_component_points > 0 and len(idx) > max_component_points:
            components.extend(tile_component(points, idx, voxel_size * 4.0, max_component_points))
        else:
            components.append(idx)
    components.sort(key=len, reverse=True)
    return components


def tile_component(points: np.ndarray, component: np.ndarray, tile_size: float, max_points: int) -> list[np.ndarray]:
    """Break a very large connected component into deterministic local tiles."""
    sub_points = points[component]
    cells = np.floor(sub_points / tile_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for local_i, cell in enumerate(cells):
        buckets[(int(cell[0]), int(cell[1]), int(cell[2]))].append(int(component[local_i]))
    out: list[np.ndarray] = []
    for indices in buckets.values():
        arr = np.array(indices, dtype=np.int64)
        if len(arr) <= max_points or max_points <= 0:
            out.append(arr)
            continue
        order = np.argsort(points[arr, 0] + points[arr, 1] * 1e-3 + points[arr, 2] * 1e-6)
        for start in range(0, len(arr), max_points):
            out.append(arr[order[start : start + max_points]])
    return out


def classify_chunk(current_label: str, points: np.ndarray, args: argparse.Namespace) -> tuple[str, str, dict, dict]:
    shape = pca_shape(points)
    bbox = bbox_stats(points)
    if len(points) < args.min_chunk_points:
        return current_label, "small_chunk_keep", shape, bbox
    label, reason = classify_object(current_label, len(points), shape, bbox, args)
    return label, "chunk_" + reason, shape, bbox


def process(args: argparse.Namespace) -> dict:
    props, data, header = read_ascii_ply(args.input_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"x", "y", "z", "red", "green", "blue", "object", "semantic"}
    if not required.issubset(idx):
        raise ValueError(f"missing required fields: {sorted(required - set(idx))}; available={props}")

    out = data.copy()
    object_values = data[:, idx["object"]].astype(np.int64)
    semantic_values = data[:, idx["semantic"]].astype(np.int64)
    objects = sorted(set(int(x) for x in object_values.tolist()))
    before_counts = Counter()
    after_counts = Counter()
    reason_counts = Counter()
    changed_points = 0
    chunk_rows = []

    for object_id in objects:
        object_mask = object_values == object_id
        object_indices = np.where(object_mask)[0]
        rows = data[object_indices]
        points = rows[:, [idx["x"], idx["y"], idx["z"]]]
        label_id, label_ratio, semantic_counts = dominant_semantic(rows[:, idx["semantic"]])
        current_label = LABEL_NAMES.get(label_id, "unknown")
        before_counts[current_label] += int(len(rows))

        object_shape = pca_shape(points)
        object_bbox = bbox_stats(points)
        whole_label, whole_reason = classify_object(current_label, len(rows), object_shape, object_bbox, args)
        needs_subclusters = (
            args.force_all_objects
            or current_label in {"equipment", "unknown", "other", "floor", "wall", "building"}
            or whole_label != current_label
            or label_ratio < args.min_object_label_ratio
        )

        if not needs_subclusters or len(rows) < args.min_chunk_points:
            components = [np.arange(len(rows), dtype=np.int64)]
        else:
            components = connected_voxel_components(points, args.component_voxel_size, args.max_component_points)

        for component_id, local_indices in enumerate(components):
            global_indices = object_indices[local_indices]
            chunk_points = points[local_indices]
            chunk_semantic = semantic_values[global_indices]
            chunk_label_id, chunk_ratio, chunk_counts = dominant_semantic(chunk_semantic)
            chunk_current = LABEL_NAMES.get(chunk_label_id, current_label)
            new_label, reason, shape, bbox = classify_chunk(chunk_current, chunk_points, args)
            new_id = LABEL_IDS.get(new_label, chunk_label_id)
            after_counts[new_label] += int(len(global_indices))
            reason_counts[reason] += int(len(global_indices))
            if new_label != chunk_current:
                changed_points += int(len(global_indices))
                color = LABEL_COLORS.get(new_id, LABEL_COLORS[0])
                out[global_indices, idx["semantic"]] = new_id
                out[global_indices, idx["red"]] = color[0]
                out[global_indices, idx["green"]] = color[1]
                out[global_indices, idx["blue"]] = color[2]
            chunk_rows.append(
                {
                    "object": int(object_id),
                    "component": int(component_id),
                    "points": int(len(global_indices)),
                    "before": chunk_current,
                    "after": new_label,
                    "reason": reason,
                    "semantic_counts": chunk_counts,
                    "dominant_label_ratio": chunk_ratio,
                    "bbox": bbox,
                    "pca": shape,
                    "whole_object_reason": whole_reason,
                }
            )

    write_ply(args.output_ply, header, props, out)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "input_ply": str(args.input_ply),
        "output_ply": str(args.output_ply),
        "objects": len(objects),
        "chunks": len(chunk_rows),
        "points": int(len(data)),
        "changed_points": int(changed_points),
        "changed_ratio": float(changed_points / max(len(data), 1)),
        "before_counts": dict(before_counts),
        "after_counts": dict(after_counts),
        "reason_counts": dict(reason_counts),
        "params": params,
        "chunks_detail": sorted(chunk_rows, key=lambda r: (-r["points"], r["object"], r["component"])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--component-voxel-size", type=float, default=0.12)
    parser.add_argument("--max-component-points", type=int, default=5000)
    parser.add_argument("--min-chunk-points", type=int, default=80)
    parser.add_argument("--min-object-label-ratio", type=float, default=0.95)
    parser.add_argument("--force-all-objects", action="store_true")
    parser.add_argument("--min-surface-points", type=int, default=120)
    parser.add_argument("--min-surface-extent", type=float, default=0.75)
    parser.add_argument("--min-surface-mid-extent", type=float, default=0.35)
    parser.add_argument("--min-surface-planarity", type=float, default=0.16)
    parser.add_argument("--max-surface-scattering", type=float, default=0.18)
    parser.add_argument("--floor-normal-z", type=float, default=0.72)
    parser.add_argument("--wall-normal-z", type=float, default=0.35)
    parser.add_argument("--min-railing-linearity", type=float, default=0.72)
    parser.add_argument("--min-railing-extent", type=float, default=0.8)
    parser.add_argument("--max-railing-mid-extent", type=float, default=0.35)
    parser.add_argument("--max-equipment-extent", type=float, default=1.8)
    parser.add_argument("--min-equipment-z-span", type=float, default=0.2)
    args = parser.parse_args()
    summary = process(args)
    keys = ["objects", "chunks", "points", "changed_points", "changed_ratio", "before_counts", "after_counts", "reason_counts"]
    print(json.dumps({k: summary[k] for k in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
