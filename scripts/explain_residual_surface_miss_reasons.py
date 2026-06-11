#!/usr/bin/env python3
"""Explain why residual points are not absorbed into stable surface objects.

The existing absorbability report intentionally keeps a simple pass/fail count.
For tuning, that is too coarse: a miss caused by label incompatibility should
not be fixed by relaxing plane or color thresholds. This tool classifies the
first failing gate for each residual point against nearby stable surface
objects.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_residual_absorbability import (
    SEMANTIC_NAMES,
    bbox_distance,
    build_index,
    cell_key,
    label_compatible,
    load_surface_objects,
    plane_distance,
    read_ascii_ply,
)


def nearest_object_distance(point: np.ndarray, objects: list[dict[str, Any]]) -> float | None:
    best: float | None = None
    for obj in objects:
        dist = bbox_distance(point, obj["bbox_3d"])
        if best is None or dist < best:
            best = dist
    return best


def classify_point(
    point: np.ndarray,
    color: np.ndarray,
    residual_label: str,
    objects: list[dict[str, Any]],
    candidate_ids: list[int],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    if not candidate_ids:
        nearest = nearest_object_distance(point, objects) if args.compute_nearest_surface_distance else None
        return "no_candidate_cell", {"nearest_surface_bbox_distance": nearest}

    label_candidates = []
    incompatible_labels = Counter()
    for object_idx in candidate_ids:
        obj = objects[object_idx]
        object_label = obj.get("semantic_label", "unknown")
        if label_compatible(residual_label, object_label):
            label_candidates.append(obj)
        else:
            incompatible_labels[object_label] += 1
    if not label_candidates:
        return "label_incompatible", {"candidate_surface_labels": dict(incompatible_labels)}

    bbox_candidates = []
    best_bbox = None
    for obj in label_candidates:
        dist = bbox_distance(point, obj["bbox_3d"])
        best_bbox = dist if best_bbox is None else min(best_bbox, dist)
        if dist <= args.bbox_padding:
            bbox_candidates.append(obj)
    if not bbox_candidates:
        return "bbox_distance_failed", {"best_bbox_distance": best_bbox}

    plane_candidates = []
    best_plane = None
    for obj in bbox_candidates:
        dist = plane_distance(point, obj)
        best_plane = dist if best_plane is None else min(best_plane, dist)
        if dist <= args.max_plane_distance:
            plane_candidates.append(obj)
    if not plane_candidates:
        return "plane_distance_failed", {"best_plane_distance": best_plane}

    best_color = None
    for obj in plane_candidates:
        dist = float(np.linalg.norm(color - np.array(obj.get("mean_color", [0, 0, 0]), dtype=np.float32)))
        best_color = dist if best_color is None else min(best_color, dist)
        if dist <= args.max_color_distance:
            return "matched_surface", {"best_color_distance": dist}
    return "color_distance_failed", {"best_color_distance": best_color}


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float32)
    return {
        "p50": float(np.quantile(arr, 0.5)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.9)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    objects = load_surface_objects(args.objects_jsonl, args.min_object_targets, args.min_object_points)
    index = build_index(objects, args.cell_size, args.bbox_padding)
    files = sorted(args.residual_dir.glob("residuals_frame_*.ply"))
    if args.limit_frames:
        files = files[: args.limit_frames]

    reason_counts = Counter()
    reason_by_label: dict[str, Counter] = defaultdict(Counter)
    label_counts = Counter()
    nearest_distances: dict[str, list[float]] = defaultdict(list)
    best_bbox_distances: dict[str, list[float]] = defaultdict(list)
    best_plane_distances: dict[str, list[float]] = defaultdict(list)
    best_color_distances: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0

    for path in files:
        props, data = read_ascii_ply(path)
        if len(data) == 0:
            continue
        idx = {name: i for i, name in enumerate(props)}
        points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
        colors = data[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.float32)
        labels = data[:, idx["semantic"]].astype(np.int32)
        frame_id = int(path.stem.rsplit("_", 1)[1])
        for row_idx, (point, color, sem) in enumerate(zip(points, colors, labels)):
            total += 1
            residual_label = SEMANTIC_NAMES.get(int(sem), "unknown")
            label_counts[residual_label] += 1
            reason, meta = classify_point(
                point,
                color,
                residual_label,
                objects,
                index.get(cell_key(point, args.cell_size), []),
                args,
            )
            reason_counts[reason] += 1
            reason_by_label[residual_label][reason] += 1
            if meta.get("nearest_surface_bbox_distance") is not None:
                nearest_distances[residual_label].append(float(meta["nearest_surface_bbox_distance"]))
            if meta.get("best_bbox_distance") is not None:
                best_bbox_distances[residual_label].append(float(meta["best_bbox_distance"]))
            if meta.get("best_plane_distance") is not None:
                best_plane_distances[residual_label].append(float(meta["best_plane_distance"]))
            if meta.get("best_color_distance") is not None:
                best_color_distances[residual_label].append(float(meta["best_color_distance"]))
            if len(examples[reason]) < args.example_limit:
                examples[reason].append(
                    {
                        "frame": frame_id,
                        "row": int(row_idx),
                        "label": residual_label,
                        "point": [float(x) for x in point],
                        **meta,
                    }
                )

    return {
        "residual_dir": str(args.residual_dir),
        "objects_jsonl": str(args.objects_jsonl),
        "surface_objects": len(objects),
        "residual_points": int(total),
        "label_counts": dict(label_counts),
        "reason_counts": dict(reason_counts),
        "reason_by_label": {label: dict(counts) for label, counts in reason_by_label.items()},
        "distance_quantiles": {
            "nearest_surface_bbox_distance": {label: quantiles(values) for label, values in nearest_distances.items()},
            "best_bbox_distance": {label: quantiles(values) for label, values in best_bbox_distances.items()},
            "best_plane_distance": {label: quantiles(values) for label, values in best_plane_distances.items()},
            "best_color_distance": {label: quantiles(values) for label, values in best_color_distances.items()},
        },
        "params": {
            "min_object_targets": args.min_object_targets,
            "min_object_points": args.min_object_points,
            "cell_size": args.cell_size,
            "bbox_padding": args.bbox_padding,
            "max_plane_distance": args.max_plane_distance,
            "max_color_distance": args.max_color_distance,
            "compute_nearest_surface_distance": args.compute_nearest_surface_distance,
        },
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-object-targets", type=int, default=5)
    parser.add_argument("--min-object-points", type=int, default=1000)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--bbox-padding", type=float, default=0.35)
    parser.add_argument("--max-plane-distance", type=float, default=0.12)
    parser.add_argument("--max-color-distance", type=float, default=70.0)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--compute-nearest-surface-distance", action="store_true")
    args = parser.parse_args()

    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "surface_objects": report["surface_objects"],
            "residual_points": report["residual_points"],
            "reason_counts": report["reason_counts"],
            "top_labels": sorted(report["label_counts"].items(), key=lambda item: item[1], reverse=True)[:8],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
