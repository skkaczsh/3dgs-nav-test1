#!/usr/bin/env python3
"""Match frame-level Targets against global colored semantic voxels/objects.

This is a prototype bridge between:
1. validated single-frame target projection, and
2. global colored / semantic voxel aggregation.

The goal is to provide additional association evidence beyond local bbox and
tracklet continuity by checking whether a target lands on the same global
colored voxel/object support.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_global_semantic_votes import iter_target_files, voxel_key
from build_targets_from_masks import read_colored_ply


def chebyshev_offsets(radius: int) -> list[tuple[int, int, int]]:
    if radius <= 0:
        return [(0, 0, 0)]
    return [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bbox_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    amin = np.array(a["min"], dtype=np.float64)
    amax = np.array(a["max"], dtype=np.float64)
    bmin = np.array(b["min"], dtype=np.float64)
    bmax = np.array(b["max"], dtype=np.float64)
    gap = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(gap))


def load_targets(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in iter_target_files(path):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return sorted(rows, key=lambda row: (int(row.get("frame_id", 0)), str(row.get("target_id", ""))))


def object_number_of(voxel_row: dict[str, Any]) -> int:
    try:
        return int(voxel_row.get("object_number", 0))
    except (TypeError, ValueError):
        return 0


def load_global_state(voxels_jsonl: Path, objects_jsonl: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[tuple[int, int, int], dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    voxels = load_jsonl(voxels_jsonl)
    objects = {int(row.get("object_number", 0)): row for row in load_jsonl(objects_jsonl)}
    voxel_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    voxels_by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in voxels:
        key_raw = row.get("key") or []
        key = tuple(int(x) for x in key_raw)
        voxel_by_key[key] = row
        voxels_by_object[object_number_of(row)].append(row)
    return voxels, objects, voxel_by_key, voxels_by_object


def target_points(target: dict[str, Any], frame_cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    frame_ply = str(target.get("colored_frame_ply") or "")
    indices = np.array(target.get("point_indices") or [], dtype=np.int64)
    if not frame_ply or indices.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    if frame_ply not in frame_cache:
        frame_cache[frame_ply] = read_colored_ply(Path(frame_ply))
    points, colors = frame_cache[frame_ply]
    valid = indices[(indices >= 0) & (indices < len(points))]
    if valid.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    return points[valid], colors[valid] if len(colors) else np.zeros((len(valid), 3), dtype=np.uint8)


def target_voxel_keys(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    return {voxel_key(point, voxel_size) for point in points}


def overlap_stats(
    target_keys: set[tuple[int, int, int]],
    object_keys: set[tuple[int, int, int]],
    neighbor_radius_voxels: int,
) -> dict[str, Any]:
    exact_overlap = target_keys & object_keys
    exact_overlap_count = len(exact_overlap)
    exact_overlap_ratio = float(exact_overlap_count / max(len(target_keys), 1))
    if neighbor_radius_voxels <= 0:
        return {
            "exact_overlap_count": int(exact_overlap_count),
            "exact_overlap_ratio": exact_overlap_ratio,
            "support_overlap_count": int(exact_overlap_count),
            "support_overlap_ratio": exact_overlap_ratio,
        }
    offsets = chebyshev_offsets(neighbor_radius_voxels)
    support_count = 0
    for key in target_keys:
        if any((key[0] + dx, key[1] + dy, key[2] + dz) in object_keys for dx, dy, dz in offsets):
            support_count += 1
    return {
        "exact_overlap_count": int(exact_overlap_count),
        "exact_overlap_ratio": exact_overlap_ratio,
        "support_overlap_count": int(support_count),
        "support_overlap_ratio": float(support_count / max(len(target_keys), 1)),
    }


def score_match(
    target: dict[str, Any],
    target_keys: set[tuple[int, int, int]],
    obj: dict[str, Any],
    obj_voxels: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    object_keys = {tuple(int(x) for x in row.get("key", [])) for row in obj_voxels}
    overlap = overlap_stats(target_keys, object_keys, args.neighbor_radius_voxels)
    centroid_dist = float(np.linalg.norm(np.array(target.get("centroid", [0, 0, 0]), dtype=np.float64) - np.array(obj.get("centroid", [0, 0, 0]), dtype=np.float64)))
    target_bbox = target.get("bbox_3d", {"min": [0, 0, 0], "max": [0, 0, 0]})
    obj_bbox = obj.get("bbox_3d", {"min": [0, 0, 0], "max": [0, 0, 0]})
    bd = bbox_distance(target_bbox, obj_bbox)
    target_color = np.array(target.get("mean_color", [0, 0, 0]), dtype=np.float64)
    object_color = np.array(np.mean([row.get("mean_color", [0, 0, 0]) for row in obj_voxels], axis=0), dtype=np.float64) if obj_voxels else np.zeros(3, dtype=np.float64)
    color_dist = float(np.linalg.norm(target_color - object_color))
    same_label = str(target.get("label", "")) == str(obj.get("semantic_label", ""))
    score = (
        overlap["support_overlap_ratio"] * args.overlap_weight
        - centroid_dist * args.centroid_penalty
        - bd * args.bbox_penalty
        - (color_dist / 255.0) * args.color_penalty
        + (args.label_bonus if same_label else 0.0)
    )
    return {
        "object_id": obj.get("object_id", ""),
        "object_number": int(obj.get("object_number", 0)),
        "semantic_label": obj.get("semantic_label", ""),
        "display_identity": obj.get("display_identity", ""),
        "voxel_overlap_count": int(overlap["exact_overlap_count"]),
        "support_overlap_count": int(overlap["support_overlap_count"]),
        "target_voxel_count": int(len(target_keys)),
        "overlap_ratio": overlap["exact_overlap_ratio"],
        "support_overlap_ratio": overlap["support_overlap_ratio"],
        "centroid_distance": centroid_dist,
        "bbox_distance": bd,
        "color_distance": color_dist,
        "same_label": same_label,
        "score": float(score),
    }


def match_targets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = load_targets(args.targets)
    _, objects_by_number, _, voxels_by_object = load_global_state(args.global_voxels, args.global_objects)
    frame_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    matches: list[dict[str, Any]] = []
    matched_count = 0
    for target in targets:
        label = str(target.get("label", ""))
        if args.match_labels and label not in args.match_labels:
            continue
        points, _colors = target_points(target, frame_cache)
        if len(points) == 0:
            matches.append({
                "target_id": target.get("target_id", ""),
                "frame_id": int(target.get("frame_id", 0)),
                "label": label,
                "status": "missing_points",
                "candidates": [],
            })
            continue
        keys = target_voxel_keys(points, args.voxel_size)
        candidates = []
        for object_number, obj in objects_by_number.items():
            if object_number <= 0:
                continue
            candidate = score_match(target, keys, obj, voxels_by_object.get(object_number, []), args)
            if candidate["support_overlap_ratio"] <= 0 and candidate["centroid_distance"] > args.max_centroid_distance:
                continue
            candidates.append(candidate)
        candidates.sort(key=lambda row: row["score"], reverse=True)
        best = candidates[0] if candidates else None
        status = "unmatched"
        if best and best["score"] >= args.min_score and (best["support_overlap_ratio"] > 0 or best["centroid_distance"] <= args.max_centroid_distance):
            status = "matched"
            matched_count += 1
        matches.append({
            "target_id": target.get("target_id", ""),
            "frame_id": int(target.get("frame_id", 0)),
            "label": label,
            "point_count": int(target.get("cluster_size", len(points))),
            "target_voxel_count": int(len(keys)),
            "status": status,
            "best_match": best,
            "candidates": candidates[: args.max_candidates],
        })
    report = {
        "targets": str(args.targets),
        "global_voxels": str(args.global_voxels),
        "global_objects": str(args.global_objects),
        "match_count": matched_count,
        "target_count": len(matches),
        "match_ratio": float(matched_count / max(len(matches), 1)),
        "params": {
            "voxel_size": args.voxel_size,
            "match_labels": list(args.match_labels),
            "max_centroid_distance": args.max_centroid_distance,
            "min_score": args.min_score,
            "overlap_weight": args.overlap_weight,
            "neighbor_radius_voxels": args.neighbor_radius_voxels,
            "centroid_penalty": args.centroid_penalty,
            "bbox_penalty": args.bbox_penalty,
            "color_penalty": args.color_penalty,
            "label_bonus": args.label_bonus,
            "max_candidates": args.max_candidates,
        },
    }
    return matches, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--global-voxels", type=Path, required=True)
    parser.add_argument("--global-objects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.06)
    parser.add_argument("--match-label", dest="match_labels", action="append", default=[])
    parser.add_argument("--max-centroid-distance", type=float, default=1.5)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--overlap-weight", type=float, default=3.0)
    parser.add_argument("--neighbor-radius-voxels", type=int, default=0)
    parser.add_argument("--centroid-penalty", type=float, default=0.4)
    parser.add_argument("--bbox-penalty", type=float, default=0.4)
    parser.add_argument("--color-penalty", type=float, default=0.8)
    parser.add_argument("--label-bonus", type=float, default=0.3)
    parser.add_argument("--max-candidates", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches, report = match_targets(args)
    with (args.output_dir / "target_global_object_matches.jsonl").open("w", encoding="utf-8") as f:
        for row in matches:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "target_global_object_match_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
