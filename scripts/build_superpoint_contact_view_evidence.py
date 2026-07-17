#!/usr/bin/env python3
"""Expand immutable Superpoint contact neighbors into already-validated views.

Object review selects a few useful views per object. That is deliberately
efficient for a VLM, but too sparse for image-space edge evidence. This stage
keeps those selected camera views fixed and asks whether each direct 3D contact
neighbor is also first-touch visible in the *same* image. Its output is marked
``edge_only`` and must not be used as a VLM review crop ledger.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def configure_dataset_from_cli() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path)
    args, _unknown = parser.parse_known_args()
    if args.data_dir is None:
        return
    data_dir = args.data_dir.resolve()
    os.environ["SCAN_DATA_DIR"] = str(data_dir)
    os.environ["SCAN_IMAGE_DIR"] = str(data_dir / "image")


configure_dataset_from_cli()
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import config

importlib.reload(config)
from build_object_image_evidence import (
    global_depth_map_path,
    load_global_depth_map,
    load_object_point_samples,
    min_depth_neighborhood,
    priority_mask_path,
    project_points,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def edge_key(object_a: int, object_b: int) -> tuple[int, int]:
    return (min(object_a, object_b), max(object_a, object_b))


def build_neighbor_requests(
    anchors: list[dict[str, Any]], contact_edges: list[dict[str, Any]], candidate_ids: set[int],
) -> dict[tuple[int, int, int], set[int]]:
    """Return target object/view tuples with the source anchors that requested them."""
    neighbors: dict[int, set[int]] = defaultdict(set)
    for edge in contact_edges:
        object_a, object_b = int(edge["object_a"]), int(edge["object_b"])
        if object_a in candidate_ids and object_b in candidate_ids:
            neighbors[object_a].add(object_b)
            neighbors[object_b].add(object_a)
    requests: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for row in anchors:
        anchor = int(row["object_id"])
        frame_id, cam_id = int(row["frame_id"]), int(row["cam_id"])
        for target in neighbors.get(anchor, ()):
            requests[(target, frame_id, cam_id)].add(anchor)
    return requests


def sampled_payload(uv: np.ndarray, depth: np.ndarray, limit: int) -> tuple[list[list[float]], list[float]]:
    if len(uv) <= limit:
        indices = np.arange(len(uv))
    else:
        indices = np.linspace(0, len(uv) - 1, limit, dtype=np.int32)
    return (
        [[round(float(x), 3), round(float(y), 3)] for x, y in uv[indices]],
        [round(float(value), 4) for value in depth[indices]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--anchor-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--contact-edges", type=Path, required=True)
    parser.add_argument("--object-ply", type=Path, required=True)
    parser.add_argument("--global-depth-map-dir", type=Path, required=True)
    parser.add_argument("--priority-dir", type=Path, required=True)
    parser.add_argument("--priority-suffix", default="_priority_refined")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-points-per-object", type=int, default=2500)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--min-projected-points", type=int, default=12)
    parser.add_argument("--depth-tolerance", type=float, default=0.2)
    parser.add_argument("--depth-neighborhood", type=int, default=1)
    parser.add_argument("--save-projected-samples", type=int, default=80)
    parser.add_argument("--max-depth-cache-entries", type=int, default=48)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.data_dir is None:
        raise SystemExit("--data-dir is required")

    anchors = read_jsonl(args.anchor_evidence_jsonl)
    contact_edges = read_jsonl(args.contact_edges)
    candidate_ids = {int(row["object_id"]) for row in anchors}
    requests = build_neighbor_requests(anchors, contact_edges, candidate_ids)
    target_ids = {object_id for object_id, _frame_id, _cam_id in requests}
    samples = load_object_point_samples(args.object_ply, target_ids, args.max_points_per_object, args.seed)
    poses = {int(row["frame_id"]): row for row in config.load_img_pos(0, None)}
    depth_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
    priority_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
    rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()

    for (object_id, frame_id, cam_id), anchor_ids in sorted(requests.items()):
        points = samples.get(object_id)
        pose = poses.get(frame_id)
        if points is None or len(points) < args.min_projected_points:
            failures["insufficient_object_samples"] += 1
            continue
        if pose is None:
            failures["missing_pose"] += 1
            continue
        uv, depth = project_points(points[:, :3], pose, cam_id, args.min_depth)
        if len(uv) < args.min_projected_points:
            failures["low_projected_before_image_filter"] += 1
            continue
        width, height = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
        in_image = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        uv, depth = uv[in_image], depth[in_image]
        if len(uv) < args.min_projected_points:
            failures["low_projected_in_image"] += 1
            continue
        cache_key = (frame_id, cam_id)
        depth_buffer = depth_cache.get(cache_key)
        if depth_buffer is None:
            path = global_depth_map_path(args.global_depth_map_dir, cam_id, frame_id)
            if not path.exists():
                failures["missing_global_depth_map"] += 1
                continue
            depth_buffer = load_global_depth_map(path)
            depth_cache[cache_key] = depth_buffer
            if len(depth_cache) > args.max_depth_cache_entries:
                depth_cache.popitem(last=False)
        uu = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, width - 1)
        vv = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, height - 1)
        local_depth = min_depth_neighborhood(depth_buffer, uu, vv, args.depth_neighborhood)
        visible = np.isfinite(local_depth) & (np.abs(depth - local_depth) <= args.depth_tolerance)
        uv, depth = uv[visible], depth[visible]
        if len(uv) < args.min_projected_points:
            failures["low_projected_after_depth_filter"] += 1
            continue
        priority = priority_cache.get(cache_key)
        if priority is None:
            path = priority_mask_path(args.priority_dir, cam_id, frame_id, args.priority_suffix)
            priority = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
            if priority is None:
                failures["missing_priority_mask"] += 1
                continue
            if priority.shape[:2] != (height, width):
                priority = cv2.resize(priority, (width, height), interpolation=cv2.INTER_NEAREST)
            priority_cache[cache_key] = priority
            if len(priority_cache) > args.max_depth_cache_entries:
                priority_cache.popitem(last=False)
        uu = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, width - 1)
        vv = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, height - 1)
        non_sky = priority[vv, uu] != 6
        uv, depth = uv[non_sky], depth[non_sky]
        if len(uv) < args.min_projected_points:
            failures["low_projected_after_sky_filter"] += 1
            continue
        uv_samples, depth_samples = sampled_payload(uv, depth, args.save_projected_samples)
        image_path = next(
            row["image_path"] for row in anchors
            if int(row["frame_id"]) == frame_id and int(row["cam_id"]) == cam_id
        )
        rows.append({
            "object_id": object_id,
            "frame_id": frame_id,
            "cam_id": cam_id,
            "image_path": image_path,
            "projected_points": int(len(uv)),
            "sample_points": int(len(points)),
            "depth_visible_ratio": round(float(visible.mean()), 6),
            "projected_uv_samples": uv_samples,
            "projected_depth_samples": depth_samples,
            "edge_only": True,
            "anchor_object_ids": sorted(anchor_ids),
        })

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = {
        "anchor_rows": len(anchors),
        "requests": len(requests),
        "accepted_rows": len(rows),
        "accepted_targets": len({int(row["object_id"]) for row in rows}),
        "failure_counts": dict(failures),
        "params": {"depth_tolerance": args.depth_tolerance, "min_projected_points": args.min_projected_points},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
