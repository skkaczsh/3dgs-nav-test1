#!/usr/bin/env python3
"""Audit whether MANIFOLD .lx section points behave like world coordinates.

The current semantic route assumes .lx points are already in the global/world
frame and therefore projects them by applying the inverse pose to each section.
If a future dataset stores points in the local LiDAR frame instead, sync and
semantic projection diagnostics will be misleading.  This script makes that
assumption explicit and testable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from project_priority_masks_to_lx import read_lx_points, read_lx_sections, transform_world_to_lidar


def parse_int_range(text: str) -> list[int]:
    values: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [int(x) for x in chunk.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"Bad range chunk: {chunk}")
            start, end = parts[0], parts[1]
            step = parts[2] if len(parts) == 3 else 1
            stop = end + (1 if step > 0 else -1)
            values.extend(range(start, stop, step))
        else:
            values.append(int(chunk))
    return sorted(set(values))


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def axis_correlations(a: np.ndarray, b: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for axis in range(3):
        if len(a) < 2 or np.std(a[:, axis]) == 0 or np.std(b[:, axis]) == 0:
            out.append(None)
        else:
            out.append(float(np.corrcoef(a[:, axis], b[:, axis])[0, 1]))
    return out


def classify_coordinate_frame(
    raw_centroid_span: np.ndarray,
    pose_span: np.ndarray,
    lidar_centroid_span: np.ndarray,
    correlations: list[float | None],
    min_corr: float,
    max_lidar_span: float,
) -> dict[str, Any]:
    valid_corr = [abs(x) for x in correlations if x is not None]
    corr_ok = len(valid_corr) >= 2 and min(valid_corr) >= min_corr
    lidar_span_ok = float(np.max(lidar_centroid_span)) <= max_lidar_span
    pose_span_large = float(np.max(pose_span)) > max_lidar_span
    world_like = corr_ok and lidar_span_ok and pose_span_large
    return {
        "status": "world_coordinates_likely" if world_like else "inconclusive_or_local_coordinates_possible",
        "world_like": bool(world_like),
        "corr_ok": bool(corr_ok),
        "lidar_span_ok": bool(lidar_span_ok),
        "pose_span_large": bool(pose_span_large),
        "thresholds": {
            "min_corr": float(min_corr),
            "max_lidar_centroid_span": float(max_lidar_span),
        },
        "raw_centroid_span": [float(x) for x in raw_centroid_span],
        "pose_span": [float(x) for x in pose_span],
        "lidar_centroid_span": [float(x) for x in lidar_centroid_span],
        "raw_centroid_vs_pose_corr": correlations,
    }


def audit_frames(lx_file: Path, frame_ids: list[int], min_corr: float, max_lidar_span: float) -> dict[str, Any]:
    sections = read_lx_sections(lx_file)
    poses = {row["frame_id"]: row for row in config.load_img_pos(min(frame_ids), max(frame_ids))}
    frame_rows = []
    raw_centroids = []
    pose_positions = []
    lidar_centroids = []
    with lx_file.open("rb") as handle:
        for frame_id in frame_ids:
            if frame_id >= len(sections) or frame_id not in poses:
                continue
            points = read_lx_points(handle, sections[frame_id])
            if len(points) == 0:
                continue
            pose = poses[frame_id]
            raw_median = np.median(points, axis=0).astype(np.float64)
            raw_mean = np.mean(points, axis=0).astype(np.float64)
            lidar_points = transform_world_to_lidar(points, pose)
            lidar_median = np.median(lidar_points, axis=0).astype(np.float64)
            raw_centroids.append(raw_median)
            pose_positions.append(np.asarray(pose["pos"], dtype=np.float64))
            lidar_centroids.append(lidar_median)
            frame_rows.append({
                "frame_id": int(frame_id),
                "point_count": int(len(points)),
                "pose": [float(x) for x in pose["pos"]],
                "raw_median": [float(x) for x in raw_median],
                "raw_mean": [float(x) for x in raw_mean],
                "raw_min": [float(x) for x in points.min(axis=0)],
                "raw_max": [float(x) for x in points.max(axis=0)],
                "raw_median_to_pose_distance": float(np.linalg.norm(raw_median - pose["pos"])),
                "world_to_lidar_median": [float(x) for x in lidar_median],
                "world_to_lidar_min": [float(x) for x in lidar_points.min(axis=0)],
                "world_to_lidar_max": [float(x) for x in lidar_points.max(axis=0)],
            })
    if len(frame_rows) < 2:
        raise SystemExit("Need at least two valid frames for coordinate-frame audit.")
    raw_centroids_arr = np.asarray(raw_centroids, dtype=np.float64)
    pose_positions_arr = np.asarray(pose_positions, dtype=np.float64)
    lidar_centroids_arr = np.asarray(lidar_centroids, dtype=np.float64)
    raw_span = raw_centroids_arr.max(axis=0) - raw_centroids_arr.min(axis=0)
    pose_span = pose_positions_arr.max(axis=0) - pose_positions_arr.min(axis=0)
    lidar_span = lidar_centroids_arr.max(axis=0) - lidar_centroids_arr.min(axis=0)
    correlations = axis_correlations(raw_centroids_arr, pose_positions_arr)
    classification = classify_coordinate_frame(raw_span, pose_span, lidar_span, correlations, min_corr, max_lidar_span)
    return {
        "lx_file": str(lx_file),
        "calib_file": config.CALIB_FILE,
        "img_pos_file": config.IMG_POS_FILE,
        "frame_ids_requested": frame_ids,
        "frame_count": len(frame_rows),
        "classification": classification,
        "raw_median_to_pose_distance": numeric_summary(
            np.asarray([row["raw_median_to_pose_distance"] for row in frame_rows], dtype=np.float64)
        ),
        "frames": frame_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx-file", type=Path, required=True)
    parser.add_argument("--frames", default="0,100,1000,2000,3400,5200,6180")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-corr", type=float, default=0.95)
    parser.add_argument("--max-lidar-centroid-span", type=float, default=5.0)
    args = parser.parse_args()

    frame_ids = parse_int_range(args.frames)
    report = audit_frames(args.lx_file, frame_ids, args.min_corr, args.max_lidar_centroid_span)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["classification"]["status"],
        "world_like": report["classification"]["world_like"],
        "frame_count": report["frame_count"],
        "raw_centroid_vs_pose_corr": report["classification"]["raw_centroid_vs_pose_corr"],
        "lidar_centroid_span": report["classification"]["lidar_centroid_span"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
