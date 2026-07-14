#!/usr/bin/env python3
"""Attach source-frame support to immutable official Superpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

try:
    from scripts.build_raw_lx_voxel_cloud import read_lx_points, read_lx_sections
except ModuleNotFoundError:  # Supports direct `python scripts/...` execution.
    from build_raw_lx_voxel_cloud import read_lx_points, read_lx_sections


def read_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"].data
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)


def update_top_frames(top_frames: np.ndarray, top_hits: np.ndarray, label: int, frame: int, hits: int) -> None:
    slot = int(np.argmin(top_hits[label]))
    if hits > int(top_hits[label, slot]):
        top_frames[label, slot] = frame
        top_hits[label, slot] = hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--lx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-distance", type=float, default=0.05)
    parser.add_argument("--top-frames", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    xyz = read_xyz(args.reference_ply)
    labels = np.load(args.labels).astype(np.int32, copy=False)
    if len(xyz) != len(labels):
        raise SystemExit(f"reference/label count mismatch: {len(xyz)} != {len(labels)}")
    label_count = int(labels.max()) + 1 if len(labels) else 0
    tree = cKDTree(xyz, compact_nodes=False, balanced_tree=False)
    sections = read_lx_sections(args.lx)
    end = min(len(sections) - 1, args.end if args.end is not None else len(sections) - 1)

    frame_min = np.full(label_count, -1, dtype=np.int32)
    frame_max = np.full(label_count, -1, dtype=np.int32)
    frame_count = np.zeros(label_count, dtype=np.int32)
    matched_hits = np.zeros(label_count, dtype=np.int64)
    top_frames = np.full((label_count, args.top_frames), -1, dtype=np.int32)
    top_hits = np.zeros((label_count, args.top_frames), dtype=np.int32)
    raw_points = matched_points = 0
    histogram = np.zeros(6, dtype=np.int64)
    bins = np.array([0.01, 0.03, 0.05, 0.10, 0.20], dtype=np.float32)

    with args.lx.open("rb") as handle:
        for frame in range(args.start, end + 1):
            points = read_lx_points(handle, sections[frame])
            if not len(points):
                continue
            distances, indices = tree.query(points, k=1, workers=-1)
            raw_points += len(points)
            histogram += np.histogram(distances, bins=np.r_[0.0, bins, np.inf])[0]
            keep = distances <= args.max_distance
            if not np.any(keep):
                continue
            matched_points += int(keep.sum())
            present, counts = np.unique(labels[indices[keep]], return_counts=True)
            frame_min[present] = np.where(frame_min[present] < 0, frame, np.minimum(frame_min[present], frame))
            frame_max[present] = np.maximum(frame_max[present], frame)
            frame_count[present] += 1
            matched_hits[present] += counts
            for label, hits in zip(present.tolist(), counts.tolist()):
                update_top_frames(top_frames, top_hits, int(label), frame, int(hits))
            if frame == args.start or frame % args.progress_every == 0:
                print(f"frame={frame} raw={raw_points} matched={matched_points}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for label in np.flatnonzero(matched_hits):
            order = np.argsort(top_hits[label])[::-1]
            top = [
                {"frame_id": int(top_frames[label, slot]), "matched_points": int(top_hits[label, slot])}
                for slot in order
                if top_frames[label, slot] >= 0 and top_hits[label, slot] > 0
            ]
            f.write(json.dumps({
                "object_id": int(label),
                "source_frame_min": int(frame_min[label]),
                "source_frame_max": int(frame_max[label]),
                "source_frame_count": int(frame_count[label]),
                "matched_lx_points": int(matched_hits[label]),
                "top_source_frames": top,
            }, ensure_ascii=False) + "\n")
    report = {
        "reference_ply": str(args.reference_ply),
        "labels": str(args.labels),
        "lx": str(args.lx),
        "frame_range": [args.start, end],
        "max_distance": args.max_distance,
        "raw_lx_points": raw_points,
        "matched_lx_points": matched_points,
        "matched_ratio": matched_points / max(raw_points, 1),
        "superpoints_with_source_support": int(np.count_nonzero(matched_hits)),
        "distance_histogram": {"bins": [0.01, 0.03, 0.05, 0.10, 0.20, "inf"], "counts": histogram.tolist()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
