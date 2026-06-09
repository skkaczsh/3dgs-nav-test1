#!/usr/bin/env python3
"""Assign small residual points to stable surface objects for QA.

This is the mutating/output version of analyze_residual_absorbability.py. It
does not rewrite target/object artifacts. It writes a residual-only PLY with
original semantic, assigned semantic, assignment status, visual RGB, and the
matched surface object index.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_residual_absorbability import (
    LABEL_COLORS,
    SEMANTIC_IDS,
    SEMANTIC_NAMES,
    bbox_distance,
    build_index,
    cell_key,
    label_compatible,
    load_surface_objects,
    plane_distance,
    read_ascii_ply,
)


def best_surface_match(
    point: np.ndarray,
    color: np.ndarray,
    residual_label: str,
    objects: list[dict],
    candidate_ids: list[int],
    args: argparse.Namespace,
) -> tuple[int, dict] | None:
    best = None
    best_meta = None
    for object_idx in candidate_ids:
        obj = objects[object_idx]
        object_label = obj.get("semantic_label", "unknown")
        if not label_compatible(residual_label, object_label):
            continue
        bd = bbox_distance(point, obj["bbox_3d"])
        if bd > args.bbox_padding:
            continue
        pd = plane_distance(point, obj)
        if pd > args.max_plane_distance:
            continue
        cd = float(np.linalg.norm(color - np.array(obj.get("mean_color", [0, 0, 0]), dtype=np.float32)))
        if cd > args.max_color_distance:
            continue
        score = bd + pd + cd / 255.0
        if best is None or score < best:
            best = score
            best_meta = {
                "object_index": object_idx,
                "object_id": obj["object_id"],
                "object_label": object_label,
                "bbox_distance": bd,
                "plane_distance": pd,
                "color_distance": cd,
            }
    if best_meta is None:
        return None
    return int(best_meta["object_index"]), best_meta


def write_assignment_ply(path: Path, rows: list[np.ndarray]) -> None:
    data = np.concatenate(rows, axis=0) if rows else np.empty((0, 13), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(data)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar original_semantic\n")
        f.write("property uchar assigned_semantic\n")
        f.write("property uchar assignment_status\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("property int object_index\n")
        f.write("end_header\n")
        for row in data:
            f.write(
                f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} "
                f"{int(row[3])} {int(row[4])} {int(row[5])} "
                f"{int(row[6])} {int(row[7])} {int(row[8])} "
                f"{int(row[9])} {int(row[10])} {int(row[11])} {int(row[12])}\n"
            )


def process(args: argparse.Namespace) -> dict:
    objects = load_surface_objects(args.objects_jsonl, args.min_object_targets, args.min_object_points)
    index = build_index(objects, args.cell_size, args.bbox_padding)
    files = sorted(args.residual_dir.glob("residuals_frame_*.ply"))
    if args.limit_frames:
        files = files[: args.limit_frames]

    total = 0
    assigned = 0
    by_label = Counter()
    assigned_by_label = Counter()
    assigned_object_counts = Counter()
    frame_rows = []
    ply_rows = []

    for path in files:
        props, data = read_ascii_ply(path)
        frame_id = int(path.stem.rsplit("_", 1)[1])
        if len(data) == 0:
            frame_rows.append({"frame": frame_id, "points": 0, "assigned_points": 0})
            continue
        idx = {name: i for i, name in enumerate(props)}
        points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
        colors = data[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.float32)
        labels = data[:, idx["semantic"]].astype(np.int32)
        assigned_labels = labels.copy()
        status = np.zeros(len(points), dtype=np.uint8)
        object_indices = np.full(len(points), -1, dtype=np.int32)

        for i, (point, color, sem) in enumerate(zip(points, colors, labels)):
            total += 1
            residual_label = SEMANTIC_NAMES.get(int(sem), "unknown")
            by_label[residual_label] += 1
            match = best_surface_match(
                point,
                color,
                residual_label,
                objects,
                index.get(cell_key(point, args.cell_size), []),
                args,
            )
            if match is None:
                continue
            object_idx, meta = match
            assigned += 1
            assigned_by_label[residual_label] += 1
            assigned_object_counts[meta["object_id"]] += 1
            assigned_labels[i] = SEMANTIC_IDS.get(meta["object_label"], int(sem))
            status[i] = 1
            object_indices[i] = object_idx

        rgb = np.array([LABEL_COLORS.get(int(x), LABEL_COLORS[0]) for x in assigned_labels], dtype=np.uint8)
        if args.write_ply:
            ply_rows.append(
                np.column_stack(
                    [
                        points,
                        rgb.astype(np.float32),
                        labels.astype(np.float32),
                        assigned_labels.astype(np.float32),
                        status.astype(np.float32),
                        colors.astype(np.float32),
                        object_indices.astype(np.float32),
                    ]
                )
            )
        frame_assigned = int(status.sum())
        frame_rows.append(
            {
                "frame": frame_id,
                "points": int(len(points)),
                "assigned_points": frame_assigned,
                "assigned_ratio": float(frame_assigned / max(len(points), 1)),
            }
        )

    if args.write_ply:
        write_assignment_ply(args.output_ply, ply_rows)

    return {
        "residual_dir": str(args.residual_dir),
        "objects_jsonl": str(args.objects_jsonl),
        "output_ply": str(args.output_ply) if args.write_ply else "",
        "surface_objects": len(objects),
        "residual_points": int(total),
        "assigned_points": int(assigned),
        "assigned_ratio": float(assigned / max(total, 1)),
        "by_label": dict(by_label),
        "assigned_by_label": dict(assigned_by_label),
        "top_assigned_objects": assigned_object_counts.most_common(30),
        "params": {
            "min_object_targets": args.min_object_targets,
            "min_object_points": args.min_object_points,
            "cell_size": args.cell_size,
            "bbox_padding": args.bbox_padding,
            "max_plane_distance": args.max_plane_distance,
            "max_color_distance": args.max_color_distance,
        },
        "frames": frame_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--min-object-targets", type=int, default=5)
    parser.add_argument("--min-object-points", type=int, default=1000)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--bbox-padding", type=float, default=0.35)
    parser.add_argument("--max-plane-distance", type=float, default=0.12)
    parser.add_argument("--max-color-distance", type=float, default=70.0)
    parser.add_argument("--limit-frames", type=int, default=None)
    args = parser.parse_args()

    report = process(args)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["surface_objects", "residual_points", "assigned_points", "assigned_ratio"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
