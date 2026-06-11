#!/usr/bin/env python3
"""Sweep surface-object coverage for residual points.

This diagnoses the largest current miss bucket: residual points that have no
stable surface object in the spatial index cell. It does not assign labels or
write point clouds. It only asks: if the candidate search radius grows, do the
points become plausible surface matches, or do they still fail geometry/color?
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


SURFACE_RESIDUAL_LABELS = {"floor", "building", "wall", "road"}


def parse_config(text: str) -> dict[str, float | str]:
    values = {}
    for part in text.split(","):
        key, raw = part.split("=", 1)
        values[key.strip()] = float(raw)
    return {
        "name": text,
        "index_padding": values["pad"],
        "bbox_padding": values["bbox"],
        "max_plane_distance": values["plane"],
        "max_color_distance": values["color"],
    }


def classify_against_candidates(
    point: np.ndarray,
    color: np.ndarray,
    residual_label: str,
    objects: list[dict[str, Any]],
    candidate_ids: list[int],
    cfg: dict[str, float | str],
) -> str:
    if not candidate_ids:
        return "no_candidate_cell"

    label_candidates = []
    for object_idx in candidate_ids:
        obj = objects[object_idx]
        if label_compatible(residual_label, obj.get("semantic_label", "unknown")):
            label_candidates.append(obj)
    if not label_candidates:
        return "label_incompatible"

    bbox_candidates = []
    for obj in label_candidates:
        if bbox_distance(point, obj["bbox_3d"]) <= float(cfg["bbox_padding"]):
            bbox_candidates.append(obj)
    if not bbox_candidates:
        return "bbox_distance_failed"

    plane_candidates = []
    for obj in bbox_candidates:
        if plane_distance(point, obj) <= float(cfg["max_plane_distance"]):
            plane_candidates.append(obj)
    if not plane_candidates:
        return "plane_distance_failed"

    for obj in plane_candidates:
        dist = float(np.linalg.norm(color - np.array(obj.get("mean_color", [0, 0, 0]), dtype=np.float32)))
        if dist <= float(cfg["max_color_distance"]):
            return "matched_surface"
    return "color_distance_failed"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    configs = [parse_config(text) for text in args.config]
    objects = load_surface_objects(args.objects_jsonl, args.min_object_targets, args.min_object_points)
    indexes = {
        cfg["name"]: build_index(objects, args.cell_size, float(cfg["index_padding"]))
        for cfg in configs
    }
    totals = {
        cfg["name"]: {
            "reason_counts": Counter(),
            "reason_by_label": defaultdict(Counter),
        }
        for cfg in configs
    }
    label_counts = Counter()
    total_points = 0
    files = sorted(args.residual_dir.glob("residuals_frame_*.ply"))
    if args.limit_frames:
        files = files[: args.limit_frames]

    for path in files:
        props, data = read_ascii_ply(path)
        if len(data) == 0:
            continue
        idx = {name: i for i, name in enumerate(props)}
        points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
        colors = data[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.float32)
        labels = data[:, idx["semantic"]].astype(np.int32)
        for point, color, sem in zip(points, colors, labels):
            residual_label = SEMANTIC_NAMES.get(int(sem), "unknown")
            if args.surface_labels_only and residual_label not in SURFACE_RESIDUAL_LABELS:
                continue
            total_points += 1
            label_counts[residual_label] += 1
            for cfg in configs:
                reason = classify_against_candidates(
                    point,
                    color,
                    residual_label,
                    objects,
                    indexes[cfg["name"]].get(cell_key(point, args.cell_size), []),
                    cfg,
                )
                totals[cfg["name"]]["reason_counts"][reason] += 1
                totals[cfg["name"]]["reason_by_label"][residual_label][reason] += 1

    rows = []
    for cfg in configs:
        reason_counts = dict(totals[cfg["name"]]["reason_counts"])
        matched = reason_counts.get("matched_surface", 0)
        rows.append(
            {
                **cfg,
                "residual_points": int(total_points),
                "matched_surface_points": int(matched),
                "matched_surface_ratio": float(matched / max(total_points, 1)),
                "reason_counts": reason_counts,
                "reason_by_label": {
                    label: dict(counts)
                    for label, counts in totals[cfg["name"]]["reason_by_label"].items()
                },
            }
        )
    return {
        "residual_dir": str(args.residual_dir),
        "objects_jsonl": str(args.objects_jsonl),
        "surface_objects": len(objects),
        "surface_labels_only": args.surface_labels_only,
        "label_counts": dict(label_counts),
        "configs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", action="append", required=True, help="pad=<m>,bbox=<m>,plane=<m>,color=<rgb_dist>")
    parser.add_argument("--min-object-targets", type=int, default=5)
    parser.add_argument("--min-object-points", type=int, default=1000)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--surface-labels-only", action="store_true")
    args = parser.parse_args()

    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        [
            {
                "name": row["name"],
                "matched_surface_ratio": row["matched_surface_ratio"],
                "reason_counts": row["reason_counts"],
            }
            for row in report["configs"]
        ],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
