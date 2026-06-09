#!/usr/bin/env python3
"""Sweep residual-to-surface absorption parameters without writing PLY files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

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


def parse_config(text: str) -> dict:
    values = {}
    for part in text.split(","):
        key, raw = part.split("=", 1)
        values[key.strip()] = float(raw)
    return {
        "name": text,
        "bbox_padding": values["bbox"],
        "max_plane_distance": values["plane"],
        "max_color_distance": values["color"],
    }


def analyze(args: argparse.Namespace) -> dict:
    configs = [parse_config(text) for text in args.config]
    max_bbox = max(c["bbox_padding"] for c in configs)
    objects = load_surface_objects(args.objects_jsonl, args.min_object_targets, args.min_object_points)
    index = build_index(objects, args.cell_size, max_bbox)

    totals = {
        cfg["name"]: {
            "assigned_points": 0,
            "assigned_by_label": Counter(),
            "assigned_to_label": Counter(),
        }
        for cfg in configs
    }
    total_points = 0
    by_label = Counter()

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
            total_points += 1
            residual_label = SEMANTIC_NAMES.get(int(sem), "unknown")
            by_label[residual_label] += 1
            candidate_ids = index.get(cell_key(point, args.cell_size), [])
            if not candidate_ids:
                continue

            candidate_metrics = []
            for object_idx in candidate_ids:
                obj = objects[object_idx]
                object_label = obj.get("semantic_label", "unknown")
                if not label_compatible(residual_label, object_label):
                    continue
                bd = bbox_distance(point, obj["bbox_3d"])
                if bd > max_bbox:
                    continue
                pd = plane_distance(point, obj)
                cd = float(np.linalg.norm(color - np.array(obj.get("mean_color", [0, 0, 0]), dtype=np.float32)))
                candidate_metrics.append((bd, pd, cd, object_label))
            if not candidate_metrics:
                continue

            for cfg in configs:
                best = None
                for bd, pd, cd, object_label in candidate_metrics:
                    if bd > cfg["bbox_padding"] or pd > cfg["max_plane_distance"] or cd > cfg["max_color_distance"]:
                        continue
                    score = bd + pd + cd / 255.0
                    if best is None or score < best[0]:
                        best = (score, object_label)
                if best is None:
                    continue
                totals[cfg["name"]]["assigned_points"] += 1
                totals[cfg["name"]]["assigned_by_label"][residual_label] += 1
                totals[cfg["name"]]["assigned_to_label"][best[1]] += 1

    rows = []
    for cfg in configs:
        row = {
            **cfg,
            "residual_points": int(total_points),
            "assigned_points": int(totals[cfg["name"]]["assigned_points"]),
            "assigned_ratio": float(totals[cfg["name"]]["assigned_points"] / max(total_points, 1)),
            "assigned_by_label": dict(totals[cfg["name"]]["assigned_by_label"]),
            "assigned_to_label": dict(totals[cfg["name"]]["assigned_to_label"]),
            "unassigned_by_label": {
                label: int(count - totals[cfg["name"]]["assigned_by_label"].get(label, 0))
                for label, count in by_label.items()
            },
        }
        rows.append(row)
    return {
        "residual_dir": str(args.residual_dir),
        "objects_jsonl": str(args.objects_jsonl),
        "surface_objects": len(objects),
        "by_label": dict(by_label),
        "configs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", action="append", required=True, help="bbox=<m>,plane=<m>,color=<rgb_dist>")
    parser.add_argument("--min-object-targets", type=int, default=5)
    parser.add_argument("--min-object-points", type=int, default=1000)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--limit-frames", type=int, default=None)
    args = parser.parse_args()

    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        [
            {
                "name": row["name"],
                "assigned_ratio": row["assigned_ratio"],
                "assigned_points": row["assigned_points"],
                "top_unassigned": sorted(row["unassigned_by_label"].items(), key=lambda kv: kv[1], reverse=True)[:5],
            }
            for row in report["configs"]
        ],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
