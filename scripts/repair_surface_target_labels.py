#!/usr/bin/env python3
"""Repair obvious surface target label conflicts before object fusion.

This is a JSONL-only post-process for frame-local Targets. It is intentionally
small in scope: relabel only geometry-obvious surface contradictions, keep
provenance fields, and leave ambiguous large mixed surfaces for a later plane
split stage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PARENT_BY_LABEL = {
    "ground": "surface",
    "wall": "surface",
    "building": "surface",
    "floor": "surface",
    "ceiling": "surface",
    "grass": "vegetation",
    "car": "object",
    "railing": "structure",
    "other": "other",
    "unknown": "other",
}
PRIORITY_ID_BY_LABEL = {
    "ground": 1,
    "wall": 2,
    "grass": 3,
    "car": 4,
    "railing": 5,
    "ceiling": 6,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def bbox_dims(row: dict[str, Any]) -> np.ndarray:
    bbox = row.get("bbox_3d") or {}
    lo = np.array(bbox.get("min", [0.0, 0.0, 0.0]), dtype=np.float64)
    hi = np.array(bbox.get("max", [0.0, 0.0, 0.0]), dtype=np.float64)
    return hi - lo


def centroid_z(row: dict[str, Any]) -> float:
    try:
        return float((row.get("centroid") or [0.0, 0.0, 0.0])[2])
    except (TypeError, ValueError, IndexError):
        bbox = row.get("bbox_3d") or {}
        lo = bbox.get("min", [0.0, 0.0, 0.0])
        hi = bbox.get("max", [0.0, 0.0, 0.0])
        return float((float(lo[2]) + float(hi[2])) * 0.5)


def normal_z(row: dict[str, Any]) -> float:
    normal = (row.get("pca") or {}).get("normal") or [0.0, 0.0, 1.0]
    try:
        return abs(float(normal[2]))
    except (TypeError, ValueError, IndexError):
        return 1.0


def pca_value(row: dict[str, Any], key: str) -> float:
    try:
        return float((row.get("pca") or {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def horizontal_extent(row: dict[str, Any]) -> float:
    d = bbox_dims(row)
    return float(np.hypot(max(d[0], 0.0), max(d[1], 0.0)))


def choose_repair(row: dict[str, Any], args: argparse.Namespace) -> tuple[str | None, str]:
    label = str(row.get("label") or "unknown")
    d = bbox_dims(row)
    nz = normal_z(row)
    planarity = pca_value(row, "planarity")
    linearity = pca_value(row, "linearity")
    z_span = float(max(d[2], 0.0))
    h_extent = horizontal_extent(row)
    cz = centroid_z(row)

    if label == "wall":
        horizontal_thin = (
            nz >= args.horizontal_surface_normal_z
            and z_span <= args.horizontal_surface_max_z_span
            and h_extent >= args.horizontal_surface_min_extent
            and planarity >= args.horizontal_surface_min_planarity
        )
        if horizontal_thin:
            new_label = "ceiling" if cz >= args.ceiling_min_z else "ground"
            return new_label, f"wall_horizontal_thin_to_{new_label}"

    if label == "ground":
        vertical_planar = (
            nz <= args.ground_to_wall_max_normal_z
            and z_span >= args.ground_to_wall_min_z_span
            and planarity >= args.ground_to_wall_min_planarity
        )
        if vertical_planar:
            return "wall", "ground_vertical_planar_to_wall"

    if label == "ceiling":
        if nz <= args.ceiling_to_wall_max_normal_z and z_span >= args.ceiling_to_wall_min_z_span:
            return "wall", "ceiling_vertical_to_wall"

    if label in {"car", "railing"}:
        flat_surface_like = (
            nz >= args.fine_to_unknown_normal_z
            and z_span <= args.fine_to_unknown_max_z_span
            and planarity >= args.fine_to_unknown_min_planarity
            and linearity < args.fine_to_unknown_max_linearity
        )
        if flat_surface_like:
            return "unknown", f"{label}_flat_surface_like_to_unknown"

    return None, "no_repair"


def repair_targets(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    flow = Counter()
    reasons = Counter()
    examples = []
    for row in rows:
        old_label = str(row.get("label") or "unknown")
        new_label, reason = choose_repair(row, args)
        if not new_label or new_label == old_label:
            out_rows.append(row)
            continue
        out = dict(row)
        out["label"] = new_label
        out["raw_label"] = str(row.get("raw_label") or old_label)
        out["surface_repaired_from_label"] = old_label
        out["surface_repair_reason"] = reason
        out["parent_class"] = PARENT_BY_LABEL.get(new_label, out.get("parent_class", "other"))
        out["priority_label_id"] = int(PRIORITY_ID_BY_LABEL.get(new_label, out.get("priority_label_id", 0)))
        out["mask_id"] = int(PRIORITY_ID_BY_LABEL.get(new_label, out.get("mask_id", 0)))
        out_rows.append(out)
        flow[(old_label, new_label)] += 1
        reasons[reason] += 1
        if len(examples) < args.example_limit:
            examples.append({
                "target_id": row.get("target_id"),
                "from": old_label,
                "to": new_label,
                "reason": reason,
                "frame_id": row.get("frame_id"),
                "cam_id": row.get("cam_id"),
                "cluster_size": row.get("cluster_size"),
                "normal_z": round(normal_z(row), 4),
                "dims": [round(float(x), 4) for x in bbox_dims(row).tolist()],
                "centroid_z": round(centroid_z(row), 4),
            })

    report = {
        "input_targets": len(rows),
        "output_targets": len(out_rows),
        "repaired_targets": int(sum(flow.values())),
        "label_flow_counts": {f"{src}->{dst}": count for (src, dst), count in sorted(flow.items())},
        "reason_counts": dict(reasons.most_common()),
        "input_label_counts": dict(Counter(str(row.get("label") or "unknown") for row in rows).most_common()),
        "output_label_counts": dict(Counter(str(row.get("label") or "unknown") for row in out_rows).most_common()),
        "examples": examples,
        "params": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, list, type(None)))
        },
    }
    return out_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizontal-surface-normal-z", type=float, default=0.92)
    parser.add_argument("--horizontal-surface-max-z-span", type=float, default=0.45)
    parser.add_argument("--horizontal-surface-min-extent", type=float, default=1.5)
    parser.add_argument("--horizontal-surface-min-planarity", type=float, default=0.30)
    parser.add_argument("--ceiling-min-z", type=float, default=2.2)
    parser.add_argument("--ground-to-wall-max-normal-z", type=float, default=0.45)
    parser.add_argument("--ground-to-wall-min-z-span", type=float, default=0.8)
    parser.add_argument("--ground-to-wall-min-planarity", type=float, default=0.35)
    parser.add_argument("--ceiling-to-wall-max-normal-z", type=float, default=0.45)
    parser.add_argument("--ceiling-to-wall-min-z-span", type=float, default=0.6)
    parser.add_argument("--fine-to-unknown-normal-z", type=float, default=0.92)
    parser.add_argument("--fine-to-unknown-max-z-span", type=float, default=0.25)
    parser.add_argument("--fine-to-unknown-min-planarity", type=float, default=0.35)
    parser.add_argument("--fine-to-unknown-max-linearity", type=float, default=0.82)
    parser.add_argument("--example-limit", type=int, default=50)
    args = parser.parse_args()

    rows, report = repair_targets(read_jsonl(args.targets_jsonl), args)
    write_jsonl(args.output_jsonl, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
