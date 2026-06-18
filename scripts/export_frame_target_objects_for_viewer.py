#!/usr/bin/env python3
"""Export frame-local target fusion outputs as viewer-ready object point data.

Inputs:
  - frame_targets.jsonl from build_frame_targets_from_priority.py
  - frame_targets.ply with per-point target index
  - objects.jsonl from fuse_targets_to_objects.py

Outputs:
  - ASCII PLY with x y z RGB object semantic frame camera target priority
  - JSONL object metadata that can be loaded by tools/semantic_ply_viewer.html

The script streams the PLY body and does not load all points into memory.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_TO_SEMANTIC = {
    "unknown": 0,
    "other": 1,
    "wall": 2,
    "floor": 3,
    "ground": 3,
    "ceiling": 4,
    "grass": 5,
    "tree": 6,
    "person": 7,
    "car": 8,
    "railing": 9,
    "building": 10,
    "sky": 11,
    "road": 12,
    "water": 13,
    "furniture": 14,
    "pipe": 15,
    "equipment": 16,
    "fine_candidate": 17,
    "ambiguous": 0,
    "ignore": 255,
}

SEMANTIC_COLORS = {
    0: (150, 150, 150),
    1: (180, 180, 180),
    2: (120, 150, 180),
    3: (196, 168, 112),
    4: (170, 170, 210),
    5: (80, 160, 80),
    6: (50, 130, 70),
    8: (235, 90, 80),
    9: (240, 210, 60),
    10: (145, 145, 160),
    12: (120, 120, 120),
    15: (220, 160, 60),
    16: (210, 90, 210),
    17: (245, 150, 40),
    255: (40, 40, 40),
}


def object_number(object_id: str) -> int:
    match = re.search(r"(\d+)$", str(object_id))
    if not match:
        raise ValueError(f"Object id has no numeric suffix: {object_id}")
    return int(match.group(1))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_target_index_map(targets_jsonl: Path) -> dict[int, str]:
    out = {}
    for row in read_jsonl(targets_jsonl):
        out[int(row["target_index"])] = str(row["target_id"])
    return out


def load_object_maps(objects_jsonl: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    objects_by_id = {}
    target_to_object = {}
    for row in read_jsonl(objects_jsonl):
        oid = str(row["object_id"])
        objects_by_id[oid] = row
        number = object_number(oid)
        for target_id in row.get("targets", []):
            target_to_object[str(target_id)] = number
    return objects_by_id, target_to_object


def read_ply_header(path: Path) -> tuple[list[str], list[str], int, int]:
    header: list[str] = []
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    header_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            header.append(line)
            if stripped == "end_header":
                break
    if vertex_count <= 0:
        raise ValueError(f"No vertex count found: {path}")
    return header, props, vertex_count, header_lines


def write_header(path: Path, vertex_count: int) -> None:
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int object\n"
        "property uchar semantic\n"
        "property int frame\n"
        "property int camera\n"
        "property int target\n"
        "property uchar priority\n"
        "end_header\n"
    )
    path.write_text(header, encoding="utf-8")


HEAVY_OBJECT_FIELDS = {
    "merged_point_indices",
    "point_indices",
    "_target_records",
    "_point_id_set",
    "color_sum",
}


def slim_object_row(row: dict[str, Any], keep_targets: bool) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in HEAVY_OBJECT_FIELDS}
    if not keep_targets:
        out.pop("targets", None)
    return out


def export_objects_jsonl(objects_by_id: dict[str, dict[str, Any]], output: Path, keep_targets: bool = True) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for oid in sorted(objects_by_id, key=object_number):
            row = slim_object_row(objects_by_id[oid], keep_targets=keep_targets)
            label = str(row.get("semantic_label") or "unknown")
            votes = row.get("label_vote_weights") or row.get("label_votes") or {}
            if votes:
                total = sum(float(v) for v in votes.values())
                best = max(votes.items(), key=lambda kv: float(kv[1]))
                row["dominant_label_ratio"] = float(float(best[1]) / max(total, 1.0))
            row["semantic_id"] = LABEL_TO_SEMANTIC.get(label, 0)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_ply(
    target_ply: Path,
    output_ply: Path,
    target_index_to_id: dict[int, str],
    target_to_object: dict[str, int],
    objects_by_id: dict[str, dict[str, Any]],
    stride: int,
) -> dict[str, Any]:
    _header, props, vertex_count, header_lines = read_ply_header(target_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"x", "y", "z", "target", "priority", "frame", "camera"}
    if not required.issubset(idx):
        raise ValueError(f"PLY missing required fields: {sorted(required - set(idx))}")

    object_label = {
        object_number(oid): str(row.get("semantic_label") or "unknown")
        for oid, row in objects_by_id.items()
    }
    kept = (vertex_count + stride - 1) // stride
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_header(output_ply, kept)
    label_counts = Counter()
    object_counts = Counter()
    missing_target = 0
    written = 0
    with target_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("a", encoding="utf-8") as dst:
        for _ in range(header_lines):
            next(src)
        for i, line in enumerate(src):
            if i % stride != 0:
                continue
            parts = line.strip().split()
            if len(parts) < len(props):
                continue
            target_index = int(round(float(parts[idx["target"]])))
            target_id = target_index_to_id.get(target_index, "")
            object_id = target_to_object.get(target_id, 0)
            if not object_id:
                missing_target += 1
            label = object_label.get(object_id, "unknown")
            semantic = LABEL_TO_SEMANTIC.get(label, 0)
            color = SEMANTIC_COLORS.get(semantic, SEMANTIC_COLORS[0])
            label_counts[label] += 1
            object_counts[object_id] += 1
            dst.write(
                f"{parts[idx['x']]} {parts[idx['y']]} {parts[idx['z']]} "
                f"{color[0]} {color[1]} {color[2]} {object_id} {semantic} "
                f"{int(round(float(parts[idx['frame']]))) } {int(round(float(parts[idx['camera']]))) } "
                f"{target_index} {int(round(float(parts[idx['priority']]))) }\n"
            )
            written += 1
    if written != kept:
        raise RuntimeError(f"stride count mismatch: expected={kept} written={written}")
    return {
        "input_vertices": vertex_count,
        "output_vertices": written,
        "stride": stride,
        "missing_target_points": missing_target,
        "label_counts": dict(label_counts),
        "object_count_with_points": sum(1 for oid in object_counts if oid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--target-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--ply-name", default="frame_object_points_stride10.ply")
    parser.add_argument("--objects-name", default="frame_objects_viewer.jsonl")
    parser.add_argument("--keep-target-list", action="store_true",
                        help="Keep object.targets in viewer JSONL. Heavy point index fields are always removed.")
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    target_index_to_id = load_target_index_map(args.targets_jsonl)
    objects_by_id, target_to_object = load_object_maps(args.objects_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = args.output_dir / args.ply_name
    output_objects = args.output_dir / args.objects_name
    report_path = args.output_dir / "frame_object_viewer_export_report.json"
    report = export_ply(
        args.target_ply,
        output_ply,
        target_index_to_id,
        target_to_object,
        objects_by_id,
        args.stride,
    )
    export_objects_jsonl(objects_by_id, output_objects, keep_targets=args.keep_target_list)
    report.update({
        "targets_jsonl": str(args.targets_jsonl),
        "target_ply": str(args.target_ply),
        "objects_jsonl": str(args.objects_jsonl),
        "output_ply": str(output_ply),
        "output_objects_jsonl": str(output_objects),
        "object_records": len(objects_by_id),
        "target_records": len(target_index_to_id),
    })
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
