#!/usr/bin/env python3
"""Apply hard geometry guards to semantic object labels.

This pass is a safety layer after teacher transfer/coarsening.  It does not
change object ownership.  It only demotes labels that are geometrically unsafe
to show as final semantics, e.g. tiny floor fragments, horizontal wall labels,
or car labels on surface-like objects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_frame_target_objects_for_viewer import LABEL_TO_SEMANTIC, SEMANTIC_COLORS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def object_key(row: dict[str, Any]) -> int:
    return int(row.get("object_id") or row.get("viewer_object_id"))


def extent(row: dict[str, Any]) -> list[float]:
    bbox = row.get("bbox_3d") or {}
    bmin = bbox.get("min") or [0.0, 0.0, 0.0]
    bmax = bbox.get("max") or [0.0, 0.0, 0.0]
    return [float(bmax[i]) - float(bmin[i]) for i in range(3)]


def max_horizontal_extent(row: dict[str, Any]) -> float:
    ex = extent(row)
    return max(float(ex[0]), float(ex[1]))


def z_extent(row: dict[str, Any]) -> float:
    return float(extent(row)[2])


def voxel_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("voxel_count") or row.get("point_count") or 0)
    except (TypeError, ValueError):
        return 0


def normal_abs_z(row: dict[str, Any]) -> float:
    n = row.get("mean_normal") or [0.0, 0.0, 0.0]
    try:
        norm = math.sqrt(float(n[0]) ** 2 + float(n[1]) ** 2 + float(n[2]) ** 2)
        return abs(float(n[2])) / norm if norm > 1e-9 else 0.0
    except (TypeError, ValueError, IndexError):
        return 0.0


def geometry_type(row: dict[str, Any]) -> str:
    return str(row.get("geometry_type") or row.get("object_type_geometry") or "unknown")


def teacher_confidence(row: dict[str, Any]) -> float:
    try:
        return float(row.get("teacher_semantic_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def has_teacher_transfer(row: dict[str, Any]) -> bool:
    status_votes = row.get("teacher_status_votes") or {}
    if isinstance(status_votes, dict) and float(status_votes.get("teacher_semantic_transfer", 0.0) or 0.0) > 0:
        return True
    return str(row.get("semantic_transfer_status") or "") == "teacher_semantic_transfer"


def choose_label(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    label = str(row.get("semantic_label") or "unknown")
    geom = geometry_type(row)
    count = voxel_count(row)
    h_extent = max_horizontal_extent(row)
    z_ext = z_extent(row)
    nz = normal_abs_z(row)
    conf = teacher_confidence(row)
    teacher_ok = has_teacher_transfer(row) and conf >= args.teacher_confidence_keep

    if label == "floor":
        if geom != "horizontal":
            return "unknown", "floor_geometry_not_horizontal"
        broad = count >= args.floor_min_voxels or h_extent >= args.floor_min_extent
        if not broad and not teacher_ok:
            return "unknown", "floor_fragment_too_small_without_teacher"
        if z_ext > args.floor_max_z_extent and not teacher_ok:
            return "unknown", "floor_fragment_too_thick"
        return label, "kept_floor_geometry_guard"

    if label == "wall":
        if geom == "horizontal" or nz >= args.wall_max_normal_abs_z:
            if (
                args.allow_wall_to_floor
                and count >= args.floor_min_voxels
                and h_extent >= args.floor_min_extent
            ):
                return "floor", "wall_horizontal_large_surface_to_floor"
            return "unknown", "wall_horizontal_or_up_normal"
        if geom not in {"vertical", "mixed", "rough_mixed", "unknown"}:
            return "unknown", "wall_geometry_veto"
        if count < args.wall_min_voxels and not teacher_ok:
            return "unknown", "wall_fragment_too_small_without_teacher"
        return label, "kept_wall_geometry_guard"

    if label == "car":
        if geom == "horizontal" and nz >= args.car_surface_normal_abs_z and z_ext <= args.car_surface_max_z_extent:
            return "unknown", "car_surface_like_veto"
        if count < args.car_min_voxels and not teacher_ok:
            return "unknown", "car_fragment_too_small_without_teacher"
        return label, "kept_car_geometry_guard"

    if label == "railing":
        if geom == "horizontal" and nz >= args.railing_surface_normal_abs_z:
            return "unknown", "railing_surface_like_veto"
        return label, "kept_railing_geometry_guard"

    return label, "kept_unchecked_label"


def parse_ply_header(path: Path) -> tuple[list[str], list[str], int]:
    header: list[str] = []
    props: list[str] = []
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            stripped = line.strip()
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            if stripped == "end_header":
                break
    return header, props, len(header)


def rewrite_ply(source_ply: Path, output_ply: Path, labels: dict[int, str]) -> dict[str, Any]:
    header, props, header_lines = parse_ply_header(source_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"red", "green", "blue", "object", "semantic"}
    missing = required - set(idx)
    if missing:
        raise ValueError(f"PLY missing required fields: {sorted(missing)}")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    label_points = Counter()
    with source_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in header:
            dst.write(line)
        for _ in range(header_lines):
            next(src)
        for line in src:
            if not line.strip():
                continue
            parts = line.strip().split()
            oid = int(float(parts[idx["object"]]))
            label = labels.get(oid, "unknown")
            semantic = int(LABEL_TO_SEMANTIC.get(label, 0))
            color = SEMANTIC_COLORS.get(semantic, SEMANTIC_COLORS[0])
            parts[idx["red"]] = str(color[0])
            parts[idx["green"]] = str(color[1])
            parts[idx["blue"]] = str(color[2])
            parts[idx["semantic"]] = str(semantic)
            dst.write(" ".join(parts) + "\n")
            rows += 1
            label_points[label] += 1
    return {"rows": rows, "label_point_counts": dict(label_points)}


def apply_guard(args: argparse.Namespace) -> dict[str, Any]:
    objects = read_jsonl(args.input_objects_jsonl)
    output_objects: list[dict[str, Any]] = []
    labels: dict[int, str] = {}
    reason_counts = Counter()
    label_counts = Counter()
    changed = 0
    for row in objects:
        oid = object_key(row)
        old = str(row.get("semantic_label") or "unknown")
        new, reason = choose_label(row, args)
        out = dict(row)
        out["semantic_label_before_geometry_guard"] = old
        out["semantic_label"] = new
        out["semantic_id"] = int(LABEL_TO_SEMANTIC.get(new, 0))
        out["semantic_geometry_guard_status"] = reason
        if new != old:
            changed += 1
        labels[oid] = new
        reason_counts[reason] += 1
        label_counts[new] += 1
        output_objects.append(out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    out_ply = args.output_dir / f"{args.output_prefix}.ply"
    report_json = args.output_dir / f"{args.output_prefix}_report.json"
    write_jsonl(out_jsonl, output_objects)
    ply_report = rewrite_ply(args.input_ply, out_ply, labels)
    report = {
        "input_ply": str(args.input_ply),
        "input_objects_jsonl": str(args.input_objects_jsonl),
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "object_count": len(output_objects),
        "changed_object_count": changed,
        "reason_counts": dict(reason_counts),
        "label_object_counts": dict(label_counts),
        **ply_report,
        "params": vars(args),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--input-objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="semantic_geometry_guarded")
    parser.add_argument("--teacher-confidence-keep", type=float, default=0.65)
    parser.add_argument("--floor-min-voxels", type=int, default=1200)
    parser.add_argument("--floor-min-extent", type=float, default=2.5)
    parser.add_argument("--floor-max-z-extent", type=float, default=0.9)
    parser.add_argument("--wall-min-voxels", type=int, default=400)
    parser.add_argument("--wall-max-normal-abs-z", type=float, default=0.62)
    parser.add_argument(
        "--allow-wall-to-floor",
        action="store_true",
        help="Allow horizontal wall conflicts to become floor. Disabled by default because mixed Patch normals can be unreliable.",
    )
    parser.add_argument("--car-min-voxels", type=int, default=120)
    parser.add_argument("--car-surface-normal-abs-z", type=float, default=0.88)
    parser.add_argument("--car-surface-max-z-extent", type=float, default=0.35)
    parser.add_argument("--railing-surface-normal-abs-z", type=float, default=0.88)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(apply_guard(parse_args()), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
