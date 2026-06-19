#!/usr/bin/env python3
"""Resolve surface-only ambiguous objects with conservative geometry rules."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SURFACE_LABELS = {"ground", "floor", "wall", "ceiling"}
LABEL_IDS = {
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
    "ambiguous": 0,
    "ignore": 255,
}
LABEL_COLORS = {
    "unknown": (150, 150, 150),
    "other": (180, 180, 180),
    "wall": (120, 150, 180),
    "floor": (196, 168, 112),
    "ground": (196, 168, 112),
    "ceiling": (170, 170, 210),
    "grass": (80, 160, 80),
    "car": (235, 90, 80),
    "railing": (240, 210, 60),
    "ambiguous": (230, 40, 210),
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


def votes(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("label_vote_weights") or row.get("label_votes") or {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def bbox_dims(row: dict[str, Any]) -> list[float]:
    bbox = row.get("bbox_3d") or {}
    lo = bbox.get("min", [0.0, 0.0, 0.0])
    hi = bbox.get("max", [0.0, 0.0, 0.0])
    return [float(hi[i]) - float(lo[i]) for i in range(3)]


def centroid_z(row: dict[str, Any]) -> float:
    try:
        return float((row.get("centroid") or [0.0, 0.0, 0.0])[2])
    except (TypeError, ValueError, IndexError):
        return 0.0


def normal_z(row: dict[str, Any]) -> float:
    normal = row.get("normal") or [0.0, 0.0, 1.0]
    try:
        return abs(float(normal[2]))
    except (TypeError, ValueError, IndexError):
        return 1.0


def dominant(vote_map: dict[str, float]) -> tuple[str, float]:
    total = sum(vote_map.values())
    if total <= 0:
        return "unknown", 0.0
    label, value = max(vote_map.items(), key=lambda item: item[1])
    return label, float(value / total)


def choose_label(row: dict[str, Any], args: argparse.Namespace) -> tuple[str | None, str, dict[str, Any]]:
    if str(row.get("status") or "") != "ambiguous_object" and str(row.get("semantic_label") or "") != "ambiguous":
        return None, "not_ambiguous", {}
    vote_map = votes(row)
    if not vote_map or not set(vote_map).issubset(SURFACE_LABELS):
        return None, "non_surface_votes", {"votes": vote_map}

    dom_label, dom_ratio = dominant(vote_map)
    nz = normal_z(row)
    dims = bbox_dims(row)
    z_span = max(float(dims[2]), 0.0)
    cz = centroid_z(row)
    meta = {
        "dominant_label": dom_label,
        "dominant_ratio": round(dom_ratio, 4),
        "normal_z": round(nz, 4),
        "z_span": round(z_span, 4),
        "centroid_z": round(cz, 4),
        "votes": vote_map,
    }

    if dom_ratio >= args.min_dominant_ratio:
        if dom_label in {"ground", "floor"}:
            if cz >= args.high_horizontal_ceiling_z and nz >= args.horizontal_normal_z:
                return "ceiling", "dominant_ground_high_horizontal_to_ceiling", meta
            return "ground", "dominant_ground", meta
        if dom_label == "ceiling":
            if nz >= args.ceiling_min_normal_z and cz >= args.ceiling_min_z:
                return "ceiling", "dominant_ceiling_geometry_ok", meta
            return None, "dominant_ceiling_geometry_rejected", meta
        if dom_label == "wall":
            if nz <= args.wall_max_normal_z or z_span >= args.wall_min_z_span:
                return "wall", "dominant_wall_geometry_ok", meta
            if nz >= args.horizontal_normal_z and cz >= args.high_horizontal_ceiling_z:
                return "ceiling", "dominant_wall_high_horizontal_to_ceiling", meta
            if nz >= args.horizontal_normal_z:
                return "ground", "dominant_wall_horizontal_to_ground", meta

    if nz >= args.strong_horizontal_normal_z and dom_ratio >= args.min_geometry_ratio:
        if cz >= args.high_horizontal_ceiling_z:
            return "ceiling", "strong_horizontal_high_to_ceiling", meta
        if z_span <= args.ground_max_z_span_for_geometry:
            return "ground", "strong_horizontal_low_to_ground", meta

    if nz <= args.strong_wall_normal_z and dom_ratio >= args.min_geometry_ratio:
        return "wall", "strong_vertical_to_wall", meta

    return None, "kept_ambiguous", meta


def resolve_objects(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    changes = []
    reasons = Counter()
    before = Counter(str(row.get("semantic_label") or "unknown") for row in rows)
    for row in rows:
        old_label = str(row.get("semantic_label") or "unknown")
        new_label, reason, meta = choose_label(row, args)
        reasons[reason] += 1
        if not new_label:
            out.append(row)
            continue
        obj = dict(row)
        obj["semantic_label_original"] = obj.get("semantic_label_original") or old_label
        obj["status_original"] = obj.get("status_original") or obj.get("status")
        obj["semantic_label"] = new_label
        obj["status"] = "surface_ambiguous_resolved"
        obj["ambiguous_surface_resolve_reason"] = reason
        obj["ambiguous_surface_resolve_metrics"] = meta
        out.append(obj)
        changes.append({
            "object_id": row.get("object_id"),
            "from": old_label,
            "to": new_label,
            "reason": reason,
            "point_count": row.get("point_count"),
            **meta,
        })
    after = Counter(str(row.get("semantic_label") or "unknown") for row in out)
    report = {
        "input_objects": len(rows),
        "output_objects": len(out),
        "changed_objects": len(changes),
        "label_counts_before": dict(before.most_common()),
        "label_counts_after": dict(after.most_common()),
        "reason_counts": dict(reasons.most_common()),
        "changes": changes[:200],
        "params": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, list, type(None)))
        },
    }
    return out, report


def read_ply_header(path: Path) -> tuple[list[str], list[str], int]:
    header = []
    props = []
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            stripped = line.strip()
            if stripped.startswith("format ") and "ascii" not in stripped:
                raise ValueError(f"Only ASCII PLY is supported: {path}")
            if stripped.startswith("element vertex"):
                in_vertex = True
            elif stripped.startswith("element "):
                in_vertex = False
            elif in_vertex and stripped.startswith("property "):
                props.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
    return header, props, len(header)


def rewrite_ply(input_ply: Path, output_ply: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    object_to_label = {str(row.get("object_id")): str(row.get("semantic_label") or "unknown") for row in rows}
    header, props, header_lines = read_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"object", "semantic", "red", "green", "blue"}
    if not required.issubset(idx):
        raise ValueError(f"PLY missing fields: {sorted(required - set(idx))}")
    changed_vertices = 0
    total_vertices = 0
    unmapped_vertices = 0
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with input_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in header:
            dst.write(line)
        for _ in range(header_lines):
            next(src)
        for line in src:
            parts = line.split()
            if len(parts) < len(props):
                continue
            total_vertices += 1
            object_id = str(int(float(parts[idx["object"]])))
            label = object_to_label.get(object_id)
            if label is None:
                unmapped_vertices += 1
                dst.write(line)
                continue
            semantic = str(LABEL_IDS.get(label, 0))
            color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
            if parts[idx["semantic"]] != semantic:
                changed_vertices += 1
            parts[idx["semantic"]] = semantic
            parts[idx["red"]] = str(color[0])
            parts[idx["green"]] = str(color[1])
            parts[idx["blue"]] = str(color[2])
            dst.write(" ".join(parts) + "\n")
    return {
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "total_vertices": total_vertices,
        "changed_vertices": changed_vertices,
        "unmapped_vertices": unmapped_vertices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path)
    parser.add_argument("--output-ply", type=Path)
    parser.add_argument("--min-dominant-ratio", type=float, default=0.68)
    parser.add_argument("--min-geometry-ratio", type=float, default=0.50)
    parser.add_argument("--horizontal-normal-z", type=float, default=0.90)
    parser.add_argument("--strong-horizontal-normal-z", type=float, default=0.96)
    parser.add_argument("--high-horizontal-ceiling-z", type=float, default=2.2)
    parser.add_argument("--ground-max-z-span-for-geometry", type=float, default=1.2)
    parser.add_argument("--ceiling-min-normal-z", type=float, default=0.86)
    parser.add_argument("--ceiling-min-z", type=float, default=2.2)
    parser.add_argument("--wall-max-normal-z", type=float, default=0.45)
    parser.add_argument("--strong-wall-normal-z", type=float, default=0.35)
    parser.add_argument("--wall-min-z-span", type=float, default=1.8)
    args = parser.parse_args()

    rows, report = resolve_objects(read_jsonl(args.objects_jsonl), args)
    write_jsonl(args.output_jsonl, rows)
    if args.input_ply or args.output_ply:
        if not args.input_ply or not args.output_ply:
            raise SystemExit("--input-ply and --output-ply must be provided together")
        report["ply"] = rewrite_ply(args.input_ply, args.output_ply, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
