#!/usr/bin/env python3
"""Apply conservative relabels from geometry-conflict findings to a viewer PLY.

This is intentionally conservative: it demotes clearly inconsistent priority
objects to `unknown`, and only promotes wall -> floor for clean horizontal
surface fragments. It does not split objects; that should be a later plane/local
geometry stage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_TO_SEMANTIC = {
    "unknown": 0,
    "other": 1,
    "wall": 2,
    "floor": 3,
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
    "ignore": 255,
}


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


def choose_relabel(finding: dict[str, Any], args: argparse.Namespace) -> tuple[str | None, str]:
    label = str(finding.get("semantic_label") or "unknown")
    reasons = set(str(r) for r in finding.get("reasons") or [])
    metrics = finding.get("metrics") or {}
    planarity = float(metrics.get("planarity") or 0.0)
    thickness = float(metrics.get("thickness_rms") or 0.0)
    centroid_z = float(metrics.get("centroid_z") or 0.0)

    if label == "wall":
        if "wall_has_horizontal_normal" in reasons and planarity >= args.clean_horizontal_planarity and thickness <= args.clean_horizontal_thickness:
            new_label = args.wall_horizontal_low_label
            if args.wall_horizontal_z_split is not None and centroid_z >= args.wall_horizontal_z_split:
                new_label = args.wall_horizontal_high_label
            return new_label, f"wall_clean_horizontal_surface_to_{new_label}"
        if reasons & {"wall_has_horizontal_normal", "wall_low_planarity", "wall_high_thickness", "wall_oblique_normal"}:
            return "unknown", "wall_geometry_conflict_to_unknown"
    if label == "grass" and reasons & {"grass_large_vertical_extent", "grass_low_planarity"}:
        return "unknown", "grass_geometry_conflict_to_unknown"
    if label == "car" and reasons & {"car_high_centroid_z", "car_too_flat"}:
        return "unknown", "car_geometry_conflict_to_unknown"
    if label == "railing" and "railing_clean_horizontal_surface" in reasons:
        return "floor", "railing_clean_horizontal_surface_to_floor"
    if label == "railing" and "railing_surface_like_horizontal" in reasons:
        return "unknown", "railing_surface_like_to_unknown"
    if label == "floor" and "floor_not_horizontal" in reasons:
        return "unknown", "floor_not_horizontal_to_unknown"
    return None, "no_conservative_relabel"


def load_relabels(path: Path, args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    relabels: dict[int, dict[str, Any]] = {}
    only_labels = set(args.only_label or [])
    only_reasons = set(args.only_relabel_reason or [])
    for finding in read_jsonl(path):
        label = str(finding.get("semantic_label") or "unknown")
        if only_labels and label not in only_labels:
            continue
        object_id = int(finding["object_id"])
        new_label, reason = choose_relabel(finding, args)
        if not new_label:
            continue
        if only_reasons and reason not in only_reasons:
            continue
        relabels[object_id] = {
            "object_id": object_id,
            "old_label": finding.get("semantic_label"),
            "new_label": new_label,
            "reason": reason,
            "geometry_conflict_reasons": finding.get("reasons") or [],
            "geometry_conflict_severity": finding.get("severity"),
            "geometry_conflict_metrics": finding.get("metrics") or {},
        }
    return relabels


def parse_ply_header(path: Path) -> tuple[list[str], int, list[str]]:
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    header: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
    if vertex_count <= 0:
        raise ValueError(f"No vertex count in PLY: {path}")
    return props, vertex_count, header


def rewrite_ply(input_ply: Path, output_ply: Path, relabels: dict[int, dict[str, Any]]) -> dict[str, Any]:
    props, vertex_count, _header = parse_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    semantic_col = idx.get("semantic")
    if object_col is None or semantic_col is None:
        raise ValueError(f"PLY needs object and semantic fields: {input_ply}")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    changed_points = 0
    changed_objects: set[int] = set()
    semantic_counts = Counter()
    with input_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            if line.strip() == "end_header":
                break
        for _ in range(vertex_count):
            line = src.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) <= max(object_col, semantic_col):
                dst.write(line)
                continue
            object_id = int(round(float(parts[object_col])))
            relabel = relabels.get(object_id)
            if relabel:
                semantic = LABEL_TO_SEMANTIC[relabel["new_label"]]
                old_semantic = int(round(float(parts[semantic_col])))
                if semantic != old_semantic:
                    parts[semantic_col] = str(semantic)
                    changed_points += 1
                    changed_objects.add(object_id)
                semantic_counts[relabel["new_label"]] += 1
                dst.write(" ".join(parts) + "\n")
            else:
                try:
                    semantic_counts[str(int(round(float(parts[semantic_col]))))] += 1
                except ValueError:
                    pass
                dst.write(line)
    return {
        "vertex_count": vertex_count,
        "changed_points": changed_points,
        "changed_object_count": len(changed_objects),
        "semantic_counts_after": dict(semantic_counts),
    }


def rewrite_objects(objects_jsonl: Path, output_jsonl: Path, relabels: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for obj in read_jsonl(objects_jsonl):
        object_id = int(obj["object_id"])
        relabel = relabels.get(object_id)
        out = dict(obj)
        if relabel:
            out["semantic_label_original"] = out.get("semantic_label_original") or out.get("semantic_label")
            out["semantic_label"] = relabel["new_label"]
            out["geometry_conflict_relabel_status"] = "relabel_applied"
            out["geometry_conflict_relabel_reason"] = relabel["reason"]
            out["geometry_conflict_reasons"] = relabel["geometry_conflict_reasons"]
            out["geometry_conflict_severity"] = relabel["geometry_conflict_severity"]
            out["status"] = f"geometry_conflict_{relabel['reason']}"
            out["downstream_stage"] = "geometry_split_or_semantic_review"
            out["review_priority"] = "high"
            out["description"] = f"{relabel['old_label']} object relabeled by conservative geometry conflict QA"
        rows.append(out)
    write_jsonl(output_jsonl, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--input-objects-jsonl", type=Path, required=True)
    parser.add_argument("--conflicts-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="geometry_relabel")
    parser.add_argument("--clean-horizontal-planarity", type=float, default=0.80)
    parser.add_argument("--clean-horizontal-thickness", type=float, default=0.50)
    parser.add_argument(
        "--wall-horizontal-z-split",
        type=float,
        help="When set, clean horizontal wall conflicts below this z use --wall-horizontal-low-label and above/equal use --wall-horizontal-high-label.",
    )
    parser.add_argument("--wall-horizontal-low-label", default="floor")
    parser.add_argument("--wall-horizontal-high-label", default="floor")
    parser.add_argument(
        "--only-label",
        action="append",
        default=[],
        help="Only apply relabels for this semantic label. Repeat for multiple labels.",
    )
    parser.add_argument(
        "--only-relabel-reason",
        action="append",
        default=[],
        help="Only apply generated relabels with this reason. Repeat for multiple reasons.",
    )
    args = parser.parse_args()

    relabels = load_relabels(args.conflicts_jsonl, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = args.output_dir / f"{args.output_prefix}.ply"
    output_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    output_relabels = args.output_dir / f"{args.output_prefix}_relabels.jsonl"
    write_jsonl(output_relabels, list(relabels.values()))
    objects = rewrite_objects(args.input_objects_jsonl, output_jsonl, relabels)
    ply_report = rewrite_ply(args.input_ply, output_ply, relabels)

    label_counts = Counter(str(obj.get("semantic_label") or "unknown") for obj in objects)
    relabel_reason_counts = Counter(str(row["reason"]) for row in relabels.values())
    report = {
        "input_ply": str(args.input_ply),
        "input_objects_jsonl": str(args.input_objects_jsonl),
        "conflicts_jsonl": str(args.conflicts_jsonl),
        "output_ply": str(output_ply),
        "output_jsonl": str(output_jsonl),
        "output_relabels": str(output_relabels),
        "object_count": len(objects),
        "relabel_count": len(relabels),
        "object_label_counts_after": dict(label_counts),
        "relabel_reason_counts": dict(relabel_reason_counts),
        **ply_report,
    }
    (args.output_dir / f"{args.output_prefix}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
