#!/usr/bin/env python3
"""Split surface-only ambiguous viewer objects by their source target labels.

This operates after viewer export/remap/ambiguous resolution.  It is deliberately
narrow: only unresolved ambiguous objects whose target labels are all surface
labels are split.  The PLY object id, semantic id, and RGB are rewritten in sync
with the JSONL metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_ply_header(path: Path) -> tuple[list[str], list[str], int]:
    header: list[str] = []
    props: list[str] = []
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


def target_maps(targets: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_id = {str(row.get("target_id")): row for row in targets}
    by_index = {}
    for row in targets:
        try:
            by_index[int(row.get("target_index"))] = row
        except (TypeError, ValueError):
            continue
    return by_id, by_index


def is_unresolved_ambiguous(row: dict[str, Any]) -> bool:
    return str(row.get("semantic_label") or "") == "ambiguous" or str(row.get("status") or "") == "ambiguous_object"


def next_object_id(rows: list[dict[str, Any]]) -> int:
    ids = []
    for row in rows:
        try:
            ids.append(int(row.get("object_id")))
        except (TypeError, ValueError):
            continue
    return max(ids, default=0) + 1


def build_split_plan(
    objects: list[dict[str, Any]],
    targets_by_id: dict[str, dict[str, Any]],
    min_labels: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], tuple[int, str]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    point_map: dict[tuple[int, int], tuple[int, str]] = {}
    new_id = next_object_id(objects)
    split_objects = 0
    split_children = 0
    kept_ambiguous = 0
    label_counts = Counter()
    examples = []

    for obj in objects:
        if not is_unresolved_ambiguous(obj):
            output_rows.append(obj)
            continue
        try:
            old_object_id = int(obj.get("object_id"))
        except (TypeError, ValueError):
            output_rows.append(obj)
            kept_ambiguous += 1
            continue
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        missing_targets = []
        non_surface_labels = set()
        for target_id in obj.get("targets") or []:
            target = targets_by_id.get(str(target_id))
            if not target:
                missing_targets.append(str(target_id))
                continue
            label = str(target.get("label") or "unknown")
            if label not in SURFACE_LABELS:
                non_surface_labels.add(label)
            groups[label].append(target)
        groups = {label: rows for label, rows in groups.items() if label in SURFACE_LABELS}
        if missing_targets or non_surface_labels or len(groups) < min_labels:
            kept = dict(obj)
            kept["surface_split_skipped_reason"] = {
                "missing_targets": missing_targets[:10],
                "non_surface_labels": sorted(non_surface_labels),
                "surface_group_count": len(groups),
            }
            output_rows.append(kept)
            kept_ambiguous += 1
            continue

        split_objects += 1
        child_ids = []
        for label, group_targets in sorted(groups.items(), key=lambda item: (-sum(int(t.get("cluster_size") or 0) for t in item[1]), item[0])):
            child_id = new_id
            new_id += 1
            split_children += 1
            child_ids.append(child_id)
            target_ids = [str(t.get("target_id")) for t in group_targets]
            target_indices = []
            point_count = 0
            for target in group_targets:
                try:
                    target_index = int(target.get("target_index"))
                except (TypeError, ValueError):
                    continue
                target_indices.append(target_index)
                point_map[(old_object_id, target_index)] = (child_id, label)
                point_count += int(target.get("cluster_size") or 0)
            child = dict(obj)
            child["object_id"] = child_id
            child["viewer_object_id"] = child_id
            child["semantic_label"] = label
            child["status"] = "surface_ambiguous_split"
            child["parent_object_id"] = old_object_id
            child["split_from_ambiguous_object"] = old_object_id
            child["split_source_target_labels"] = sorted(groups)
            child["targets"] = target_ids
            child["target_count"] = len(target_ids)
            child["point_count"] = point_count
            child["label_votes"] = {label: point_count}
            child["split_target_indices"] = target_indices
            output_rows.append(child)
            label_counts[label] += 1
        examples.append({
            "object_id": old_object_id,
            "child_object_ids": child_ids,
            "labels": {label: len(rows) for label, rows in groups.items()},
            "point_count": obj.get("point_count"),
        })

    report = {
        "input_objects": len(objects),
        "output_objects": len(output_rows),
        "split_objects": split_objects,
        "split_children": split_children,
        "kept_ambiguous": kept_ambiguous,
        "child_label_counts": dict(label_counts),
        "examples": examples[:100],
        "point_map_entries": len(point_map),
    }
    return output_rows, point_map, report


def rewrite_ply(input_ply: Path, output_ply: Path, point_map: dict[tuple[int, int], tuple[int, str]]) -> dict[str, Any]:
    header, props, header_lines = read_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"object", "target", "semantic", "red", "green", "blue"}
    if not required.issubset(idx):
        raise ValueError(f"PLY missing fields: {sorted(required - set(idx))}")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    changed_vertices = 0
    total_vertices = 0
    label_point_counts = Counter()
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
            key = (int(float(parts[idx["object"]])), int(float(parts[idx["target"]])))
            replacement = point_map.get(key)
            if replacement:
                new_object, label = replacement
                color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
                parts[idx["object"]] = str(new_object)
                parts[idx["semantic"]] = str(LABEL_IDS.get(label, 0))
                parts[idx["red"]] = str(color[0])
                parts[idx["green"]] = str(color[1])
                parts[idx["blue"]] = str(color[2])
                label_point_counts[label] += 1
                changed_vertices += 1
            dst.write(" ".join(parts) + "\n")
    return {
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "total_vertices": total_vertices,
        "changed_vertices": changed_vertices,
        "changed_label_point_counts": dict(label_point_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-labels", type=int, default=2)
    args = parser.parse_args()

    objects = read_jsonl(args.objects_jsonl)
    targets_by_id, _targets_by_index = target_maps(read_jsonl(args.targets_jsonl))
    output_rows, point_map, report = build_split_plan(objects, targets_by_id, args.min_labels)
    write_jsonl(args.output_jsonl, output_rows)
    report["ply"] = rewrite_ply(args.input_ply, args.output_ply, point_map)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
