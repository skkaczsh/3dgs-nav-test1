#!/usr/bin/env python3
"""Merge reviewed fine-candidate splits back into a full-scene preview.

The fine review layer is a replacement for its parent candidate objects, not an
overlay.  This script removes parent fine-object IDs from the base full-scene
PLY/JSONL and appends reviewed split objects/points.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_ascii_ply_header(path: Path) -> tuple[list[str], list[str], int, int]:
    header: list[str] = []
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "format" and parts[1] != "ascii":
                raise ValueError(f"Only ascii PLY is supported: {path}")
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
    return header, props, vertex_count, len(header)


def object_column(props: list[str]) -> int:
    for name in ("object", "object_id", "obj"):
        if name in props:
            return props.index(name)
    raise ValueError("PLY missing object/object_id field")


def count_kept_rows(path: Path, skip_objects: set[int]) -> tuple[int, Counter]:
    _header, props, _vertex_count, header_lines = parse_ascii_ply_header(path)
    obj_col = object_column(props)
    kept = 0
    skipped = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) <= obj_col:
                continue
            object_id = int(round(float(parts[obj_col])))
            if object_id in skip_objects:
                skipped[object_id] += 1
            else:
                kept += 1
    return kept, skipped


def write_merged_ply(base_ply: Path, fine_ply: Path, output_ply: Path, skip_objects: set[int]) -> dict[str, Any]:
    base_header, base_props, base_vertices, base_header_lines = parse_ascii_ply_header(base_ply)
    fine_header, fine_props, fine_vertices, fine_header_lines = parse_ascii_ply_header(fine_ply)
    if base_props != fine_props:
        raise ValueError(f"PLY schemas differ: base={base_props} fine={fine_props}")

    kept_base, skipped_counts = count_kept_rows(base_ply, skip_objects)
    total_vertices = kept_base + fine_vertices
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    obj_col = object_column(base_props)

    with output_ply.open("w", encoding="utf-8") as out:
        for line in base_header:
            if line.startswith("element vertex "):
                out.write(f"element vertex {total_vertices}\n")
            else:
                out.write(line)

        with base_ply.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(base_header_lines):
                next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) <= obj_col:
                    continue
                object_id = int(round(float(parts[obj_col])))
                if object_id not in skip_objects:
                    out.write(line)

        with fine_ply.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(fine_header_lines):
                next(f)
            for line in f:
                out.write(line)

    return {
        "base_vertices": base_vertices,
        "fine_vertices": fine_vertices,
        "kept_base_vertices": kept_base,
        "removed_base_vertices": int(sum(skipped_counts.values())),
        "output_vertices": total_vertices,
        "removed_parent_object_count": len([k for k, v in skipped_counts.items() if v > 0]),
        "top_removed_parent_points": [
            {"object_id": int(k), "points": int(v)}
            for k, v in skipped_counts.most_common(20)
        ],
    }


def merge_objects(base_objects: list[dict[str, Any]], fine_objects: list[dict[str, Any]], parent_ids: set[int]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for obj in base_objects:
        object_id = int(obj["object_id"])
        if object_id in parent_ids:
            continue
        merged.append(obj)
    for obj in fine_objects:
        out = dict(obj)
        out["merged_into_full_scene"] = True
        out["source_layer"] = "fine_candidate_geometry_review_v17"
        merged.append(out)
    merged.sort(key=lambda row: int(row["object_id"]))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ply", type=Path, required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--fine-ply", type=Path, required=True)
    parser.add_argument("--fine-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_objects = read_jsonl(args.base_jsonl)
    fine_objects = read_jsonl(args.fine_jsonl)
    parent_ids = {int(obj["parent_object_id"]) for obj in fine_objects if obj.get("parent_object_id") is not None}

    out_ply = args.output_dir / "full_scene_fine_review_v18.ply"
    out_jsonl = args.output_dir / "full_scene_fine_review_v18.jsonl"
    ply_report = write_merged_ply(args.base_ply, args.fine_ply, out_ply, parent_ids)
    merged_objects = merge_objects(base_objects, fine_objects, parent_ids)
    write_jsonl(out_jsonl, merged_objects)

    report = {
        "base_ply": str(args.base_ply),
        "base_jsonl": str(args.base_jsonl),
        "fine_ply": str(args.fine_ply),
        "fine_jsonl": str(args.fine_jsonl),
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "base_object_count": len(base_objects),
        "fine_object_count": len(fine_objects),
        "removed_parent_object_count": len(parent_ids),
        "output_object_count": len(merged_objects),
        "semantic_label_counts": dict(Counter(str(row.get("semantic_label")) for row in merged_objects)),
        "review_label_counts": dict(Counter(str(row.get("review_label") or "") for row in merged_objects if row.get("review_label"))),
        "review_status_counts": dict(Counter(str(row.get("review_status") or "") for row in merged_objects if row.get("review_status"))),
        "ply_report": ply_report,
    }
    (args.output_dir / "full_scene_fine_review_v18_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
