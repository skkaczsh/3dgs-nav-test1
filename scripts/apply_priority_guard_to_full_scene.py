#!/usr/bin/env python3
"""Apply priority candidate guard decisions to a full-scene viewer pair.

This rewrites a review PLY/JSONL pair without dropping points:

- geometry_plausible: keep original semantic label
- needs_visual_review: keep original semantic label, attach review status
- geometry_rejected: demote to unknown in the viewer PLY and JSONL

The raw object id stays unchanged so the rejected points remain inspectable.
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


def load_guard(path: Path) -> dict[int, dict[str, Any]]:
    guard = {}
    for row in read_jsonl(path):
        guard[int(row["object_id"])] = row
    return guard


def transform_object(obj: dict[str, Any], guard: dict[int, dict[str, Any]]) -> dict[str, Any]:
    object_id = int(obj["object_id"])
    g = guard.get(object_id)
    out = dict(obj)
    if not g:
        return out
    original_label = str(out.get("semantic_label") or "unknown")
    status = str(g.get("priority_guard_status") or "")
    reasons = list(g.get("priority_guard_reasons") or [])
    out["priority_guard_status"] = status
    out["priority_guard_reasons"] = reasons
    out["semantic_label_original"] = out.get("semantic_label_original") or original_label
    if "evidence_rank1" in g:
        out["evidence_rank1"] = g["evidence_rank1"]
    if status == "geometry_rejected":
        out["semantic_label"] = "unknown"
        out["status"] = "priority_geometry_rejected"
        out["downstream_stage"] = "fine_semantic_review"
        out["review_priority"] = "high"
        out["description"] = f"rejected {original_label} priority candidate"
    elif status == "needs_visual_review":
        out["status"] = f"priority_{original_label}_needs_visual_review"
        out["downstream_stage"] = "dino_fine_object_review"
        out["review_priority"] = "high"
    elif status == "geometry_plausible":
        out["status"] = f"priority_{original_label}_geometry_plausible"
        out["downstream_stage"] = "dino_fine_object_review"
    return out


def parse_ply_header(path: Path) -> tuple[list[str], int, list[str]]:
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    header: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            s = line.strip()
            parts = s.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif s == "end_header":
                break
    return props, vertex_count, header


def rewrite_ply(input_ply: Path, output_ply: Path, transformed_objects: dict[int, dict[str, Any]]) -> dict[str, Any]:
    props, vertex_count, header = parse_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    semantic_col = idx.get("semantic")
    if object_col is None or semantic_col is None:
        raise ValueError(f"PLY needs object and semantic fields: {input_ply}")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    changed_points = 0
    changed_objects = set()
    semantic_counts = Counter()
    with input_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            if line.strip() == "end_header":
                break
        for line_no in range(vertex_count):
            line = src.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) <= max(object_col, semantic_col):
                dst.write(line)
                continue
            object_id = int(round(float(parts[object_col])))
            obj = transformed_objects.get(object_id)
            if obj:
                semantic = LABEL_TO_SEMANTIC.get(str(obj.get("semantic_label") or "unknown"), 0)
                old_semantic = int(round(float(parts[semantic_col])))
                if semantic != old_semantic:
                    changed_points += 1
                    changed_objects.add(object_id)
                    parts[semantic_col] = str(semantic)
                semantic_counts[str(obj.get("semantic_label") or "unknown")] += 1
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--input-objects-jsonl", type=Path, required=True)
    parser.add_argument("--guard-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    guard = load_guard(args.guard_jsonl)
    objects = read_jsonl(args.input_objects_jsonl)
    transformed = [transform_object(obj, guard) for obj in objects]
    transformed_by_id = {int(obj["object_id"]): obj for obj in transformed}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "full_scene_objects_guarded.jsonl"
    output_ply = args.output_dir / "full_scene_objects_guarded_ascii.ply"
    write_jsonl(output_jsonl, transformed)
    ply_report = rewrite_ply(args.input_ply, output_ply, transformed_by_id)

    status_counts = Counter(str(obj.get("priority_guard_status") or "not_guarded") for obj in transformed)
    semantic_counts = Counter(str(obj.get("semantic_label") or "unknown") for obj in transformed)
    report = {
        "input_ply": str(args.input_ply),
        "input_objects_jsonl": str(args.input_objects_jsonl),
        "guard_jsonl": str(args.guard_jsonl),
        "output_ply": str(output_ply),
        "output_jsonl": str(output_jsonl),
        "object_count": len(transformed),
        "guarded_object_count": len(guard),
        "priority_guard_status_counts": dict(status_counts),
        "object_semantic_counts_after": dict(semantic_counts),
        **ply_report,
    }
    (args.output_dir / "full_scene_guard_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
