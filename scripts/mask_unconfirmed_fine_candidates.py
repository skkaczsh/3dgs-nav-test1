#!/usr/bin/env python3
"""Mask unconfirmed fine-object candidates in a full-scene viewer pair.

The priority 2D segmenter has high recall for `car` and `railing`, but those
labels are only candidates until crop-level visual review confirms them. For
user QA, showing unconfirmed candidates as final semantic labels is misleading:
walls and surface seams can appear as cars/railings.

This script keeps all points and object ids, but rewrites selected unconfirmed
fine-object labels to `fine_candidate`. The original candidate label is kept in
JSON metadata for later DINO/GroundingDINO promotion.
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
    "fine_candidate": 17,
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


def parse_ply_header(path: Path) -> tuple[list[str], int]:
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
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
    return props, vertex_count


def should_mask(obj: dict[str, Any], labels: set[str], stages: set[str], keep_visual_confirmed: bool) -> bool:
    label = str(obj.get("semantic_label") or "unknown")
    if label not in labels:
        return False
    if stages and str(obj.get("downstream_stage") or "") not in stages:
        return False
    if keep_visual_confirmed and str(obj.get("visual_review_status") or "") == "visual_confirmed":
        return False
    return True


def transform_objects(objects: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[int, str]]:
    labels = set(args.labels)
    stages = set(args.downstream_stage)
    masked: dict[int, str] = {}
    out_rows: list[dict[str, Any]] = []
    for obj in objects:
        out = dict(obj)
        object_id = int(out["object_id"])
        old_label = str(out.get("semantic_label") or "unknown")
        if should_mask(out, labels, stages, args.keep_visual_confirmed):
            masked[object_id] = old_label
            out["semantic_label_original"] = out.get("semantic_label_original") or old_label
            out["candidate_label"] = old_label
            out["candidate_status"] = "unconfirmed_fine_object"
            out["semantic_label"] = args.mask_label
            out["status"] = f"unconfirmed_{old_label}_candidate"
            out["downstream_stage"] = "dino_fine_object_review"
            out["review_priority"] = "high"
            out["stable_surface"] = False
            out["scene_context"] = "unconfirmed_fine_object_candidate"
            out["scene_description"] = f"unconfirmed {old_label} candidate pending DINO/GroundingDINO review"
        out_rows.append(out)
    return out_rows, masked


def rewrite_ply(input_ply: Path, output_ply: Path, objects_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    props, vertex_count = parse_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    semantic_col = idx.get("semantic")
    if object_col is None or semantic_col is None:
        raise ValueError(f"PLY needs object and semantic fields: {input_ply}")

    changed_points = 0
    changed_objects: set[int] = set()
    semantic_counts = Counter()
    output_ply.parent.mkdir(parents=True, exist_ok=True)
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
            obj = objects_by_id.get(object_id)
            if obj:
                label = str(obj.get("semantic_label") or "unknown")
                semantic = LABEL_TO_SEMANTIC.get(label, 0)
                old_semantic = int(round(float(parts[semantic_col])))
                if semantic != old_semantic:
                    parts[semantic_col] = str(semantic)
                    changed_points += 1
                    changed_objects.add(object_id)
                semantic_counts[label] += 1
                dst.write(" ".join(parts) + "\n")
            else:
                dst.write(line)
    return {
        "vertex_count": vertex_count,
        "changed_points": changed_points,
        "changed_object_count": len(changed_objects),
        "semantic_counts_after": dict(semantic_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--input-objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="full_scene_objects_candidate_safe")
    parser.add_argument("--labels", nargs="+", default=["car", "railing"])
    parser.add_argument("--downstream-stage", action="append", default=["dino_fine_object_review"])
    parser.add_argument("--mask-label", default="fine_candidate")
    parser.add_argument("--keep-visual-confirmed", action="store_true")
    args = parser.parse_args()

    objects = read_jsonl(args.input_objects_jsonl)
    transformed, masked = transform_objects(objects, args)
    objects_by_id = {int(obj["object_id"]): obj for obj in transformed}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    out_ply = args.output_dir / f"{args.output_prefix}.ply"
    out_masked = args.output_dir / f"{args.output_prefix}_masked.jsonl"
    write_jsonl(out_jsonl, transformed)
    write_jsonl(
        out_masked,
        [
            {
                "object_id": object_id,
                "candidate_label": label,
                "new_label": args.mask_label,
            }
            for object_id, label in sorted(masked.items())
        ],
    )
    ply_report = rewrite_ply(args.input_ply, out_ply, objects_by_id)

    label_counts = Counter(str(obj.get("semantic_label") or "unknown") for obj in transformed)
    candidate_label_counts = Counter(masked.values())
    report = {
        "input_ply": str(args.input_ply),
        "input_objects_jsonl": str(args.input_objects_jsonl),
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "output_masked_jsonl": str(out_masked),
        "object_count": len(transformed),
        "masked_object_count": len(masked),
        "masked_candidate_label_counts": dict(candidate_label_counts),
        "object_label_counts_after": dict(label_counts),
        **ply_report,
    }
    (args.output_dir / f"{args.output_prefix}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
