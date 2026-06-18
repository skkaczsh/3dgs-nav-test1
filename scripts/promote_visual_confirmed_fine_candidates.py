#!/usr/bin/env python3
"""Promote visually confirmed fine-object candidates back to final labels.

The v9 full-scene viewer intentionally masks unconfirmed car/railing candidates
as `fine_candidate`. This script consumes crop-level visual review output
(for example GroundingDINO) and promotes only confirmed candidates back to their
candidate label. Unconfirmed candidates remain candidates, not final semantics.
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


def review_is_confirmed(review: dict[str, Any], min_score: float) -> bool:
    return (
        str(review.get("visual_status") or "") == "visual_confirmed"
        and float(review.get("best_score") or 0.0) >= min_score
    )


def transform_objects(
    objects: list[dict[str, Any]],
    review_by_id: dict[int, dict[str, Any]],
    min_score: float,
    candidate_labels: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    promoted: dict[int, str] = {}
    reviewed_candidates = 0
    candidate_status_counts = Counter()
    label_status_counts = Counter()

    for obj in objects:
        out = dict(obj)
        object_id = int(out["object_id"])
        review = review_by_id.get(object_id)
        candidate_label = str(out.get("candidate_label") or out.get("semantic_label_original") or "")
        is_candidate = str(out.get("semantic_label") or "") == "fine_candidate" and candidate_label in candidate_labels

        if review:
            out["visual_review_status"] = review.get("visual_status", "")
            out["visual_review_best_score"] = float(review.get("best_score") or 0.0)
            out["visual_review_best_phrase"] = review.get("best_phrase", "")
            out["visual_review_evidence_count"] = int(review.get("evidence_count") or 0)
            out["visual_review_source"] = "groundingdino"

        if is_candidate:
            reviewed_candidates += int(review is not None)
            status = str(review.get("visual_status") if review else "not_visual_reviewed")
            candidate_status_counts[status] += 1
            label_status_counts[f"{candidate_label}:{status}"] += 1
            if review and review_is_confirmed(review, min_score):
                out["semantic_label"] = candidate_label
                out["candidate_status"] = "visual_confirmed_fine_object"
                out["status"] = f"visual_confirmed_{candidate_label}"
                out["scene_context"] = f"visual_confirmed_{candidate_label}"
                out["scene_description"] = f"visual confirmed {candidate_label}"
                out["review_priority"] = "low"
                promoted[object_id] = candidate_label
            elif review:
                out["candidate_status"] = f"visual_unconfirmed_{candidate_label}"
                out["status"] = f"unconfirmed_{candidate_label}_candidate"
                out["review_priority"] = "high"
        transformed.append(out)

    report = {
        "reviewed_candidate_count": reviewed_candidates,
        "promoted_object_count": len(promoted),
        "promoted_label_counts": dict(Counter(promoted.values())),
        "candidate_visual_status_counts": dict(candidate_status_counts),
        "candidate_label_visual_status_counts": dict(label_status_counts),
    }
    return transformed, report


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
            if not obj:
                dst.write(line)
                continue
            label = str(obj.get("semantic_label") or "unknown")
            semantic = LABEL_TO_SEMANTIC.get(label, 0)
            old_semantic = int(round(float(parts[semantic_col])))
            if semantic != old_semantic:
                parts[semantic_col] = str(semantic)
                changed_points += 1
                changed_objects.add(object_id)
            semantic_counts[label] += 1
            dst.write(" ".join(parts) + "\n")
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
    parser.add_argument("--visual-review-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="full_scene_objects_visual_promoted")
    parser.add_argument("--min-score", type=float, default=0.34)
    parser.add_argument("--candidate-labels", nargs="+", default=["car", "railing"])
    args = parser.parse_args()

    objects = read_jsonl(args.input_objects_jsonl)
    visual_rows = read_jsonl(args.visual_review_jsonl)
    review_by_id = {int(row["object_id"]): row for row in visual_rows}
    transformed, transform_report = transform_objects(
        objects,
        review_by_id,
        args.min_score,
        set(args.candidate_labels),
    )
    objects_by_id = {int(obj["object_id"]): obj for obj in transformed}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    out_ply = args.output_dir / f"{args.output_prefix}.ply"
    write_jsonl(out_jsonl, transformed)
    ply_report = rewrite_ply(args.input_ply, out_ply, objects_by_id)

    report = {
        "input_ply": str(args.input_ply),
        "input_objects_jsonl": str(args.input_objects_jsonl),
        "visual_review_jsonl": str(args.visual_review_jsonl),
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "object_count": len(transformed),
        "visual_review_rows": len(visual_rows),
        "object_label_counts_after": dict(Counter(str(obj.get("semantic_label") or "unknown") for obj in transformed)),
        **transform_report,
        **ply_report,
    }
    (args.output_dir / f"{args.output_prefix}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
