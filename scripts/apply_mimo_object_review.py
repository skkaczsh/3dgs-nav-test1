#!/usr/bin/env python3
"""Apply high-confidence Mimo object review results to viewer PLY/JSONL.

This pass is intentionally conservative:

- Mimo metadata is always copied into object JSONL.
- PLY labels are changed only for parsed, high-confidence relabel decisions.
- Large geometry-trusted surfaces cannot be promoted to fine labels such as
  car/railing.
- `building_part` is mapped to wall for the current parking-scene viewer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


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

CONTROLLED_TO_VIEWER_LABEL = {
    "floor": "floor",
    "wall": "wall",
    "grass": "grass",
    "car": "car",
    "railing": "railing",
    "tree_or_shrub": "tree",
    "equipment": "equipment",
    "hvac_outdoor_unit": "equipment",
    "traffic_cone": "equipment",
    "pipe_or_pole": "pipe",
    "door_or_window": "wall",
    "sign_or_box": "equipment",
    "curb_or_low_barrier": "railing",
    "building_part": "wall",
    "unknown": "unknown",
}

LABEL_COLORS = {
    "unknown": (90, 90, 90),
    "wall": (160, 170, 180),
    "floor": (190, 172, 135),
    "ceiling": (180, 180, 210),
    "grass": (70, 150, 80),
    "tree": (60, 130, 65),
    "car": (235, 90, 80),
    "railing": (245, 200, 35),
    "pipe": (125, 210, 220),
    "equipment": (230, 55, 220),
    "building": (155, 155, 170),
}

SURFACE_LABELS = {"floor", "wall", "grass", "building"}
FINE_LABELS = {"car", "railing", "pipe", "equipment", "tree"}


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


def parse_ascii_ply_header(path: Path) -> tuple[list[str], list[str], int]:
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
    return header, props, vertex_count


def load_ascii_ply(path: Path) -> tuple[list[str], list[str], list[list[str]], np.ndarray]:
    header, props, vertex_count = parse_ascii_ply_header(path)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    if object_col is None or "semantic" not in idx:
        raise ValueError(f"PLY must contain object/object_id and semantic fields: {path}")
    rows: list[list[str]] = []
    object_ids = np.empty(vertex_count, dtype=np.uint32)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(len(header)):
            next(f)
        for i, line in enumerate(f):
            if i >= vertex_count:
                break
            parts = line.strip().split()
            rows.append(parts)
            object_ids[i] = int(round(float(parts[object_col])))
    if len(rows) != vertex_count:
        object_ids = object_ids[: len(rows)]
    return header, props, rows, object_ids


def write_ascii_ply(path: Path, header: list[str], props: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_cols = [props.index(k) for k in ("x", "y", "z")]
    with path.open("w", encoding="utf-8") as f:
        for line in header:
            f.write(line)
        for parts in rows:
            for col in xyz_cols:
                parts[col] = f"{float(parts[col]):.6f}"
            f.write(" ".join(parts) + "\n")


def normalized_review_label(parsed: dict[str, Any]) -> str:
    controlled = str(parsed.get("controlled_label") or "unknown")
    return CONTROLLED_TO_VIEWER_LABEL.get(controlled, "unknown")


def is_geometry_trusted_surface(obj: dict[str, Any], min_ratio: float) -> bool:
    majority = str(obj.get("surface_trust_guard_majority_label") or "")
    ratio = float(obj.get("surface_trust_guard_majority_ratio") or 0.0)
    if majority in {"floor", "wall", "grass"} and ratio >= min_ratio:
        return True
    status = str(obj.get("status") or "")
    if status.startswith("priority_ground") or status.startswith("priority_wall"):
        return True
    return False


def should_apply_relabel(
    obj: dict[str, Any],
    parsed: dict[str, Any],
    new_label: str,
    min_confidence: float,
    surface_guard_ratio: float,
) -> tuple[bool, str]:
    confidence = float(parsed.get("confidence") or 0.0)
    action = str(parsed.get("action") or "")
    if action not in {"relabel", "demote_to_unknown"}:
        return False, "action_not_relabel"
    if confidence < min_confidence:
        return False, "low_confidence"
    if new_label not in LABEL_TO_SEMANTIC:
        return False, "unsupported_viewer_label"

    old_label = str(obj.get("semantic_label") or "unknown")
    trusted_surface = is_geometry_trusted_surface(obj, surface_guard_ratio)
    if trusted_surface and new_label in FINE_LABELS:
        return False, "blocked_fine_label_on_trusted_surface"
    if old_label in SURFACE_LABELS and new_label in FINE_LABELS:
        return False, "blocked_surface_to_fine"
    if action == "demote_to_unknown" and new_label != "unknown":
        return False, "invalid_demote_label"
    return True, "applied"


def build_review_map(path: Path) -> dict[int, dict[str, Any]]:
    reviews: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        parsed = row.get("parsed")
        if parsed:
            reviews[int(row["object_id"])] = row
    return reviews


def apply_to_objects(
    objects: list[dict[str, Any]],
    reviews: dict[int, dict[str, Any]],
    min_confidence: float,
    surface_guard_ratio: float,
) -> tuple[list[dict[str, Any]], dict[int, str], Counter[str]]:
    relabels: dict[int, str] = {}
    reasons: Counter[str] = Counter()
    updated: list[dict[str, Any]] = []
    for obj in objects:
        out = dict(obj)
        object_id = int(out["object_id"])
        review = reviews.get(object_id)
        if not review:
            updated.append(out)
            continue
        parsed = dict(review["parsed"])
        new_label = normalized_review_label(parsed)
        ok, reason = should_apply_relabel(out, parsed, new_label, min_confidence, surface_guard_ratio)
        reasons[reason] += 1
        out["mimo_review"] = {
            "input_semantic_label": review.get("input_semantic_label"),
            "controlled_label": parsed.get("controlled_label"),
            "viewer_label": new_label,
            "description_zh": parsed.get("description_zh"),
            "is_true_object": parsed.get("is_true_object"),
            "is_surface_fragment": parsed.get("is_surface_fragment"),
            "confidence": parsed.get("confidence"),
            "action": parsed.get("action"),
            "reason_zh": parsed.get("reason_zh"),
            "apply_decision": reason,
        }
        if ok:
            out["semantic_label_before_mimo"] = out.get("semantic_label")
            out["semantic_label"] = new_label
            out["description"] = parsed.get("description_zh") or out.get("description", "")
            out["status"] = f"{out.get('status', 'object')}_mimo_{reason}"
            relabels[object_id] = new_label
        updated.append(out)
    return updated, relabels, reasons


def apply_to_ply(
    input_ply: Path,
    output_ply: Path,
    relabels: dict[int, str],
    recolor: bool,
) -> tuple[int, Counter[str]]:
    header, props, rows, object_ids = load_ascii_ply(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    semantic_col = idx["semantic"]
    red_col = idx.get("red")
    green_col = idx.get("green")
    blue_col = idx.get("blue")
    changed = 0
    label_counts: Counter[str] = Counter()
    for i, parts in enumerate(rows):
        object_id = int(object_ids[i])
        new_label = relabels.get(object_id)
        if not new_label:
            continue
        parts[semantic_col] = str(LABEL_TO_SEMANTIC[new_label])
        if recolor and red_col is not None and green_col is not None and blue_col is not None:
            r, g, b = LABEL_COLORS.get(new_label, LABEL_COLORS["unknown"])
            parts[red_col] = str(r)
            parts[green_col] = str(g)
            parts[blue_col] = str(b)
        changed += 1
        label_counts[new_label] += 1
    write_ascii_ply(output_ply, header, props, rows)
    return changed, label_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--mimo-review-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="full_scene_objects_mimo_reviewed")
    parser.add_argument("--min-confidence", type=float, default=0.72)
    parser.add_argument("--surface-guard-ratio", type=float, default=0.55)
    parser.add_argument("--no-recolor", action="store_true")
    args = parser.parse_args()

    objects = read_jsonl(args.objects_jsonl)
    reviews = build_review_map(args.mimo_review_jsonl)
    updated_objects, relabels, reasons = apply_to_objects(
        objects,
        reviews,
        args.min_confidence,
        args.surface_guard_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / f"{args.output_name}.jsonl"
    output_ply = args.output_dir / f"{args.output_name}.ply"
    write_jsonl(output_jsonl, updated_objects)
    changed_points, changed_by_label = apply_to_ply(
        args.input_ply,
        output_ply,
        relabels,
        recolor=not args.no_recolor,
    )

    report = {
        "input_ply": str(args.input_ply),
        "objects_jsonl": str(args.objects_jsonl),
        "mimo_review_jsonl": str(args.mimo_review_jsonl),
        "output_ply": str(output_ply),
        "output_jsonl": str(output_jsonl),
        "review_count": len(reviews),
        "relabel_object_count": len(relabels),
        "relabel_point_count": changed_points,
        "relabel_points_by_label": dict(changed_by_label),
        "decision_reasons": dict(reasons),
        "min_confidence": args.min_confidence,
        "surface_guard_ratio": args.surface_guard_ratio,
    }
    (args.output_dir / f"{args.output_name}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
