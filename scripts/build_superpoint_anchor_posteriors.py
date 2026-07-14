#!/usr/bin/env python3
"""Turn reviewed Superpoint descriptions into conservative graph anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .audit_superpoint_structural_conflicts import conflict_reason
except ImportError:  # Direct script execution keeps scripts/ on sys.path.
    from audit_superpoint_structural_conflicts import conflict_reason


# `building_part` is deliberately excluded: it is a VLM fallback description,
# not a specific structural class safe to spread over a contact graph.
STRUCTURAL_LABELS = {"floor", "wall", "grass", "roof", "ceiling", "stair"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def anchor_row(object_row: dict[str, Any], review_row: dict[str, Any] | None, min_confidence: float) -> dict[str, Any]:
    object_id = int(object_row["object_id"])
    parsed = (review_row or {}).get("parsed") or {}
    label = str(parsed.get("controlled_label") or "unknown")
    attachment = str(parsed.get("surface_attachment") or "unknown")
    confidence = float(parsed.get("confidence") or 0.0)
    is_surface = bool(parsed.get("is_surface_fragment"))
    geometry_type = str(object_row.get("geometry_type") or "unknown")
    geometry_conflict = conflict_reason(geometry_type, label)
    propagate = is_surface and label in STRUCTURAL_LABELS and confidence >= min_confidence and not geometry_conflict
    if geometry_conflict:
        status = "geometry_conflict_local_only"
    elif propagate:
        status = "structural_anchor"
    elif is_surface and label == "building_part":
        status = "needs_structural_refinement"
    elif parsed:
        status = "local_only"
    else:
        status = "no_visual_evidence"
    return {
        "object_id": object_id,
        "geometry_type": geometry_type,
        "description_zh": str(parsed.get("description_zh") or ""),
        "candidate_label": label,
        "surface_attachment": attachment,
        "confidence": confidence,
        "is_surface_fragment": is_surface,
        "is_true_object": bool(parsed.get("is_true_object")),
        "geometry_conflict": geometry_conflict,
        "anchor_label": label if propagate else "unknown",
        "propagation_eligible": propagate,
        "anchor_status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args()

    reviews = {int(row["object_id"]): row for row in read_jsonl(args.review_jsonl)}
    rows = [anchor_row(row, reviews.get(int(row["object_id"])), args.min_confidence) for row in read_jsonl(args.objects_jsonl)]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "objects": len(rows),
        "reviewed": sum(bool(row["description_zh"]) for row in rows),
        "structural_anchors": sum(bool(row["propagation_eligible"]) for row in rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
