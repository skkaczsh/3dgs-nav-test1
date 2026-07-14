#!/usr/bin/env python3
"""Turn reviewed Superpoint descriptions into conservative graph anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# `building_part` is deliberately excluded: it is a VLM fallback description,
# not a specific structural class safe to spread over a contact graph.
STRUCTURAL_LABELS = {"floor", "wall", "grass"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def anchor_row(object_row: dict[str, Any], review_row: dict[str, Any] | None, min_confidence: float) -> dict[str, Any]:
    object_id = int(object_row["object_id"])
    parsed = (review_row or {}).get("parsed") or {}
    label = str(parsed.get("controlled_label") or "unknown")
    confidence = float(parsed.get("confidence") or 0.0)
    is_surface = bool(parsed.get("is_surface_fragment"))
    propagate = is_surface and label in STRUCTURAL_LABELS and confidence >= min_confidence
    return {
        "object_id": object_id,
        "geometry_type": str(object_row.get("geometry_type") or "unknown"),
        "description_zh": str(parsed.get("description_zh") or ""),
        "candidate_label": label,
        "confidence": confidence,
        "is_surface_fragment": is_surface,
        "is_true_object": bool(parsed.get("is_true_object")),
        "anchor_label": label if propagate else "unknown",
        "propagation_eligible": propagate,
        "anchor_status": (
            "structural_anchor" if propagate
            else ("needs_structural_refinement" if is_surface and label == "building_part" else ("local_only" if parsed else "no_visual_evidence"))
        ),
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
