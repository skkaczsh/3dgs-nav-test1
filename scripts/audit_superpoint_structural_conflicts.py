#!/usr/bin/env python3
"""Flag geometry-label conflicts for QA without changing any semantic label."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HORIZONTAL_LABELS = {"floor", "grass", "roof", "ceiling", "stair"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def conflict_reason(geometry_type: str, label: str) -> str:
    geometry_type = {
        "horizontal_surface": "horizontal",
        "vertical_surface": "vertical",
    }.get(geometry_type, geometry_type)
    if geometry_type == "thin_linear" and label in HORIZONTAL_LABELS | {"wall"}:
        return "thin_linear_labeled_as_surface"
    if geometry_type == "vertical" and label in HORIZONTAL_LABELS:
        return "vertical_labeled_as_horizontal_surface"
    if geometry_type == "horizontal" and label == "wall":
        return "horizontal_labeled_as_wall"
    return ""


def audit(objects: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    object_by_id = {int(row["object_id"]): row for row in objects}
    rows = []
    for review in reviews:
        object_id = int(review["object_id"])
        obj = object_by_id.get(object_id, {})
        parsed = review.get("parsed") or {}
        label = str(parsed.get("controlled_label") or "unknown")
        geometry_type = str(obj.get("geometry_type") or "unknown")
        reason = conflict_reason(geometry_type, label)
        if reason:
            rows.append({
                "object_id": object_id,
                "geometry_type": geometry_type,
                "candidate_label": label,
                "confidence": float(parsed.get("confidence") or 0.0),
                "conflict_reason": reason,
                "description_zh": str(parsed.get("description_zh") or ""),
            })
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--reviews-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = audit(read_jsonl(args.objects_jsonl), read_jsonl(args.reviews_jsonl))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"conflicts": len(rows), "reasons": dict(Counter(row["conflict_reason"] for row in rows))}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
