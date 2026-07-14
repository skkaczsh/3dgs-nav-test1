#!/usr/bin/env python3
"""Select only generic structural VLM reviews for a specific second pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_candidates(objects: list[dict[str, Any]], reviews: list[dict[str, Any]], min_confidence: float) -> list[dict[str, Any]]:
    by_id = {int(row["object_id"]): row for row in reviews}
    selected = []
    for obj in objects:
        review = by_id.get(int(obj["object_id"]), {})
        parsed = review.get("parsed") or {}
        if (
            parsed.get("controlled_label") == "building_part"
            and bool(parsed.get("is_surface_fragment"))
            and float(parsed.get("confidence") or 0.0) >= min_confidence
        ):
            selected.append({
                **obj,
                "first_pass_label": "building_part",
                "first_pass_confidence": float(parsed["confidence"]),
                "first_pass_description_zh": str(parsed.get("description_zh") or ""),
            })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--output-objects", type=Path, required=True)
    parser.add_argument("--output-ids", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args()

    selected = select_candidates(read_jsonl(args.objects_jsonl), read_jsonl(args.review_jsonl), args.min_confidence)
    args.output_objects.parent.mkdir(parents=True, exist_ok=True)
    args.output_objects.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    args.output_ids.write_text(json.dumps([int(row["object_id"]) for row in selected]) + "\n", encoding="utf-8")
    report = {"candidates": len(selected), "min_confidence": args.min_confidence}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
