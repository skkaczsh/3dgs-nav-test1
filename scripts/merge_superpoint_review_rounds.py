#!/usr/bin/env python3
"""Apply specific second-pass structural reviews without discarding round one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPECIFIC_STRUCTURES = {"floor", "wall", "grass", "roof", "ceiling", "stair"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_rounds(first_rows: list[dict[str, Any]], second_rows: list[dict[str, Any]], min_confidence: float) -> list[dict[str, Any]]:
    second_by_id = {int(row["object_id"]): row for row in second_rows}
    merged = []
    for first in first_rows:
        second = second_by_id.get(int(first["object_id"]))
        parsed = (second or {}).get("parsed") or {}
        valid = (
            str(parsed.get("controlled_label") or "") in SPECIFIC_STRUCTURES
            and bool(parsed.get("is_surface_fragment"))
            and float(parsed.get("confidence") or 0.0) >= min_confidence
        )
        if valid:
            merged.append({**second, "first_pass_review": first, "review_resolution": "specific_structure"})
        elif second:
            merged.append({**first, "structural_review": second, "review_resolution": "first_pass_retained"})
        else:
            merged.append({**first, "review_resolution": "no_structural_review"})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-review-jsonl", type=Path, required=True)
    parser.add_argument("--structural-review-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args()

    rows = merge_rounds(read_jsonl(args.first_review_jsonl), read_jsonl(args.structural_review_jsonl), args.min_confidence)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "first_review_count": len(rows),
        "specific_structure_overrides": sum(row["review_resolution"] == "specific_structure" for row in rows),
        "first_pass_retained": sum(row["review_resolution"] == "first_pass_retained" for row in rows),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
