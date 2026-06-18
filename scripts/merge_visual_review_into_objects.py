#!/usr/bin/env python3
"""Merge crop-level visual review metadata into object JSONL.

This does not rewrite point labels. It only attaches detector/reviewer metadata
so the viewer and later scripts can distinguish:

- geometry plausible
- visual confirmed
- visual weak / not detected
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--visual-review-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    objects = read_jsonl(args.objects_jsonl)
    visual_rows = {int(row["object_id"]): row for row in read_jsonl(args.visual_review_jsonl)}
    merged = []
    status_counts = Counter()
    label_status_counts = Counter()
    merged_count = 0
    for obj in objects:
        out = dict(obj)
        object_id = int(out["object_id"])
        visual = visual_rows.get(object_id)
        if visual:
            merged_count += 1
            out["visual_review_status"] = visual.get("visual_status", "")
            out["visual_review_best_score"] = visual.get("best_score", 0.0)
            out["visual_review_best_phrase"] = visual.get("best_phrase", "")
            out["visual_review_evidence_count"] = visual.get("evidence_count", 0)
            out["visual_review_source"] = str(args.visual_review_jsonl)
            if visual.get("visual_status") == "visual_confirmed":
                out["status"] = f"{out.get('status', '')}_visual_confirmed".strip("_")
            elif visual.get("visual_status"):
                out["review_priority"] = "high"
        status = str(out.get("visual_review_status") or "not_visual_reviewed")
        label = str(out.get("semantic_label") or "unknown")
        status_counts[status] += 1
        label_status_counts[f"{label}:{status}"] += 1
        merged.append(out)

    write_jsonl(args.output_jsonl, merged)
    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "visual_review_jsonl": str(args.visual_review_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "object_count": len(objects),
        "visual_review_rows": len(visual_rows),
        "merged_count": merged_count,
        "visual_status_counts": dict(status_counts),
        "label_visual_status_counts": dict(label_status_counts),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
