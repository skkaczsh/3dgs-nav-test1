#!/usr/bin/env python3
"""Run manual cross-candidate review normalization and object merge application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apply_cross_candidate_merge_reviews import apply_reviews, load_jsonl as load_objects, write_outputs as write_applied_outputs
from normalize_manual_merge_decisions import normalize, write_outputs as write_normalized_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-csv", type=Path, required=True)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    args = parser.parse_args()

    normalized_dir = args.output_dir / "normalized"
    applied_dir = args.output_dir / "applied"
    rows, errors = normalize(args.manual_csv, args.review_jsonl)
    write_normalized_outputs(rows, errors, normalized_dir)

    objects = load_objects(args.objects)
    merged, decisions = apply_reviews(objects, rows, args.min_confidence)
    write_applied_outputs(merged, decisions, applied_dir)

    report = {
        "manual_csv": str(args.manual_csv),
        "review_jsonl": str(args.review_jsonl),
        "objects": str(args.objects),
        "normalized_dir": str(normalized_dir),
        "applied_dir": str(applied_dir),
        "manual_review_count": len(rows),
        "manual_error_count": len(errors),
        "input_object_count": len(objects),
        "output_object_count": len(merged),
        "accepted_merge_count": sum(1 for row in decisions if row["accepted"]),
        "min_confidence": args.min_confidence,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manual_merge_workflow_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
