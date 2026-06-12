#!/usr/bin/env python3
"""Promotion gate for SAM2 TensorRT mask candidates.

The C++/TensorRT runner is allowed to produce production-shaped artifacts, but
promotion to the main SAM mask directory must pass this comparison gate against
the Python SAM2 baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fail(reasons: list[str], message: str) -> None:
    reasons.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-ok-images", type=int, default=10)
    parser.add_argument("--min-mean-matched-iou", type=float, default=0.93)
    parser.add_argument("--max-abs-mean-coverage-delta", type=float, default=0.06)
    parser.add_argument("--max-abs-row-coverage-delta", type=float, default=0.25)
    parser.add_argument("--max-mean-unmatched-baseline", type=float, default=4.0)
    parser.add_argument("--max-mean-unmatched-candidate", type=float, default=8.0)
    parser.add_argument("--max-missing", type=int, default=0)
    args = parser.parse_args()

    data = json.loads(args.compare_json.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    rows = data.get("rows", [])
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    reasons: list[str] = []

    ok_images = int(summary.get("ok_images", 0))
    missing = int(summary.get("missing_baseline", 0)) + int(summary.get("missing_candidate", 0))
    mean_iou = float(summary.get("mean_matched_iou", 0.0))
    mean_cov_delta = float(summary.get("mean_coverage_delta", 0.0))
    mean_unmatched_baseline = float(summary.get("mean_unmatched_baseline_masks", 0.0))
    mean_unmatched_candidate = float(summary.get("mean_unmatched_candidate_masks", 0.0))

    if ok_images < args.min_ok_images:
      fail(reasons, f"ok_images {ok_images} < {args.min_ok_images}")
    if missing > args.max_missing:
      fail(reasons, f"missing masks {missing} > {args.max_missing}")
    if mean_iou < args.min_mean_matched_iou:
      fail(reasons, f"mean_matched_iou {mean_iou:.4f} < {args.min_mean_matched_iou:.4f}")
    if abs(mean_cov_delta) > args.max_abs_mean_coverage_delta:
      fail(
          reasons,
          f"abs(mean_coverage_delta) {abs(mean_cov_delta):.4f} > {args.max_abs_mean_coverage_delta:.4f}",
      )
    if mean_unmatched_baseline > args.max_mean_unmatched_baseline:
      fail(
          reasons,
          f"mean_unmatched_baseline_masks {mean_unmatched_baseline:.4f} > {args.max_mean_unmatched_baseline:.4f}",
      )
    if mean_unmatched_candidate > args.max_mean_unmatched_candidate:
      fail(
          reasons,
          f"mean_unmatched_candidate_masks {mean_unmatched_candidate:.4f} > {args.max_mean_unmatched_candidate:.4f}",
      )

    worst_rows = sorted(
        ok_rows,
        key=lambda row: abs(float(row.get("coverage_delta", 0.0))),
        reverse=True,
    )[:10]
    if worst_rows:
        worst_delta = abs(float(worst_rows[0].get("coverage_delta", 0.0)))
        if worst_delta > args.max_abs_row_coverage_delta:
            fail(
                reasons,
                f"worst abs row coverage_delta {worst_delta:.4f} > {args.max_abs_row_coverage_delta:.4f}",
            )

    report = {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "thresholds": {
            "min_ok_images": args.min_ok_images,
            "min_mean_matched_iou": args.min_mean_matched_iou,
            "max_abs_mean_coverage_delta": args.max_abs_mean_coverage_delta,
            "max_abs_row_coverage_delta": args.max_abs_row_coverage_delta,
            "max_mean_unmatched_baseline": args.max_mean_unmatched_baseline,
            "max_mean_unmatched_candidate": args.max_mean_unmatched_candidate,
            "max_missing": args.max_missing,
        },
        "summary": summary,
        "worst_coverage_delta_rows": worst_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
