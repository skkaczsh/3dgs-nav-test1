#!/usr/bin/env python3
"""Compare an SPG candidate against a trusted baseline for over-merge risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORT = "superpoint_graph_v1_report.json"
OVERLAP = "overlap_top1000_fine005/bbox_overlap_top1000_report.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def fine_metrics(run_dir: Path) -> dict[str, int] | None:
    path = run_dir / OVERLAP
    if not path.is_file():
        return None
    fine = read_json(path).get("fine_cell_overlap", {})
    if not isinstance(fine, dict):
        return None
    return {
        "fine_high_pairs_50": int(fine.get("fine_high_pairs_50", 0)),
        "fine_high_pairs_95": int(fine.get("fine_high_pairs_95", 0)),
    }


def load_metrics(run_dir: Path) -> dict[str, Any]:
    report = read_json(run_dir / REPORT)
    accepted_reasons = report.get("accepted_reasons", {})
    if not isinstance(accepted_reasons, dict):
        accepted_reasons = {}
    metrics = {
        "run_dir": str(run_dir),
        "input_patch_count": int(report.get("input_patch_count", 0)),
        "output_patch_count": int(report.get("output_patch_count", 0)),
        "accepted_edges": int(report.get("accepted_edges", 0)),
        "uncertain_fragment_bridge": int(accepted_reasons.get("uncertain_fragment_bridge", 0)),
    }
    metrics.update(fine_metrics(run_dir) or {})
    return metrics


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_metrics(args.baseline_dir)
    candidate = load_metrics(args.candidate_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if candidate["uncertain_fragment_bridge"] > args.max_uncertain_fragment_edges:
        errors.append(
            "uncertain_fragment_bridge_exceeded="
            f"{candidate['uncertain_fragment_bridge']}>{args.max_uncertain_fragment_edges}"
        )

    accepted_limit = int(baseline["accepted_edges"] * (1.0 + args.max_accepted_edge_growth))
    if candidate["accepted_edges"] > accepted_limit:
        errors.append(f"accepted_edges_growth={candidate['accepted_edges']}>{accepted_limit}")

    for key, limit_arg in (
        ("fine_high_pairs_50", args.max_fine50_regression),
        ("fine_high_pairs_95", args.max_fine95_regression),
    ):
        if key not in baseline or key not in candidate:
            warnings.append(f"missing_{key}_overlap_metric")
            continue
        limit = baseline[key] + limit_arg
        if candidate[key] > limit:
            errors.append(f"{key}_regression={candidate[key]}>{limit}")

    return {
        "schema": "spg-overmerge-risk-compare/v1",
        "passed": not errors,
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {
            "output_patch_count": candidate["output_patch_count"] - baseline["output_patch_count"],
            "accepted_edges": candidate["accepted_edges"] - baseline["accepted_edges"],
            "fine_high_pairs_50": candidate.get("fine_high_pairs_50", 0) - baseline.get("fine_high_pairs_50", 0),
            "fine_high_pairs_95": candidate.get("fine_high_pairs_95", 0) - baseline.get("fine_high_pairs_95", 0),
        },
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-uncertain-fragment-edges", type=int, default=0)
    parser.add_argument("--max-accepted-edge-growth", type=float, default=0.5)
    parser.add_argument("--max-fine50-regression", type=int, default=0)
    parser.add_argument("--max-fine95-regression", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output = args.output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
