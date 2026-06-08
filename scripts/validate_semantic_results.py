#!/usr/bin/env python3
"""Validate semantic eval/projection artifacts against small-sample gates."""

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gate(value, threshold, op=">=") -> dict:
    if value is None:
        return {"value": value, "threshold": threshold, "pass": False}
    passed = value >= threshold if op == ">=" else value <= threshold
    return {"value": value, "threshold": threshold, "pass": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--projection", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-vlm-parse", type=float, default=0.9)
    parser.add_argument("--min-2d-coverage-with-sky", type=float, default=0.7)
    parser.add_argument("--min-point-labeled-ratio", type=float, default=0.7)
    args = parser.parse_args()

    qa = load_json(args.qa)
    combos = qa.get("semantic", {}).get("combos", {})
    combo_rows = {}
    for name, row in combos.items():
        combo_rows[name] = {
            "images": row.get("images"),
            "avg_coverage": row.get("avg_coverage"),
            "avg_coverage_with_sky": row.get("avg_coverage_with_sky"),
            "vlm_parse_success_rate": row.get("vlm_parse_success_rate"),
            "gates": {
                "vlm_parse": gate(row.get("vlm_parse_success_rate"), args.min_vlm_parse),
                "coverage_with_sky": gate(row.get("avg_coverage_with_sky"), args.min_2d_coverage_with_sky),
            },
        }

    projections = {}
    for path in args.projection:
        report = load_json(path)
        name = report.get("combo") or path.parent.name
        merged = report.get("merged", {})
        summary = report.get("summary", {})
        projections[name] = {
            "projection_dir": str(path.parent),
            "frame_count": summary.get("frame_count"),
            "ok_count": summary.get("ok_count"),
            "avg_labeled_ratio": summary.get("avg_labeled_ratio"),
            "merged_labeled_ratio": merged.get("labeled_ratio"),
            "merged_points": merged.get("points"),
            "gates": {
                "point_labeled_ratio": gate(merged.get("labeled_ratio"), args.min_point_labeled_ratio),
            },
        }

    all_gates = []
    for row in combo_rows.values():
        all_gates.extend(row["gates"].values())
    for row in projections.values():
        all_gates.extend(row["gates"].values())

    result = {
        "qa": str(args.qa),
        "thresholds": {
            "min_vlm_parse": args.min_vlm_parse,
            "min_2d_coverage_with_sky": args.min_2d_coverage_with_sky,
            "min_point_labeled_ratio": args.min_point_labeled_ratio,
        },
        "pass": all(g["pass"] for g in all_gates) if all_gates else False,
        "semantic_combos": combo_rows,
        "projections": projections,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
