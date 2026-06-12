#!/usr/bin/env python3
"""Summarize SAM2 TensorRT AMG per-crop trace JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TOTAL_KEYS = [
    "raw_candidates",
    "after_within_crop_nms",
    "dropped_near_crop_edge",
    "after_crop_edge_filter",
    "before_cross_crop_nms",
    "after_cross_crop_nms",
    "after_overlap_resolution",
]


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    traces = []
    for path in sorted(args.trace_dir.glob("*_trace.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        totals = data.get("totals", {})
        row = {"image_id": data.get("image_name", path.name.removesuffix("_trace.json"))}
        row.update({key: int(totals.get(key, 0)) for key in TOTAL_KEYS})
        row["crop_boxes"] = int(data.get("crop_boxes", len(data.get("crops", []))))
        traces.append(row)

    summary = {"trace_dir": str(args.trace_dir), "images": len(traces)}
    if traces:
        for key in TOTAL_KEYS + ["crop_boxes"]:
            summary[f"mean_{key}"] = float(np.mean([row[key] for row in traces]))
        summary["within_crop_nms_keep_ratio"] = safe_ratio(
            summary["mean_after_within_crop_nms"], summary["mean_raw_candidates"]
        )
        summary["crop_edge_drop_ratio"] = safe_ratio(
            summary["mean_dropped_near_crop_edge"], summary["mean_after_within_crop_nms"]
        )
        summary["cross_crop_nms_keep_ratio"] = safe_ratio(
            summary["mean_after_cross_crop_nms"], summary["mean_before_cross_crop_nms"]
        )
        summary["overlap_resolution_keep_ratio"] = safe_ratio(
            summary["mean_after_overlap_resolution"], summary["mean_after_cross_crop_nms"]
        )
    report = {"summary": summary, "rows": traces}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
