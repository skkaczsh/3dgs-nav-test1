#!/usr/bin/env python3
"""Audit immutable official Superpoint ownership before semantic evidence attaches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ownership_report(labels: np.ndarray, point_count: int, small_threshold: int) -> dict[str, Any]:
    """Summarize the one-point-one-superpoint ownership invariant."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if len(labels) != point_count:
        raise ValueError(f"label count differs from reference PLY: {len(labels)} != {point_count}")
    if len(labels) == 0:
        return {
            "points": 0,
            "superpoints": 0,
            "labels_contiguous": True,
            "unassigned_points": 0,
            "small_superpoints": 0,
        }
    if np.any(labels < 0):
        raise ValueError("labels contain negative ids")

    ids, counts = np.unique(labels.astype(np.int64, copy=False), return_counts=True)
    contiguous = bool(np.array_equal(ids, np.arange(len(ids), dtype=np.int64)))
    sorted_counts = np.sort(counts)
    return {
        "points": int(len(labels)),
        "superpoints": int(len(ids)),
        "labels_contiguous": contiguous,
        "min_label": int(ids[0]),
        "max_label": int(ids[-1]),
        "unassigned_points": 0,
        "small_threshold": int(small_threshold),
        "small_superpoints": int((counts < small_threshold).sum()),
        "singleton_superpoints": int((counts == 1).sum()),
        "point_count_quantiles": {
            "p05": float(np.quantile(sorted_counts, 0.05)),
            "p50": float(np.quantile(sorted_counts, 0.50)),
            "p95": float(np.quantile(sorted_counts, 0.95)),
            "max": int(sorted_counts[-1]),
        },
        "largest_superpoints": [int(value) for value in sorted_counts[-20:][::-1]],
    }


def objects_agree(rows: list[dict[str, Any]], labels: np.ndarray) -> dict[str, Any]:
    """Check that exported object metadata is an exact view of the label array."""
    counts = np.bincount(labels.astype(np.int64, copy=False))
    by_id = {int(row["object_id"]): row for row in rows}
    expected = set(np.flatnonzero(counts).tolist())
    actual = set(by_id)
    count_mismatches = [
        object_id for object_id in sorted(expected & actual)
        if int(by_id[object_id].get("count", -1)) != int(counts[object_id])
    ]
    return {
        "objects_rows": len(rows),
        "missing_object_rows": len(expected - actual),
        "unexpected_object_rows": len(actual - expected),
        "count_mismatches": len(count_mismatches),
        "count_mismatch_examples": count_mismatches[:20],
        "exact": not (expected - actual or actual - expected or count_mismatches),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--small-threshold", type=int, default=10)
    args = parser.parse_args()

    vertex_count = int(PlyData.read(str(args.reference_ply))["vertex"].count)
    labels = np.load(args.labels)
    report: dict[str, Any] = {
        "schema": "official-superpoint-ownership-qa/v1",
        "reference_ply": str(args.reference_ply),
        "labels": str(args.labels),
        "ownership": ownership_report(labels, vertex_count, args.small_threshold),
    }
    if args.objects_jsonl:
        report["objects"] = objects_agree(read_jsonl(args.objects_jsonl), labels)
    report["passed"] = bool(
        report["ownership"]["labels_contiguous"]
        and report["ownership"]["unassigned_points"] == 0
        and (not args.objects_jsonl or report["objects"]["exact"])
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
