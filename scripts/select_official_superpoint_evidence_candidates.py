#!/usr/bin/env python3
"""Select a deterministic, geometry- and scale-stratified evidence canary.

The full official Superpoint partition has tens of thousands of cells.  A
visual evidence pass must cover both large structural patches and reviewable
mid-scale patches without making size or geometry an accidental proxy label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    """Pick `count` ordered representatives including both ends."""
    if count <= 0 or size <= 0:
        return []
    if count >= size:
        return list(range(size))
    return [round(index * (size - 1) / (count - 1)) for index in range(count)] if count > 1 else [size // 2]


def select_candidates(
    rows: list[dict[str, Any]], per_geometry: int, min_points: int, coverage_budget: int = 0,
) -> list[dict[str, Any]]:
    """Select diagnostic strata, then optionally add point-mass coverage anchors.

    The strata answer "does the evidence chain work for every geometry and
    scale?".  They are not a good proxy for scene coverage: a handful of
    facade/ground Superpoints can contain most of the dense voxels.  The
    optional second budget adds only the largest *unselected* cells, without
    allowing that mass objective to erase the diagnostic strata.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("count") or 0) >= min_points:
            groups[str(row.get("geometry_type") or "unknown")].append(row)

    selected: list[dict[str, Any]] = []
    for geometry in sorted(groups):
        # Log count bins make a 50-point railing candidate as likely to be
        # reviewed as a 100k-point facade, while preserving determinism.
        ordered = sorted(groups[geometry], key=lambda row: (int(row.get("count") or 0), int(row["object_id"])))
        for rank, index in enumerate(evenly_spaced_indices(len(ordered), per_geometry), 1):
            row = dict(ordered[index])
            row.update({
                "evidence_candidate_policy": "geometry_log_scale_stratified/v1",
                "evidence_geometry_rank": rank,
                "evidence_geometry_pool": len(ordered),
                "evidence_min_points": min_points,
            })
            selected.append(row)
    selected_ids = {int(row["object_id"]) for row in selected}
    eligible = [row for row in rows if int(row.get("count") or 0) >= min_points]
    total_eligible_points = sum(int(row.get("count") or 0) for row in eligible)
    for rank, source in enumerate(
        sorted(
            (row for row in eligible if int(row["object_id"]) not in selected_ids),
            key=lambda row: (-int(row.get("count") or 0), int(row["object_id"])),
        )[:max(coverage_budget, 0)],
        1,
    ):
        row = dict(source)
        point_count = int(row.get("count") or 0)
        row.update({
            "evidence_candidate_policy": "point_mass_coverage/v1",
            "evidence_coverage_rank": rank,
            "evidence_coverage_point_share": round(point_count / max(total_eligible_points, 1), 9),
            "evidence_min_points": min_points,
        })
        selected.append(row)
    return sorted(selected, key=lambda row: int(row["object_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--per-geometry", type=int, default=60)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--coverage-budget", type=int, default=0,
                        help="Add this many largest unselected Superpoints after geometry/scale strata.")
    args = parser.parse_args()

    rows = read_jsonl(args.objects_jsonl)
    selected = select_candidates(rows, args.per_geometry, args.min_points, args.coverage_budget)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": "official-superpoint-evidence-candidates/v2",
        "objects_jsonl": str(args.objects_jsonl),
        "candidate_count": len(selected),
        "geometry_counts": dict(Counter(str(row.get("geometry_type") or "unknown") for row in selected)),
        "policy_counts": dict(Counter(str(row.get("evidence_candidate_policy") or "unknown") for row in selected)),
        "candidate_point_count": sum(int(row.get("count") or 0) for row in selected),
        "eligible_point_count": sum(int(row.get("count") or 0) for row in rows if int(row.get("count") or 0) >= args.min_points),
        "params": {
            "per_geometry": args.per_geometry,
            "min_points": args.min_points,
            "coverage_budget": args.coverage_budget,
        },
    }
    report_path = args.report or args.output_jsonl.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
