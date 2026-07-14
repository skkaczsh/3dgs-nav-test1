#!/usr/bin/env python3
"""Correct locally contradictory floor/ceiling anchors using horizontal strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return zero for overlap and the empty distance otherwise."""
    return max(0.0, max(first[0], second[0]) - min(first[1], second[1]))


def planar_gap(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_min, first_max = first["bbox_min"], first["bbox_max"]
    second_min, second_max = second["bbox_min"], second["bbox_max"]
    return max(
        interval_gap((float(first_min[0]), float(first_max[0])), (float(second_min[0]), float(second_max[0]))),
        interval_gap((float(first_min[1]), float(first_max[1])), (float(second_min[1]), float(second_max[1]))),
    )


def correct_anchors(
    anchors: list[dict[str, Any]], geometry: dict[int, dict[str, Any]], max_planar_gap: float,
    max_z_gap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Correct only a floor co-planar with a nearby ceiling anchor.

    Global Z is deliberately not used: roofs and elevated floors make it invalid.
    A correction requires a local ceiling reference at effectively the same height.
    """
    ceilings = [
        row for row in anchors
        if row.get("propagation_eligible") and row.get("anchor_label") == "ceiling"
        and geometry.get(int(row["object_id"]), {}).get("geometry_type") == "horizontal"
    ]
    output: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for row in anchors:
        corrected = dict(row)
        object_id = int(row["object_id"])
        current = geometry.get(object_id, {})
        if not (
            row.get("propagation_eligible") and row.get("anchor_label") == "floor"
            and current.get("geometry_type") == "horizontal"
        ):
            output.append(corrected)
            continue
        matches = []
        for ceiling in ceilings:
            reference_id = int(ceiling["object_id"])
            reference = geometry[reference_id]
            gap_xy = planar_gap(current, reference)
            gap_z = abs(float(current["centroid"][2]) - float(reference["centroid"][2]))
            if gap_xy <= max_planar_gap and gap_z <= max_z_gap:
                matches.append((gap_xy, gap_z, reference_id))
        if matches:
            gap_xy, gap_z, reference_id = min(matches)
            corrected["anchor_label"] = "ceiling"
            corrected["candidate_label_before_strata"] = row.get("candidate_label")
            corrected["anchor_status"] = "structural_anchor_strata_corrected"
            corrected["strata_correction"] = {
                "from": "floor",
                "to": "ceiling",
                "reference_ceiling_id": reference_id,
                "planar_gap_m": round(gap_xy, 4),
                "z_gap_m": round(gap_z, 4),
                "reason": "nearby horizontal ceiling anchor occupies the same local height stratum",
            }
            corrections.append({"object_id": object_id, **corrected["strata_correction"]})
        output.append(corrected)
    return output, corrections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors-jsonl", type=Path, required=True)
    parser.add_argument("--geometry-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-planar-gap", type=float, default=0.75)
    parser.add_argument("--max-z-gap", type=float, default=0.45)
    args = parser.parse_args()

    anchors = read_jsonl(args.anchors_jsonl)
    geometry = {int(row["object_id"]): row for row in read_jsonl(args.geometry_jsonl)}
    corrected, corrections = correct_anchors(anchors, geometry, args.max_planar_gap, args.max_z_gap)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as stream:
        for row in corrected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.write_text(json.dumps({
        "input_anchors": len(anchors),
        "corrections": corrections,
        "max_planar_gap_m": args.max_planar_gap,
        "max_z_gap_m": args.max_z_gap,
        "policy": "local horizontal co-stratum only; never use global height",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
