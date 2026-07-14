#!/usr/bin/env python3
"""Make Region QA traceable back to its anchor observations and VLM reviews."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_manifest(
    regions: list[dict[str, Any]], observations: list[dict[str, Any]], reviews: list[dict[str, Any]],
    conflicts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    best_observation: dict[int, dict[str, Any]] = {}
    for row in observations:
        object_id = int(row["superpoint_id"])
        if object_id not in best_observation or float(row["evidence_score"]) > float(best_observation[object_id]["evidence_score"]):
            best_observation[object_id] = row
    review_by_id = {int(row["object_id"]): row for row in reviews}
    conflict_by_id = {int(row["object_id"]): row for row in conflicts or []}
    manifest = []
    for region in regions:
        anchors = [int(value) for value in region.get("source_anchor_ids", [])]
        evidence = []
        for anchor_id in anchors:
            observation = best_observation.get(anchor_id, {})
            parsed = review_by_id.get(anchor_id, {}).get("parsed") or {}
            conflict = conflict_by_id.get(anchor_id, {})
            evidence.append({
                "superpoint_id": anchor_id,
                "candidate_label": str(parsed.get("controlled_label") or "unknown"),
                "description_zh": str(parsed.get("description_zh") or ""),
                "confidence": float(parsed.get("confidence") or 0.0),
                "overlay_path": str(observation.get("overlay_path") or ""),
                "crop_path": str(observation.get("crop_path") or ""),
                "frame_id": observation.get("frame_id"),
                "cam_id": observation.get("cam_id"),
                "geometry_conflict": conflict.get("conflict_reason", ""),
            })
        count = int(region["superpoint_count"])
        anchors_count = max(len(anchors), 1)
        member_conflicts = [conflict_by_id[node] for node in region.get("superpoint_ids", []) if node in conflict_by_id]
        # Review both spatial impact and weak provenance; neither can hide the other.
        priority = (1 + int(region.get("max_hops") or 0)) * math.sqrt(count) * (1 + 1 / anchors_count)
        priority *= 1 + min(len(member_conflicts), 4) * 0.25
        manifest.append({
            "region_id": region["region_id"],
            "region_label": region["region_label"],
            "superpoint_count": count,
            "max_hops": int(region.get("max_hops") or 0),
            "qa_priority": round(priority, 4),
            "member_geometry_conflict_count": len(member_conflicts),
            "member_geometry_conflicts": member_conflicts,
            "source_anchor_evidence": evidence,
        })
    return sorted(manifest, key=lambda row: row["qa_priority"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions-jsonl", type=Path, required=True)
    parser.add_argument("--observations-jsonl", type=Path, required=True)
    parser.add_argument("--reviews-jsonl", type=Path, required=True)
    parser.add_argument("--conflicts-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = build_manifest(
        read_jsonl(args.regions_jsonl), read_jsonl(args.observations_jsonl), read_jsonl(args.reviews_jsonl),
        read_jsonl(args.conflicts_jsonl) if args.conflicts_jsonl else None,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"regions": len(rows), "top_region_ids": [row["region_id"] for row in rows[:20]]}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
