#!/usr/bin/env python3
"""Materialize the existing per-view evidence as a stable Superpoint ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rows_by_object(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["object_id"]): row for row in rows}


def source_frames(rows: list[dict[str, Any]]) -> dict[int, set[int]]:
    return {
        int(row["object_id"]): {int(item["frame_id"]) for item in row.get("top_source_frames", [])}
        for row in rows
    }


def materialize(
    evidence_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one row per accepted `(superpoint, frame, camera)` observation."""
    objects = rows_by_object(object_rows)
    reviews = rows_by_object(review_rows)
    support = source_frames(provenance_rows)
    out = []
    for evidence in evidence_rows:
        object_id = int(evidence["object_id"])
        frame_id, cam_id = int(evidence["frame_id"]), int(evidence["cam_id"])
        obj = objects.get(object_id, {})
        parsed = reviews.get(object_id, {}).get("parsed") or {}
        out.append({
            "observation_id": f"sp{object_id}:f{frame_id}:c{cam_id}:r{int(evidence.get('rank') or 0)}",
            "superpoint_id": object_id,
            "frame_id": frame_id,
            "cam_id": cam_id,
            "rank": int(evidence.get("rank") or 0),
            "source_frame_confirmed": frame_id in support.get(object_id, set()),
            "geometry_type": str(obj.get("geometry_type") or "unknown"),
            "point_count": int(obj.get("count") or 0),
            "projected_points": int(evidence.get("projected_points") or 0),
            "depth_visible_ratio": float(evidence.get("depth_visible_ratio") or 0.0),
            "sky_filtered_ratio": float(evidence.get("sky_filtered_ratio") or 0.0),
            "median_depth": float(evidence.get("median_depth") or 0.0),
            "evidence_score": float(evidence.get("score") or 0.0),
            "crop_path": str(evidence.get("crop_path") or ""),
            "overlay_path": str(evidence.get("overlay_path") or ""),
            "vlm_description_zh": str(parsed.get("description_zh") or ""),
            "vlm_candidate_label": str(parsed.get("controlled_label") or "unknown"),
            "vlm_confidence": float(parsed.get("confidence") or 0.0),
            "vlm_surface_fragment": bool(parsed.get("is_surface_fragment")),
        })
    return sorted(out, key=lambda row: (row["superpoint_id"], row["rank"], row["cam_id"], row["frame_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--source-frame-support", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = materialize(
        read_jsonl(args.evidence_jsonl), read_jsonl(args.objects_jsonl),
        read_jsonl(args.review_jsonl), read_jsonl(args.source_frame_support),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema": "superpoint-observation-ledger/v1",
        "observations": len(rows),
        "superpoints": len({row["superpoint_id"] for row in rows}),
        "source_frame_confirmed_ratio": sum(row["source_frame_confirmed"] for row in rows) / max(len(rows), 1),
        "reviewed_superpoints": len({row["superpoint_id"] for row in rows if row["vlm_description_zh"]}),
        "candidate_labels": dict(Counter(row["vlm_candidate_label"] for row in rows if row["vlm_description_zh"])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
