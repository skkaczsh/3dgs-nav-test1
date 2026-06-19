#!/usr/bin/env python3
"""Trace high-risk fused objects back to compact source target evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_frame_local_object_qa_pack import object_risk, read_jsonl


DROP_TARGET_KEYS = {"point_indices", "merged_point_indices"}


def compact_bbox(bbox: dict[str, Any] | None) -> dict[str, Any]:
    if not bbox:
        return {}
    out: dict[str, Any] = {}
    for key in ("min", "max"):
        values = bbox.get(key)
        if isinstance(values, list):
            out[key] = [round(float(v), 4) for v in values[:3]]
    return out


def compact_target(target: dict[str, Any]) -> dict[str, Any]:
    pca = target.get("pca") or {}
    return {
        "target_id": target.get("target_id"),
        "frame_id": target.get("frame_id"),
        "cam_id": target.get("cam_id"),
        "mask_id": target.get("mask_id"),
        "label": target.get("label"),
        "raw_label": target.get("raw_label"),
        "refined_from_label": target.get("refined_from_label"),
        "cluster_size": target.get("cluster_size"),
        "bbox_3d": compact_bbox(target.get("bbox_3d")),
        "bbox_2d": target.get("bbox_2d"),
        "linearity": round(float(pca.get("linearity", 0.0)), 4),
        "planarity": round(float(pca.get("planarity", 0.0)), 4),
        "normal": [round(float(v), 4) for v in (pca.get("normal") or [])[:3]],
        "image_path": target.get("image_path"),
        "mask_path": target.get("mask_path"),
        "refinement_reasons": target.get("refinement_reasons") or [],
    }


def target_lookup(targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("target_id")): row for row in targets}


def trace_objects(objects: list[dict[str, Any]], targets: list[dict[str, Any]], limit: int, evidence_per_object: int) -> list[dict[str, Any]]:
    by_target = target_lookup(targets)
    rows: list[dict[str, Any]] = []
    for obj in objects:
        score, reasons = object_risk(obj)
        if score <= 0:
            continue
        target_ids = [str(t) for t in obj.get("targets", [])]
        source_targets = [by_target[t] for t in target_ids if t in by_target]
        source_targets.sort(
            key=lambda row: (
                -int(row.get("cluster_size") or 0),
                int(row.get("frame_id") or 0),
                int(row.get("cam_id") or 0),
                str(row.get("target_id") or ""),
            )
        )
        rows.append(
            {
                "object_id": obj.get("object_id"),
                "semantic_label": obj.get("semantic_label"),
                "status": obj.get("status"),
                "risk_score": round(float(score), 3),
                "risk_reasons": reasons,
                "point_count": obj.get("point_count"),
                "target_count": obj.get("target_count"),
                "bbox_3d": compact_bbox(obj.get("bbox_3d")),
                "centroid": [round(float(v), 4) for v in (obj.get("centroid") or [])[:3]],
                "normal": [round(float(v), 4) for v in (obj.get("normal") or [])[:3]],
                "geometry_stats": obj.get("geometry_stats") or {},
                "label_votes": obj.get("label_votes") or {},
                "top_targets": [compact_target(t) for t in source_targets[:evidence_per_object]],
                "missing_target_count": max(0, len(target_ids) - len(source_targets)),
            }
        )
    rows.sort(key=lambda row: (-float(row["risk_score"]), -int(row.get("point_count") or 0), str(row["object_id"])))
    return rows[:limit]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "object_id",
        "semantic_label",
        "status",
        "risk_score",
        "risk_reasons",
        "point_count",
        "target_count",
        "first_frame",
        "first_cam",
        "first_mask",
        "first_target",
        "first_image_path",
        "first_mask_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            first = (row.get("top_targets") or [{}])[0]
            writer.writerow(
                {
                    "object_id": row.get("object_id"),
                    "semantic_label": row.get("semantic_label"),
                    "status": row.get("status"),
                    "risk_score": row.get("risk_score"),
                    "risk_reasons": "|".join(row.get("risk_reasons") or []),
                    "point_count": row.get("point_count"),
                    "target_count": row.get("target_count"),
                    "first_frame": first.get("frame_id"),
                    "first_cam": first.get("cam_id"),
                    "first_mask": first.get("mask_id"),
                    "first_target": first.get("target_id"),
                    "first_image_path": first.get("image_path"),
                    "first_mask_path": first.get("mask_path"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--evidence-per-object", type=int, default=5)
    args = parser.parse_args()

    rows = trace_objects(
        read_jsonl(args.objects_jsonl),
        read_jsonl(args.targets_jsonl),
        limit=args.limit,
        evidence_per_object=args.evidence_per_object,
    )
    write_jsonl(rows, args.output_jsonl)
    if args.output_csv:
        write_csv(rows, args.output_csv)
    print(
        json.dumps(
            {
                "objects_traced": len(rows),
                "output_jsonl": str(args.output_jsonl),
                "output_csv": str(args.output_csv) if args.output_csv else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
