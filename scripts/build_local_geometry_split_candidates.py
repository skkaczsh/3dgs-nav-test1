#!/usr/bin/env python3
"""Build object-level local-geometry split candidates from viewer JSONL.

This is a narrow bridge between viewer QA and
split_priority_objects_by_local_geometry.py.  It does not rewrite points.  It
selects large fine-object viewer records whose 3D geometry is already
suspicious, then emits a conflicts JSONL with the fields consumed by the local
geometry splitter.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REASONS = {
    "large_fine_object",
    "large_single_target_object",
    "railing_not_linear",
    "railing_extent_too_large",
    "car_extent_suspicious",
    "car_surface_like",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def object_point_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("point_count") or 0)
    except (TypeError, ValueError):
        return 0


def viewer_object_id(row: dict[str, Any]) -> int:
    value = row.get("viewer_object_id", row.get("object_id"))
    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("obj_"):
        text = text.rsplit("_", 1)[-1]
    return int(text)


def bbox_extent(bbox: dict[str, Any]) -> tuple[float, float, float]:
    lo = bbox.get("min") or [0, 0, 0]
    hi = bbox.get("max") or [0, 0, 0]
    return tuple(float(hi[i]) - float(lo[i]) for i in range(3))


def dominant_ratio(votes: dict[str, Any]) -> float:
    if not votes:
        return 0.0
    values = [float(v) for v in votes.values()]
    return float(max(values) / max(sum(values), 1.0))


def object_risk(obj: dict[str, Any]) -> tuple[float, list[str]]:
    label = str(obj.get("semantic_label") or "unknown")
    status = str(obj.get("status") or "")
    votes = obj.get("label_vote_weights") or obj.get("label_votes") or {}
    target_count = int(obj.get("target_count") or 0)
    point_count = object_point_count(obj)
    dx, dy, dz = bbox_extent(obj.get("bbox_3d") or {})
    horizontal = math.hypot(dx, dy)
    planarity = float((obj.get("geometry_stats") or {}).get("planarity_mean") or obj.get("planarity") or 0.0)
    linearity = float((obj.get("geometry_stats") or {}).get("linearity_mean") or 0.0)
    ratio = dominant_ratio(votes)

    reasons: list[str] = []
    score = 0.0
    if label == "ambiguous" or status == "ambiguous_object" or ratio < 0.8:
        score += 100.0
        reasons.append("label_vote_conflict")
    if target_count <= 1 and point_count >= 500:
        score += 55.0
        reasons.append("large_single_target_object")
    if label in {"railing", "car"} and point_count >= 10000:
        score += 50.0
        reasons.append("large_fine_object")
    if label == "railing":
        if linearity < 0.45:
            score += 45.0
            reasons.append("railing_not_linear")
        if dz > 1.8 or horizontal > 8.0:
            score += 35.0
            reasons.append("railing_extent_too_large")
    if label == "car":
        if dz < 0.25 or dz > 3.0 or horizontal > 10.0:
            score += 35.0
            reasons.append("car_extent_suspicious")
        if planarity > 0.65 and linearity < 0.2:
            score += 30.0
            reasons.append("car_surface_like")
    if len(votes) > 1:
        score += 10.0 * (len(votes) - 1)
        reasons.append("multiple_label_votes")
    return score, reasons


def candidate_action(label: str) -> str:
    if label == "railing":
        return "split_railing_candidate"
    if label == "car":
        return "split_car_candidate"
    return "split_local_geometry_candidate"


def candidate_row(obj: dict[str, Any], score: float, reasons: list[str]) -> dict[str, Any]:
    label = str(obj.get("semantic_label") or "unknown")
    dx, dy, dz = bbox_extent(obj.get("bbox_3d") or {})
    return {
        "object_id": viewer_object_id(obj),
        "source_object_id": obj.get("object_id"),
        "semantic_label": label,
        "suggested_action": candidate_action(label),
        "score": round(float(score), 3),
        "reasons": reasons,
        "point_count": object_point_count(obj),
        "target_count": int(obj.get("target_count") or 0),
        "metrics": {
            "bbox_extent": [round(float(dx), 4), round(float(dy), 4), round(float(dz), 4)],
            "centroid": [round(float(v), 4) for v in (obj.get("centroid") or [])[:3]],
            "normal": [round(float(v), 4) for v in (obj.get("normal") or [])[:3]],
            "label_votes": obj.get("label_votes") or {},
        },
    }


def select_candidates(objects: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = parse_csv_set(args.labels)
    reason_filter = parse_csv_set(args.require_reasons) or set(DEFAULT_REASONS)
    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for obj in objects:
        label = str(obj.get("semantic_label") or "unknown")
        if labels and label not in labels:
            skipped["label"] += 1
            continue
        points = object_point_count(obj)
        if points < int(args.min_points):
            skipped["min_points"] += 1
            continue
        score, reasons = object_risk(obj)
        matched = sorted(set(reasons) & reason_filter)
        if not matched:
            skipped["reason"] += 1
            continue
        row = candidate_row(obj, score, reasons)
        row["matched_reasons"] = matched
        candidates.append(row)
    candidates.sort(key=lambda row: (-float(row["score"]), -int(row["point_count"]), int(row["object_id"])))
    if args.limit and args.limit > 0:
        candidates = candidates[: int(args.limit)]
    summary = {
        "input_objects": len(objects),
        "selected_candidates": len(candidates),
        "labels": sorted(labels),
        "min_points": int(args.min_points),
        "require_reasons": sorted(reason_filter),
        "skipped": dict(skipped),
        "selected_label_counts": dict(Counter(row["semantic_label"] for row in candidates)),
        "selected_reason_counts": dict(Counter(reason for row in candidates for reason in row.get("matched_reasons", []))),
        "top_candidates": candidates[:20],
    }
    return candidates, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--labels", default="railing")
    parser.add_argument("--min-points", type=int, default=2000)
    parser.add_argument("--require-reasons", default="large_single_target_object,railing_not_linear,railing_extent_too_large")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    candidates, summary = select_candidates(read_jsonl(args.objects_jsonl), args)
    write_jsonl(args.output_jsonl, candidates)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
