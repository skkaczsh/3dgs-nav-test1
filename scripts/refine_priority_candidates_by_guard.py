#!/usr/bin/env python3
"""Geometry/evidence guard for priority fine-object candidates.

The priority segmentation stage is useful for recall but not precise enough to
trust car/railing objects directly. This script is the cheap, deterministic
gate before running a heavier detector/reviewer on image crops:

- keep geometrically plausible candidates for downstream visual confirmation
- send weak/overmerged candidates to visual review
- reject objects that are clearly inconsistent with the claimed class

It does not rewrite point clouds. It writes JSONL manifests that can feed DINO,
GroundingDINO, VLM, or manual review.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extent(obj: dict[str, Any]) -> list[float]:
    val = obj.get("extent")
    if isinstance(val, list) and len(val) >= 3:
        return [float(val[0]), float(val[1]), float(val[2])]
    bbox = obj.get("bbox_3d") or {}
    lo = bbox.get("min") or obj.get("bbox_min") or [0, 0, 0]
    hi = bbox.get("max") or obj.get("bbox_max") or [0, 0, 0]
    return [float(hi[i]) - float(lo[i]) for i in range(3)]


def centroid_z(obj: dict[str, Any]) -> float:
    c = obj.get("centroid") or [0, 0, 0]
    return float(c[2])


def height_layer_name(obj: dict[str, Any]) -> str:
    layer = obj.get("height_layer") or {}
    return str(layer.get("name") or "")


def normal_abs(obj: dict[str, Any]) -> list[float]:
    n = obj.get("pca_normal") or [0, 0, 0]
    return [abs(float(n[0])), abs(float(n[1])), abs(float(n[2]))]


def eigen_ratios(obj: dict[str, Any]) -> tuple[float, float]:
    vals = [float(v) for v in (obj.get("pca_eigenvalues") or [])]
    if len(vals) < 3 or vals[0] <= 1e-9:
        return 0.0, 0.0
    linearity = max(0.0, 1.0 - vals[1] / vals[0])
    planarity = max(0.0, 1.0 - vals[2] / max(vals[1], 1e-9))
    return linearity, planarity


def rank1_evidence(evidence_by_object: dict[int, list[dict[str, Any]]], object_id: int) -> dict[str, Any]:
    rows = evidence_by_object.get(object_id) or []
    for row in rows:
        if int(row.get("rank", 999)) == 1:
            return row
    return rows[0] if rows else {}


def evidence_quality(row: dict[str, Any], image_area: float) -> tuple[str, list[str]]:
    if not row:
        return "missing", ["no_image_evidence"]
    bbox_area = float(row.get("bbox_area") or 0)
    area_ratio = bbox_area / max(image_area, 1.0)
    points = int(row.get("projected_points") or 0)
    reasons = []
    if area_ratio > 0.70:
        reasons.append("huge_image_bbox")
    elif area_ratio > 0.45:
        reasons.append("large_image_bbox")
    if points < 80:
        reasons.append("weak_projection")
    if reasons:
        return "weak", reasons
    return "usable", []


def car_guard(obj: dict[str, Any], ev: dict[str, Any], image_area: float) -> tuple[str, list[str]]:
    ex = extent(obj)
    z = centroid_z(obj)
    layer = height_layer_name(obj)
    planarity = float(obj.get("planarity") or 0.0)
    thickness = float(obj.get("thickness_rms") or 0.0)
    max_extent = float(obj.get("max_extent") or max(ex))
    point_count = int(obj.get("point_count") or 0)
    nabs = normal_abs(obj)
    ev_status, ev_reasons = evidence_quality(ev, image_area)
    reasons: list[str] = []

    if layer.startswith("upper") or z > 5.0:
        reasons.append("car_candidate_in_upper_level_or_high_z")
    if max_extent > 12.0 or point_count > 15000:
        reasons.append("car_candidate_overmerged_large_extent")
    if ex[2] < 0.45:
        reasons.append("car_candidate_too_flat_z_extent")
    if planarity > 0.94 and thickness < 0.08 and nabs[2] > 0.65:
        reasons.append("car_candidate_surface_like_horizontal_panel")
    reasons.extend(ev_reasons)

    if "car_candidate_in_upper_level_or_high_z" in reasons:
        return "geometry_rejected", reasons
    if "car_candidate_surface_like_horizontal_panel" in reasons and ev_status == "weak":
        return "geometry_rejected", reasons
    if reasons:
        return "needs_visual_review", reasons
    if 1.0 <= ex[2] <= 3.2 and 1.8 <= max_extent <= 9.0:
        return "geometry_plausible", ["ground_level_vehicle_sized_geometry"]
    return "needs_visual_review", ["car_geometry_outside_nominal_range"]


def railing_guard(obj: dict[str, Any], ev: dict[str, Any], image_area: float) -> tuple[str, list[str]]:
    ex = extent(obj)
    planarity = float(obj.get("planarity") or 0.0)
    thickness = float(obj.get("thickness_rms") or 0.0)
    max_extent = float(obj.get("max_extent") or max(ex))
    point_count = int(obj.get("point_count") or 0)
    nabs = normal_abs(obj)
    linearity, local_planarity = eigen_ratios(obj)
    ev_status, ev_reasons = evidence_quality(ev, image_area)
    reasons: list[str] = []

    if point_count > 12000 or max_extent > 18.0:
        reasons.append("railing_candidate_overmerged_large_extent")
    if planarity > 0.88 and nabs[2] > 0.82 and thickness > 0.16:
        reasons.append("railing_candidate_horizontal_surface_like")
    if ex[2] > 5.0 and linearity < 0.55:
        reasons.append("railing_candidate_tall_non_linear_surface")
    if max_extent < 1.2:
        reasons.append("railing_candidate_too_small")
    reasons.extend(ev_reasons)

    if "railing_candidate_horizontal_surface_like" in reasons and ev_status == "weak":
        return "geometry_rejected", reasons
    if "railing_candidate_tall_non_linear_surface" in reasons:
        return "geometry_rejected", reasons
    if reasons:
        return "needs_visual_review", reasons
    if linearity > 0.55 or local_planarity > 0.75:
        return "geometry_plausible", ["line_or_guardrail_like_geometry"]
    return "needs_visual_review", ["railing_geometry_not_strongly_linear"]


def guard_object(obj: dict[str, Any], ev: dict[str, Any], image_area: float) -> tuple[str, list[str]]:
    label = str(obj.get("semantic_label") or obj.get("dominant_label") or "")
    if label == "car":
        return car_guard(obj, ev, image_area)
    if label == "railing":
        return railing_guard(obj, ev, image_area)
    return "needs_visual_review", [f"unsupported_priority_label:{label or 'unknown'}"]


def attach_evidence_summary(obj: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
    out = dict(obj)
    if ev:
        out["evidence_rank1"] = {
            "frame_id": ev.get("frame_id"),
            "cam_id": ev.get("cam_id"),
            "bbox_xyxy": ev.get("bbox_xyxy"),
            "bbox_area": ev.get("bbox_area"),
            "projected_points": ev.get("projected_points"),
            "crop_path": ev.get("crop_path"),
            "overlay_path": ev.get("overlay_path"),
            "score": ev.get("score"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=1296)
    args = parser.parse_args()

    objects = read_jsonl(args.objects_jsonl)
    evidence_rows = read_jsonl(args.evidence_jsonl)
    evidence_by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_object[int(row["object_id"])].append(row)
    for rows in evidence_by_object.values():
        rows.sort(key=lambda r: int(r.get("rank", 999)))

    image_area = float(args.image_width * args.image_height)
    outputs: dict[str, list[dict[str, Any]]] = {
        "geometry_plausible": [],
        "needs_visual_review": [],
        "geometry_rejected": [],
    }
    reason_counts = Counter()
    label_status = Counter()
    for obj in objects:
        object_id = int(obj["object_id"])
        ev = rank1_evidence(evidence_by_object, object_id)
        status, reasons = guard_object(obj, ev, image_area)
        row = attach_evidence_summary(obj, ev)
        row["priority_guard_status"] = status
        row["priority_guard_reasons"] = reasons
        row["priority_guard_version"] = "v1"
        outputs[status].append(row)
        label_status[(str(obj.get("semantic_label", "unknown")), status)] += 1
        reason_counts.update(reasons)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for status, rows in outputs.items():
        rows.sort(key=lambda r: (str(r.get("semantic_label", "")), int(r["object_id"])))
        write_jsonl(args.output_dir / f"{status}.jsonl", rows)
        all_rows.extend(rows)
    all_rows.sort(key=lambda r: int(r["object_id"]))
    write_jsonl(args.output_dir / "priority_candidate_guard_all.jsonl", all_rows)

    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "evidence_jsonl": str(args.evidence_jsonl),
        "output_dir": str(args.output_dir),
        "candidate_count": len(objects),
        "evidence_object_count": len(evidence_by_object),
        "status_counts": {k: len(v) for k, v in outputs.items()},
        "label_status_counts": {f"{label}:{status}": count for (label, status), count in sorted(label_status.items())},
        "top_reasons": dict(reason_counts.most_common(30)),
        "outputs": {status: str(args.output_dir / f"{status}.jsonl") for status in outputs},
    }
    (args.output_dir / "priority_candidate_guard_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
