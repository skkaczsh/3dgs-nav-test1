#!/usr/bin/env python3
"""Report priority-object semantic labels that conflict with 3D geometry.

This is a read-only QA stage. It does not rewrite points or labels. The output
is intended to drive the next deterministic refinement pass before any heavier
visual reviewer is trusted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SURFACE_LABELS = {"floor", "wall", "grass"}
FINE_LABELS = {"car", "railing"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def vec3(obj: dict[str, Any], key: str, default: list[float]) -> list[float]:
    value = obj.get(key)
    if isinstance(value, list) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return list(default)


def extent(obj: dict[str, Any]) -> list[float]:
    value = obj.get("extent")
    if isinstance(value, list) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    bbox = obj.get("bbox_3d") if isinstance(obj.get("bbox_3d"), dict) else {}
    lo = bbox.get("min") or obj.get("bbox_min") or [0.0, 0.0, 0.0]
    hi = bbox.get("max") or obj.get("bbox_max") or [0.0, 0.0, 0.0]
    return [float(hi[i]) - float(lo[i]) for i in range(3)]


def centroid_z(obj: dict[str, Any]) -> float:
    return vec3(obj, "centroid", [0.0, 0.0, 0.0])[2]


def normal_z_abs(obj: dict[str, Any]) -> float:
    normal = vec3(obj, "pca_normal", [0.0, 0.0, 0.0])
    return abs(normal[2])


def orientation(obj: dict[str, Any], horizontal_z: float, vertical_z: float) -> str:
    nz = normal_z_abs(obj)
    if nz >= horizontal_z:
        return "horizontal"
    if nz <= vertical_z:
        return "vertical"
    return "oblique"


def metric_summary(obj: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ex = extent(obj)
    return {
        "point_count": int(obj.get("point_count") or 0),
        "centroid_z": centroid_z(obj),
        "extent": ex,
        "max_extent": float(obj.get("max_extent") or max(ex)),
        "z_extent": ex[2],
        "normal_z_abs": normal_z_abs(obj),
        "orientation": orientation(obj, args.horizontal_normal_z, args.vertical_normal_z),
        "planarity": float(obj.get("planarity") or 0.0),
        "thickness_rms": float(obj.get("thickness_rms") or 0.0),
    }


def assess_object(obj: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str], str]:
    label = str(obj.get("semantic_label") or "unknown")
    metrics = metric_summary(obj, args)
    reasons: list[str] = []
    suggested_action = "keep"

    if label == "floor":
        if metrics["orientation"] != "horizontal":
            reasons.append("floor_not_horizontal")
            suggested_action = "surface_split_or_relabel"
        if metrics["z_extent"] > args.floor_max_z_extent:
            reasons.append("floor_large_vertical_extent")
            suggested_action = "split_floor_by_height_or_visibility"
    elif label == "wall":
        if metrics["orientation"] == "horizontal":
            reasons.append("wall_has_horizontal_normal")
            suggested_action = "split_wall_or_relabel_horizontal_surface"
        elif metrics["orientation"] == "oblique":
            reasons.append("wall_oblique_normal")
            suggested_action = "wall_geometry_review"
        if metrics["planarity"] < args.surface_min_planarity:
            reasons.append("wall_low_planarity")
            suggested_action = "split_mixed_wall_component"
        if metrics["thickness_rms"] > args.wall_max_thickness:
            reasons.append("wall_high_thickness")
            suggested_action = "split_mixed_wall_component"
    elif label == "grass":
        if metrics["z_extent"] > args.grass_max_z_extent:
            reasons.append("grass_large_vertical_extent")
            suggested_action = "split_grass_or_relabel_mixed_object"
        if metrics["planarity"] < args.grass_min_planarity:
            reasons.append("grass_low_planarity")
            suggested_action = "grass_geometry_review"
    elif label == "car":
        status = str(obj.get("priority_guard_status") or "")
        if status == "geometry_rejected":
            reasons.append("car_geometry_rejected")
            suggested_action = "demote_to_unknown"
        if centroid_z(obj) > args.car_max_centroid_z:
            reasons.append("car_high_centroid_z")
            suggested_action = "demote_or_visual_review"
        if metrics["max_extent"] > args.car_max_extent:
            reasons.append("car_overmerged_extent")
            suggested_action = "split_car_candidate"
        if metrics["z_extent"] < args.car_min_z_extent:
            reasons.append("car_too_flat")
            suggested_action = "demote_or_visual_review"
    elif label == "railing":
        status = str(obj.get("priority_guard_status") or "")
        if status == "geometry_rejected":
            reasons.append("railing_geometry_rejected")
            suggested_action = "demote_to_unknown"
        if metrics["orientation"] == "horizontal" and metrics["thickness_rms"] > args.railing_max_horizontal_thickness:
            reasons.append("railing_surface_like_horizontal")
            suggested_action = "split_or_demote_railing_candidate"
        if metrics["max_extent"] > args.railing_max_extent:
            reasons.append("railing_overmerged_extent")
            suggested_action = "split_railing_candidate"

    if not reasons and label in SURFACE_LABELS and metrics["point_count"] < args.tiny_surface_points:
        reasons.append("tiny_surface_fragment")
        suggested_action = "merge_with_neighbor_or_demote"

    severity = "ok"
    if reasons:
        severity = "high" if any("rejected" in r or "horizontal_normal" in r or "large_vertical" in r for r in reasons) else "medium"
    return severity, reasons, suggested_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizontal-normal-z", type=float, default=0.85)
    parser.add_argument("--vertical-normal-z", type=float, default=0.35)
    parser.add_argument("--surface-min-planarity", type=float, default=0.70)
    parser.add_argument("--wall-max-thickness", type=float, default=1.50)
    parser.add_argument("--floor-max-z-extent", type=float, default=4.00)
    parser.add_argument("--grass-max-z-extent", type=float, default=8.00)
    parser.add_argument("--grass-min-planarity", type=float, default=0.55)
    parser.add_argument("--car-max-centroid-z", type=float, default=5.00)
    parser.add_argument("--car-max-extent", type=float, default=12.00)
    parser.add_argument("--car-min-z-extent", type=float, default=0.45)
    parser.add_argument("--railing-max-extent", type=float, default=18.00)
    parser.add_argument("--railing-max-horizontal-thickness", type=float, default=0.16)
    parser.add_argument("--tiny-surface-points", type=int, default=500)
    args = parser.parse_args()

    rows = read_jsonl(args.objects_jsonl)
    findings: list[dict[str, Any]] = []
    all_status = Counter()
    label_status = Counter()
    reason_counts = Counter()
    action_counts = Counter()
    point_counts_by_severity = Counter()

    for obj in rows:
        label = str(obj.get("semantic_label") or "unknown")
        severity, reasons, action = assess_object(obj, args)
        all_status[severity] += 1
        label_status[(label, severity)] += 1
        reason_counts.update(reasons)
        action_counts[action] += 1
        point_counts_by_severity[severity] += int(obj.get("point_count") or 0)
        if severity != "ok":
            item = {
                "object_id": int(obj.get("object_id") or 0),
                "semantic_label": label,
                "severity": severity,
                "reasons": reasons,
                "suggested_action": action,
                "metrics": metric_summary(obj, args),
                "priority_guard_status": obj.get("priority_guard_status"),
                "priority_guard_reasons": obj.get("priority_guard_reasons"),
                "visual_review_status": obj.get("visual_review_status"),
                "visual_review_best_phrase": obj.get("visual_review_best_phrase"),
                "description": obj.get("description"),
            }
            findings.append(item)

    findings.sort(key=lambda row: (-int(row["metrics"]["point_count"]), row["semantic_label"], row["object_id"]))
    write_jsonl(args.output_jsonl, findings)
    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "object_count": len(rows),
        "finding_count": len(findings),
        "severity_counts": dict(all_status),
        "label_severity_counts": {f"{label}:{status}": count for (label, status), count in sorted(label_status.items())},
        "point_counts_by_severity": dict(point_counts_by_severity),
        "top_reasons": dict(reason_counts.most_common(30)),
        "suggested_action_counts": dict(action_counts.most_common()),
        "top_findings": findings[:25],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
