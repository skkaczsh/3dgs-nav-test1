#!/usr/bin/env python3
"""Diagnose source-mask/frame-target geometry conflicts.

This is a read-only upstream QA stage for the frame-local route.  It inspects
Target JSONL records before object fusion and reports source priority masks that
already disagree with same-frame 3D geometry.  The output is intended to select
bad windows for mask refinement, not to relabel production points directly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FINE_LABELS = {"car", "railing"}
SURFACE_LABELS = {"ground", "wall", "ceiling"}


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
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def bbox_extent(row: dict[str, Any]) -> list[float]:
    bbox = row.get("bbox_3d") or {}
    lo = bbox.get("min") or [0.0, 0.0, 0.0]
    hi = bbox.get("max") or [0.0, 0.0, 0.0]
    return [float(hi[i]) - float(lo[i]) for i in range(3)]


def normal(row: dict[str, Any]) -> list[float]:
    pca = row.get("pca") or {}
    value = pca.get("normal") or row.get("pca_normal") or [0.0, 0.0, 1.0]
    return [float(value[0]), float(value[1]), float(value[2])] if len(value) >= 3 else [0.0, 0.0, 1.0]


def metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    pca = row.get("pca") or {}
    try:
        return float(pca.get(key, row.get(key, default)))
    except (TypeError, ValueError):
        return default


def centroid_z(row: dict[str, Any]) -> float:
    value = row.get("centroid") or [0.0, 0.0, 0.0]
    try:
        return float(value[2])
    except (TypeError, ValueError, IndexError):
        return 0.0


def frame_window(frame_id: int, size: int) -> str:
    if size <= 0:
        return str(frame_id)
    start = (int(frame_id) // size) * size
    end = start + size
    return f"{start:06d}_{end:06d}"


def target_metrics(row: dict[str, Any]) -> dict[str, Any]:
    dims = bbox_extent(row)
    n = normal(row)
    normal_z = abs(float(n[2]))
    horizontal_extent = math.hypot(dims[0], dims[1])
    return {
        "cluster_size": int(row.get("cluster_size") or 0),
        "dims": [round(float(x), 4) for x in dims],
        "horizontal_extent": round(float(horizontal_extent), 4),
        "normal": [round(float(x), 4) for x in n],
        "normal_z_abs": round(float(normal_z), 4),
        "linearity": round(metric(row, "linearity"), 4),
        "planarity": round(metric(row, "planarity"), 4),
        "centroid_z": round(centroid_z(row), 4),
    }


def assess_target(row: dict[str, Any], args: argparse.Namespace) -> tuple[int, list[str], str]:
    label = str(row.get("label") or "unknown")
    dims = bbox_extent(row)
    n = normal(row)
    nz = abs(float(n[2]))
    linearity = metric(row, "linearity")
    planarity = metric(row, "planarity")
    size = int(row.get("cluster_size") or 0)
    horizontal_extent = math.hypot(dims[0], dims[1])
    reasons: list[str] = []
    action = "keep"
    score = 0

    if label == "car":
        if nz >= args.fine_horizontal_normal_z and dims[2] <= args.car_flat_max_z_span:
            reasons.append("car_flat_horizontal_surface")
            score += 70
            action = "mask_review_or_demote_surface"
        if size < args.min_fine_points:
            reasons.append("car_low_point_count")
            score += 25
            action = "visual_review"
        if horizontal_extent > args.car_max_horizontal_extent or dims[2] > args.car_max_z_span:
            reasons.append("car_extent_out_of_range")
            score += 35
            action = "split_or_visual_review"
    elif label == "railing":
        if (
            nz >= args.fine_horizontal_normal_z
            and dims[2] <= args.railing_flat_max_z_span
            and planarity >= args.railing_flat_min_planarity
            and linearity < args.railing_keep_linearity
        ):
            reasons.append("railing_flat_horizontal_surface")
            score += 75
            action = "mask_review_or_demote_surface"
        if size < args.min_fine_points:
            reasons.append("railing_low_point_count")
            score += 25
            action = "visual_review"
        if horizontal_extent > args.railing_max_horizontal_extent or dims[2] > args.railing_max_z_span:
            reasons.append("railing_extent_out_of_range")
            score += 35
            action = "split_or_visual_review"
    elif label == "ground":
        if nz <= args.ground_min_normal_z and planarity >= args.surface_min_planarity:
            reasons.append("ground_not_horizontal")
            score += 60
            action = "surface_split_or_relabel_wall"
        if dims[2] > args.ground_max_z_span and size <= args.ground_height_span_max_points:
            reasons.append("ground_large_z_span")
            score += 30
            action = "height_split_review"
    elif label == "wall":
        if nz >= args.wall_horizontal_normal_z and planarity >= args.surface_min_planarity:
            reasons.append("wall_horizontal_surface")
            score += 55
            action = "surface_split_or_relabel_horizontal"
        if dims[2] < args.wall_min_z_span and horizontal_extent >= args.wall_flat_min_horizontal_extent:
            reasons.append("wall_too_flat")
            score += 35
            action = "surface_split_or_relabel_horizontal"
    elif label == "ceiling":
        if nz < args.ceiling_min_normal_z and planarity >= args.surface_min_planarity:
            reasons.append("ceiling_not_horizontal")
            score += 45
            action = "surface_review"

    if not reasons and label in FINE_LABELS and planarity >= args.fine_surface_like_planarity and nz >= args.fine_surface_like_normal_z:
        reasons.append(f"{label}_surface_like")
        score += 25
        action = "visual_review"

    return score, reasons, action


def diagnose_targets(targets: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    target_label_counts = Counter()
    finding_label_counts = Counter()
    reason_counts = Counter()
    action_counts = Counter()
    window_stats: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "target_count": 0,
            "finding_count": 0,
            "finding_points": 0,
            "score_sum": 0,
            "labels": Counter(),
            "reasons": Counter(),
        }
    )

    for row in targets:
        label = str(row.get("label") or "unknown")
        target_label_counts[label] += 1
        frame_id = int(row.get("frame_id") or 0)
        cam_id = int(row.get("cam_id") or 0)
        window_key = (frame_window(frame_id, args.window_size), cam_id)
        window_stats[window_key]["target_count"] += 1
        score, reasons, action = assess_target(row, args)
        if score < args.min_score:
            continue
        size = int(row.get("cluster_size") or 0)
        finding = {
            "target_id": row.get("target_id"),
            "target_index": row.get("target_index"),
            "frame_id": frame_id,
            "cam_id": cam_id,
            "window": window_key[0],
            "mask_id": row.get("mask_id"),
            "label": label,
            "raw_label": row.get("raw_label"),
            "score": int(score),
            "reasons": reasons,
            "suggested_action": action,
            "metrics": target_metrics(row),
            "image_path": row.get("image_path"),
            "mask_path": row.get("mask_path"),
        }
        findings.append(finding)
        finding_label_counts[label] += 1
        reason_counts.update(reasons)
        action_counts[action] += 1
        stats = window_stats[window_key]
        stats["finding_count"] += 1
        stats["finding_points"] += size
        stats["score_sum"] += int(score)
        stats["labels"][label] += 1
        stats["reasons"].update(reasons)

    findings.sort(
        key=lambda row: (
            -int(row["score"]),
            -int(row["metrics"]["cluster_size"]),
            int(row["frame_id"]),
            int(row["cam_id"]),
            str(row["target_id"]),
        )
    )
    top_windows = []
    for (window, cam_id), stats in window_stats.items():
        if int(stats["finding_count"]) <= 0:
            continue
        top_windows.append(
            {
                "window": window,
                "cam_id": cam_id,
                "target_count": int(stats["target_count"]),
                "finding_count": int(stats["finding_count"]),
                "finding_points": int(stats["finding_points"]),
                "score_sum": int(stats["score_sum"]),
                "labels": dict(stats["labels"].most_common()),
                "reasons": dict(stats["reasons"].most_common()),
            }
        )
    top_windows.sort(key=lambda row: (-int(row["score_sum"]), -int(row["finding_points"]), row["window"], row["cam_id"]))

    report = {
        "target_count": len(targets),
        "finding_count": len(findings),
        "target_label_counts": dict(target_label_counts.most_common()),
        "finding_label_counts": dict(finding_label_counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "suggested_action_counts": dict(action_counts.most_common()),
        "top_windows": top_windows[: args.top_windows],
        "top_findings": findings[: args.top_findings],
        "params": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, list, type(None)))
        },
    }
    return findings, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--min-score", type=int, default=25)
    parser.add_argument("--top-windows", type=int, default=30)
    parser.add_argument("--top-findings", type=int, default=50)
    parser.add_argument("--min-fine-points", type=int, default=80)
    parser.add_argument("--fine-horizontal-normal-z", type=float, default=0.92)
    parser.add_argument("--fine-surface-like-normal-z", type=float, default=0.85)
    parser.add_argument("--fine-surface-like-planarity", type=float, default=0.55)
    parser.add_argument("--car-flat-max-z-span", type=float, default=0.25)
    parser.add_argument("--car-max-z-span", type=float, default=3.0)
    parser.add_argument("--car-max-horizontal-extent", type=float, default=10.0)
    parser.add_argument("--railing-flat-max-z-span", type=float, default=0.25)
    parser.add_argument("--railing-flat-min-planarity", type=float, default=0.35)
    parser.add_argument("--railing-keep-linearity", type=float, default=0.82)
    parser.add_argument("--railing-max-z-span", type=float, default=2.2)
    parser.add_argument("--railing-max-horizontal-extent", type=float, default=8.0)
    parser.add_argument("--surface-min-planarity", type=float, default=0.35)
    parser.add_argument("--ground-min-normal-z", type=float, default=0.55)
    parser.add_argument("--ground-max-z-span", type=float, default=0.9)
    parser.add_argument("--ground-height-span-max-points", type=int, default=5000)
    parser.add_argument("--wall-horizontal-normal-z", type=float, default=0.75)
    parser.add_argument("--wall-min-z-span", type=float, default=0.4)
    parser.add_argument("--wall-flat-min-horizontal-extent", type=float, default=1.5)
    parser.add_argument("--ceiling-min-normal-z", type=float, default=0.75)
    args = parser.parse_args()

    findings, report = diagnose_targets(read_jsonl(args.targets_jsonl), args)
    write_jsonl(args.output_jsonl, findings)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "target_count": report["target_count"],
        "finding_count": report["finding_count"],
        "finding_label_counts": report["finding_label_counts"],
        "top_windows": report["top_windows"][:5],
        "output_jsonl": str(args.output_jsonl),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
