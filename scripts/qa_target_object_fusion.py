#!/usr/bin/env python3
"""QA summaries for Target/Object fusion outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_target_reports(path: Path) -> list[dict]:
    if path.is_dir():
        files = sorted((path / "reports").glob("targets_frame_*_report.json"))
        if not files:
            files = sorted(path.glob("targets_frame_*_report.json"))
        return [json.loads(p.read_text(encoding="utf-8")) for p in files]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("frames", [data])


def problematic_objects(objects: list[dict], limit: int) -> dict[str, list[dict]]:
    conflicts = []
    few_targets = []
    high_color_var = []
    low_quality = []
    for obj in objects:
        votes = obj.get("label_votes", {})
        vote_total = sum(int(v) for v in votes.values())
        top = max(votes.values()) if votes else 0
        dominant_ratio = top / max(vote_total, 1)
        base = {
            "object_id": obj.get("object_id"),
            "semantic_label": obj.get("semantic_label"),
            "status": obj.get("status"),
            "target_count": obj.get("target_count", len(obj.get("targets", []))),
            "point_count": obj.get("point_count", 0),
            "dominant_label_ratio": dominant_ratio,
            "label_votes": votes,
        }
        if len(votes) > 1 or dominant_ratio < 0.8:
            conflicts.append(base)
        if base["target_count"] <= 1:
            few_targets.append(base)
        color_var = obj.get("color_stats", {}).get("target_rgb_variance", 0.0)
        if color_var > 1200:
            row = dict(base)
            row["target_rgb_variance"] = color_var
            high_color_var.append(row)
        quality = obj.get("quality_stats", {})
        if quality.get("low_confidence_targets", 0) or quality.get("mixed_targets", 0):
            row = dict(base)
            row["quality_stats"] = quality
            low_quality.append(row)
    conflicts.sort(key=lambda x: (x["dominant_label_ratio"], -x["point_count"]))
    few_targets.sort(key=lambda x: -x["point_count"])
    high_color_var.sort(key=lambda x: -x["target_rgb_variance"])
    low_quality.sort(
        key=lambda x: (
            -int(x.get("quality_stats", {}).get("low_confidence_targets", 0)),
            -int(x.get("quality_stats", {}).get("mixed_targets", 0)),
            -int(x["point_count"]),
        )
    )
    return {
        "multi_label_conflict": conflicts[:limit],
        "single_target_large_objects": few_targets[:limit],
        "high_color_variance": high_color_var[:limit],
        "low_quality_vlm_targets": low_quality[:limit],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--fusion-report", type=Path, default=None)
    parser.add_argument("--zones-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem-limit", type=int, default=20)
    args = parser.parse_args()

    target_reports = load_target_reports(args.target_report)
    objects = read_jsonl(args.objects_jsonl)
    fusion = json.loads(args.fusion_report.read_text(encoding="utf-8")) if args.fusion_report and args.fusion_report.exists() else {}
    zones = json.loads(args.zones_json.read_text(encoding="utf-8")).get("zones", []) if args.zones_json and args.zones_json.exists() else []

    ok_frames = [r for r in target_reports if r.get("status") == "ok"]
    target_count = sum(int(r.get("targets", 0)) for r in ok_frames)
    target_points = sum(int(r.get("target_points", 0)) for r in ok_frames)
    residual_points = sum(int(r.get("small_target_residual_points", 0)) for r in ok_frames)
    label_counts = Counter()
    for row in ok_frames:
        label_counts.update({k: int(v) for k, v in row.get("label_counts", {}).items()})
    status_counts = Counter(obj.get("status", "unknown") for obj in objects)
    object_label_counts = Counter(obj.get("semantic_label", "unknown") for obj in objects)
    target_counts_per_frame = [int(r.get("targets", 0)) for r in ok_frames]
    low_confidence_object_count = sum(
        1 for obj in objects if int(obj.get("quality_stats", {}).get("low_confidence_targets", 0)) > 0
    )
    mixed_object_count = sum(1 for obj in objects if int(obj.get("quality_stats", {}).get("mixed_targets", 0)) > 0)
    confidence_values = [
        float(obj.get("quality_stats", {}).get("confidence_mean", 1.0))
        for obj in objects
        if obj.get("quality_stats")
    ]

    summary = {
        "frames": {
            "count": len(target_reports),
            "ok_count": len(ok_frames),
            "missing_or_failed_count": len(target_reports) - len(ok_frames),
            "avg_targets_per_ok_frame": float(np.mean(target_counts_per_frame)) if target_counts_per_frame else 0.0,
            "max_targets_per_ok_frame": int(max(target_counts_per_frame)) if target_counts_per_frame else 0,
        },
        "targets": {
            "count": int(target_count),
            "points": int(target_points),
            "small_target_residual_points": int(residual_points),
            "small_residual_ratio": float(residual_points / max(target_points + residual_points, 1)),
            "label_counts": dict(label_counts),
        },
        "objects": {
            "count": len(objects),
            "merge_ratio": float(1.0 - len(objects) / max(target_count, 1)),
            "status_counts": dict(status_counts),
            "ambiguous_ratio": float(status_counts.get("ambiguous_object", 0) / max(len(objects), 1)),
            "semantic_label_counts": dict(object_label_counts),
            "low_confidence_object_count": int(low_confidence_object_count),
            "mixed_object_count": int(mixed_object_count),
            "avg_object_vlm_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
        },
        "zones": {
            "count": len(zones),
            "ids": [z.get("zone_id") for z in zones],
        },
        "fusion_report": fusion,
        "problematic_objects": problematic_objects(objects, args.problem_limit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"targets={target_count} objects={len(objects)} ambiguous={summary['objects']['ambiguous_ratio']:.3f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
