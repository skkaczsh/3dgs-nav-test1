#!/usr/bin/env python3
"""Summarize target/object semantic pipeline QA into actionable tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def top(rows: list[dict], key: str, n: int) -> list[dict]:
    return sorted(rows, key=lambda r: r.get(key, 0), reverse=True)[:n]


def summarize_objects(objects_jsonl: Path, top_n: int) -> dict:
    status_counts = Counter()
    label_counts = Counter()
    point_by_status = Counter()
    point_by_label = Counter()
    ambiguous = []
    single = []
    low_vote = []
    with objects_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            status = obj.get("status", "unknown")
            label = obj.get("semantic_label", "unknown")
            points = int(obj.get("point_count", 0))
            targets = int(obj.get("target_count", 0))
            ratio = float(obj.get("dominant_label_ratio", 0.0))
            row = {
                "object_id": obj.get("object_id"),
                "status": status,
                "semantic_label": label,
                "dominant_label": obj.get("dominant_label", label),
                "dominant_label_ratio": ratio,
                "point_count": points,
                "target_count": targets,
                "label_votes": obj.get("label_votes", {}),
                "bbox_3d": obj.get("bbox_3d", {}),
                "centroid": obj.get("centroid", []),
            }
            status_counts[status] += 1
            label_counts[label] += 1
            point_by_status[status] += points
            point_by_label[label] += points
            if status == "ambiguous_object":
                ambiguous.append(row)
            if status == "single_target":
                single.append(row)
            if targets > 1 and ratio < 0.9:
                low_vote.append(row)
    return {
        "status_counts": dict(status_counts),
        "label_counts": dict(label_counts),
        "point_by_status": dict(point_by_status),
        "point_by_label": dict(point_by_label),
        "top_ambiguous_by_points": top(ambiguous, "point_count", top_n),
        "top_single_target_by_points": top(single, "point_count", top_n),
        "top_low_vote_by_points": top(low_vote, "point_count", top_n),
    }


def summarize_target_report(path: Path, top_n: int) -> dict:
    report = load_json(path)
    frames = []
    residual_labels = Counter()
    for row in report.get("frames", []):
        if row.get("status") != "ok":
            continue
        target_points = int(row.get("target_points", 0))
        residual_points = int(row.get("small_target_residual_points", 0))
        total = target_points + residual_points
        frame_row = {
            "frame": int(row.get("frame_id", row.get("frame", -1))),
            "targets": int(row.get("targets", 0)),
            "target_points": target_points,
            "small_target_residual_points": residual_points,
            "small_residual_ratio": float(residual_points / max(total, 1)),
            "label_counts": row.get("label_counts", {}),
            "small_target_residual_label_counts": row.get("small_target_residual_label_counts", {}),
        }
        frames.append(frame_row)
        residual_labels.update({k: int(v) for k, v in row.get("small_target_residual_label_counts", {}).items()})
    return {
        "summary": report.get("summary", {}),
        "residual_label_counts": dict(residual_labels),
        "top_residual_frames": top(frames, "small_target_residual_points", top_n),
        "top_residual_ratio_frames": top(frames, "small_residual_ratio", top_n),
    }


def summarize_semantic_projection(path: Path, top_n: int) -> dict:
    report = load_json(path)
    frames = [
        {
            "frame": int(row.get("frame", -1)),
            "labeled_ratio": float(row.get("labeled_ratio", 0.0)),
            "labeled_points": int(row.get("labeled_points", 0)),
            "points": int(row.get("points", 0)),
            "label_counts": row.get("label_counts", {}),
        }
        for row in report.get("frames", [])
        if row.get("status") == "ok"
    ]
    return {
        "summary": report.get("summary", {}),
        "lowest_labeled_ratio_frames": sorted(frames, key=lambda r: r["labeled_ratio"])[:top_n],
    }


def summarize_residual_assignment(path: Path) -> dict:
    report = load_json(path)
    by_label = {k: int(v) for k, v in report.get("by_label", {}).items()}
    assigned_by_label = {k: int(v) for k, v in report.get("assigned_by_label", {}).items()}
    unassigned_by_label = {
        label: int(count - assigned_by_label.get(label, 0))
        for label, count in by_label.items()
    }
    return {
        "residual_points": int(report.get("residual_points", 0)),
        "assigned_points": int(report.get("assigned_points", 0)),
        "assigned_ratio": float(report.get("assigned_ratio", 0.0)),
        "by_label": by_label,
        "assigned_by_label": assigned_by_label,
        "unassigned_by_label": unassigned_by_label,
        "top_unassigned_labels": sorted(unassigned_by_label.items(), key=lambda kv: kv[1], reverse=True),
    }


def build_recommendations(summary: dict) -> list[str]:
    recs = []
    residual = summary.get("residual_assignment", {})
    unassigned = dict(residual.get("top_unassigned_labels", []))
    if unassigned.get("floor", 0) > 0:
        recs.append("floor remains the largest unassigned residual source; tune surface object absorption before changing 2D labels.")
    if unassigned.get("building", 0) > 0:
        recs.append("building/wall residual remains significant; inspect vertical-plane thresholds and wall/building label compatibility.")
    objects = summary.get("objects", {})
    status = objects.get("status_counts", {})
    if status.get("ambiguous_object", 0):
        recs.append("review top ambiguous objects by point_count first; they dominate semantic conflict more than small objects.")
    if status.get("single_target", 0):
        recs.append("large single-target objects should be checked for missed cross-frame merges before expanding frame count.")
    semantic = summary.get("semantic_projection", {}).get("summary", {})
    if semantic.get("avg_labeled_ratio", 0) >= 0.9:
        recs.append("2D-to-point coverage is high enough for this stage; next gains should come from object fusion and residual absorption.")
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--semantic-projection-report", type=Path, required=True)
    parser.add_argument("--residual-assignment-report", type=Path, required=True)
    parser.add_argument("--consolidated-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    summary = {
        "objects": summarize_objects(args.objects_jsonl, args.top_n),
        "targets": summarize_target_report(args.target_report, args.top_n),
        "semantic_projection": summarize_semantic_projection(args.semantic_projection_report, args.top_n),
        "residual_assignment": summarize_residual_assignment(args.residual_assignment_report),
        "consolidated": load_json(args.consolidated_report),
        "validation": load_json(args.validation_report),
    }
    summary["recommendations"] = build_recommendations(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "validation": summary["validation"].get("status"),
        "objects": summary["objects"].get("status_counts", {}),
        "residual_assignment": {
            "assigned_ratio": summary["residual_assignment"].get("assigned_ratio"),
            "top_unassigned_labels": summary["residual_assignment"].get("top_unassigned_labels", [])[:5],
        },
        "recommendations": summary["recommendations"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
