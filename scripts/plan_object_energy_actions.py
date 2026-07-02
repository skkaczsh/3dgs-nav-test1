#!/usr/bin/env python3
"""Turn object-energy QA into a deterministic action queue.

This does not modify point ownership or semantic labels.  It converts high-risk
QA flags into a small set of next operations so patch/object work can proceed
from measured bottlenecks instead of ad-hoc visual impressions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLIT_FLAGS = {
    "surface_label_on_mixed_geometry",
    "low_bucket_purity_large_object",
    "high_bucket_entropy_large_object",
}
OVERLAP_FLAGS = {"coarse_voxel_overlap_with_other_object"}
SEMANTIC_FLAGS = {
    "railing_label_without_linear_support",
    "grass_label_without_rough_or_horizontal_support",
    "teacher_top_vote_differs_from_label",
    "fine_label_has_large_scene_extent",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def classify_action(row: dict[str, Any]) -> tuple[str, str]:
    flags = set(row.get("flags") or [])
    if flags & OVERLAP_FLAGS and flags & SPLIT_FLAGS:
        return "split_then_overlap_review", "mixed object also overlaps another object"
    if flags & SPLIT_FLAGS:
        return "split_geometry_mixed_object", "object contains multiple geometry buckets"
    if flags & OVERLAP_FLAGS:
        return "review_overlap_pair", "object shares coarse occupied voxels with another object"
    if flags & SEMANTIC_FLAGS:
        return "semantic_review_only", "geometry ownership is not the primary issue"
    return "monitor", "no high-risk action"


def priority_score(row: dict[str, Any], action: str) -> float:
    base = float(row.get("energy_score") or 0.0)
    voxels = max(int(row.get("voxel_count") or 0), 1)
    size_bonus = min(1.0, voxels / 50000.0)
    action_bonus = {
        "split_then_overlap_review": 2.0,
        "split_geometry_mixed_object": 1.4,
        "review_overlap_pair": 1.1,
        "semantic_review_only": 0.4,
        "monitor": 0.0,
    }.get(action, 0.0)
    return base + size_bonus + action_bonus


def build_action(row: dict[str, Any]) -> dict[str, Any]:
    action, reason = classify_action(row)
    return {
        "object_id": int(row["object_id"]),
        "action": action,
        "priority_score": round(priority_score(row, action), 6),
        "reason": reason,
        "semantic_label": row.get("semantic_label"),
        "geometry_type": row.get("geometry_type"),
        "voxel_count": int(row.get("voxel_count") or 0),
        "patch_count": int(row.get("patch_count") or 0),
        "bucket_purity": float(row.get("bucket_purity") or 0.0),
        "bucket_entropy": float(row.get("bucket_entropy") or 0.0),
        "dominant_bucket": row.get("dominant_bucket"),
        "bucket_ratios": row.get("bucket_ratios") or {},
        "overlap_max_intersection_over_min": float(row.get("overlap_max_intersection_over_min") or 0.0),
        "flags": row.get("flags") or [],
        "source_energy_score": float(row.get("energy_score") or 0.0),
    }


def plan_actions(
    qa_report: dict[str, Any],
    *,
    top_n: int,
    include_monitor: bool,
    min_priority_score: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    objects = list(qa_report.get("top_problem_objects") or [])
    actions = [build_action(row) for row in objects]
    if not include_monitor:
        actions = [row for row in actions if row["action"] != "monitor"]
    if min_priority_score > 0:
        actions = [row for row in actions if float(row["priority_score"]) >= min_priority_score]
    actions.sort(key=lambda row: (-float(row["priority_score"]), -int(row["voxel_count"]), int(row["object_id"])))
    if top_n > 0:
        actions = actions[:top_n]

    action_counts = Counter(row["action"] for row in actions)
    flag_counts = Counter(flag for row in actions for flag in row.get("flags", []))
    report = {
        "schema": "object-energy-action-plan/v1",
        "source_schema": qa_report.get("schema"),
        "source_objects_jsonl": qa_report.get("objects_jsonl"),
        "source_ply": qa_report.get("ply"),
        "source_high_risk_object_count": qa_report.get("high_risk_object_count"),
        "action_count": len(actions),
        "action_counts": dict(action_counts),
        "flag_counts": dict(flag_counts.most_common()),
        "top_actions": actions[:30],
    }
    return actions, report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Object Energy Action Plan",
        "",
        f"Actions: `{report['action_count']}`",
        f"Source high-risk objects: `{report.get('source_high_risk_object_count')}`",
        "",
        "## Action Counts",
        "",
    ]
    for key, value in report["action_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Actions", ""])
    lines.append("| object | action | label | geom | voxels | priority | flags |")
    lines.append("|---:|---|---|---|---:|---:|---|")
    for row in report["top_actions"]:
        flags = ", ".join(row["flags"][:5])
        lines.append(
            f"| {row['object_id']} | {row['action']} | {row['semantic_label']} | "
            f"{row['geometry_type']} | {row['voxel_count']} | {row['priority_score']:.3f} | {flags} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--include-monitor", action="store_true")
    parser.add_argument("--min-priority-score", type=float, default=0.0)
    args = parser.parse_args()

    actions, report = plan_actions(
        read_json(args.qa_report),
        top_n=args.top_n,
        include_monitor=args.include_monitor,
        min_priority_score=args.min_priority_score,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "object_energy_actions.jsonl", actions)
    (args.output_dir / "object_energy_action_plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "object_energy_action_plan.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
