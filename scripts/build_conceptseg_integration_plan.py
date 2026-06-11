#!/usr/bin/env python3
"""Build an integration plan for accepted ConceptSeg instance intersections."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    intersection = row.get("intersection", {})
    return {
        "review_id": row.get("review_id"),
        "target_id": row.get("target_id"),
        "object_id": row.get("object_id"),
        "tracklet_id": row.get("tracklet_id"),
        "frame": row.get("frame"),
        "cam": row.get("cam"),
        "mask": row.get("mask"),
        "semantic": row.get("semantic"),
        "source_label": row.get("source_label"),
        "concept": row.get("concept"),
        "concept_class": row.get("concept_class"),
        "answer": row.get("answer"),
        "answer_class": row.get("answer_class"),
        "bbox": row.get("bbox"),
        "candidate_pixels": intersection.get("candidate_pixels"),
        "instance_pixels": intersection.get("instance_pixels"),
        "intersection_pixels": intersection.get("intersection_pixels"),
        "candidate_inside_instance_ratio": intersection.get("candidate_inside_instance_ratio"),
        "instance_covered_ratio": intersection.get("instance_covered_ratio"),
        "iou": intersection.get("iou"),
        "red_overlay_ratio": row.get("red_overlay_ratio"),
        "output_path": row.get("output_path"),
        "local_assets": row.get("local_assets", {}),
        "remote_assets": row.get("remote_assets", {}),
    }


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = read_json(args.intersection_report)
    side_track = read_json(args.side_track_readiness)
    rows = read_jsonl(args.intersections)
    summaries = read_jsonl(args.target_summary)
    accepted = [row for row in rows if row.get("intersection_accept") is True]
    accepted_compact = [compact_candidate(row) for row in accepted]

    by_concept = Counter(str(row.get("concept_class", "unknown")) for row in accepted)
    by_source_label = Counter(str(row.get("source_label", "unknown")) for row in accepted)
    by_answer = Counter(str(row.get("answer_class", "unknown")) for row in accepted)
    by_object: dict[str, list[str]] = defaultdict(list)
    by_target: dict[str, list[str]] = defaultdict(list)
    for row in accepted:
        if row.get("object_id") is not None:
            by_object[str(row["object_id"])].append(str(row.get("target_id")))
        if row.get("target_id") is not None:
            by_target[str(row["target_id"])].append(str(row.get("concept_class", "unknown")))

    target_status = Counter(str(row.get("status", "unknown")) for row in summaries)
    plan = {
        "plan_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "intersections": file_info(args.intersections),
            "intersection_report": file_info(args.intersection_report),
            "target_summary": file_info(args.target_summary),
            "side_track_readiness": file_info(args.side_track_readiness),
        },
        "decision": "candidate_review_only",
        "role": "second_stage_fine_object_split_refine_candidates",
        "policy": [
            "Do not overwrite dense semantic labels from ConceptSeg candidates.",
            "Intersect ConceptSeg candidate mask with the existing SAM2/Qwen instance mask before use.",
            "Project only the strict intersection to 3D and run connected-component filtering.",
            "Promote only coherent 3D components as child candidates or split proposals.",
            "Keep the original object label unless reviewed evidence supports a refined class.",
        ],
        "summary": {
            "candidate_count": report.get("candidate_count", len(rows)),
            "target_count": report.get("target_count", len(summaries)),
            "accepted_candidate_count": len(accepted),
            "accepted_target_count": len(by_target),
            "accepted_object_count": len(by_object),
            "accepted_tracklet_count": len({str(row.get("tracklet_id")) for row in accepted if row.get("tracklet_id") is not None}),
            "accepted_concept_counts": dict(sorted(by_concept.items())),
            "accepted_source_label_counts": dict(sorted(by_source_label.items())),
            "accepted_answer_counts": dict(sorted(by_answer.items())),
            "target_status_counts": dict(sorted(target_status.items())),
            "side_track_decision": side_track.get("conceptseg_r1", {}).get("decision"),
            "side_track_accepted_target_ratio": side_track.get("conceptseg_r1", {}).get("accepted_target_ratio"),
        },
        "integration_actions": [
            {
                "name": "candidate_mask_intersection",
                "input": "ConceptSeg candidate mask + SAM2/Qwen instance mask",
                "output": "strict accepted 2D pixels",
                "gate": "candidate_inside_instance_ratio >= policy threshold and red_overlay_ratio <= policy threshold",
            },
            {
                "name": "project_intersection_to_frame_points",
                "input": "strict accepted 2D pixels + correct projection route frame points",
                "output": "candidate point subset",
                "gate": "z-buffer nearest visible points only",
            },
            {
                "name": "3d_connected_component_filter",
                "input": "candidate point subset",
                "output": "coherent split/refine component",
                "gate": "minimum point count, local density, and bbox sanity",
            },
            {
                "name": "reviewed_object_update",
                "input": "coherent component + source object",
                "output": "child object or refined fine-object proposal",
                "gate": "manual/VLM review before changing object semantic_label",
            },
        ],
        "risk_assessment": {
            "main_risk": "2D overlap validates compatibility but not 3D coherence.",
            "coverage_limit": "Accepted candidates cover too few targets for dense semantics.",
            "expected_best_use": "thin railings, pipes, and selected equipment candidates in residual/fine-object review.",
        },
        "accepted_candidates": accepted_compact,
    }
    return plan, accepted_compact


def render_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    lines = [
        "# ConceptSeg Integration Plan",
        "",
        f"- decision: `{plan['decision']}`",
        f"- role: `{plan['role']}`",
        f"- accepted candidates: `{summary['accepted_candidate_count']}`",
        f"- accepted targets: `{summary['accepted_target_count']}`",
        f"- accepted objects: `{summary['accepted_object_count']}`",
        f"- accepted concepts: `{summary['accepted_concept_counts']}`",
        f"- accepted source labels: `{summary['accepted_source_label_counts']}`",
        f"- side-track decision: `{summary['side_track_decision']}`",
        f"- side-track accepted target ratio: `{summary['side_track_accepted_target_ratio']}`",
        "",
        "## Policy",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["policy"])
    lines.extend(["", "## Integration Actions", ""])
    for action in plan["integration_actions"]:
        lines.append(f"- `{action['name']}`: {action['output']} ; gate: {action['gate']}")
    lines.extend(["", "## Accepted Candidates", ""])
    for row in plan["accepted_candidates"]:
        lines.append(
            "- "
            f"`{row['target_id']}` object=`{row['object_id']}` frame=`{row['frame']}` cam=`{row['cam']}` "
            f"source=`{row['source_label']}` concept=`{row['concept_class']}` "
            f"inside=`{row['candidate_inside_instance_ratio']}` covered=`{row['instance_covered_ratio']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--intersections", type=Path, default=root / "server_conceptseg_instance_intersection_v008/conceptseg_instance_intersections.jsonl")
    parser.add_argument("--intersection-report", type=Path, default=root / "server_conceptseg_instance_intersection_v008/conceptseg_instance_intersection_report.json")
    parser.add_argument("--target-summary", type=Path, default=root / "server_conceptseg_instance_intersection_v008/conceptseg_instance_target_summary.jsonl")
    parser.add_argument("--side-track-readiness", type=Path, default=root / "route_status_20260610/side_track_readiness_20260611.json")
    parser.add_argument("--output-json", type=Path, default=root / "route_status_20260610/conceptseg_integration_plan_20260611.json")
    parser.add_argument("--output-md", type=Path, default=root / "route_status_20260610/conceptseg_integration_plan_20260611.md")
    parser.add_argument("--accepted-jsonl", type=Path, default=root / "route_status_20260610/conceptseg_accepted_integration_candidates_20260611.jsonl")
    args = parser.parse_args()

    plan, accepted = build_plan(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(plan), encoding="utf-8")
    args.accepted_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
        encoding="utf-8",
    )
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md), "accepted_jsonl": str(args.accepted_jsonl), "accepted": len(accepted)}, indent=2))


if __name__ == "__main__":
    main()
