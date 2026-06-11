#!/usr/bin/env python3
"""Summarize new-model and old-route side-track readiness from artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    concept_qa = read_json(args.conceptseg_qa)
    concept_align = read_json(args.conceptseg_alignment)
    concept_intersection = read_json(args.conceptseg_intersection)
    old_validation = read_json(args.old_route_validation)
    old_summary = read_json(args.old_route_summary)

    concept_items = int(concept_qa.get("items", 0) or 0)
    concept_success = int(concept_qa.get("returncode_counts", {}).get("0", 0) or 0)
    accepted = int(concept_intersection.get("accepted_candidate_count", 0) or 0)
    target_count = int(concept_intersection.get("target_count", 0) or 0)
    targets_with_accept = int(concept_intersection.get("target_status_counts", {}).get("has_intersection_candidate", 0) or 0)
    sem_discriminative = int(concept_align.get("semantically_discriminative_target_count", 0) or 0)
    aligned_targets = int(concept_align.get("target_count", 0) or 0)

    concept_decision = "candidate_generator_only"
    if sem_discriminative > 0:
        concept_decision = "requires_manual_review_before_classifier_use"
    if accepted <= 0:
        concept_decision = "not_ready"

    old_passed = bool(old_validation.get("passed", False))
    old_colored_ratio = float(old_validation.get("colored_ratio", 0.0) or 0.0)
    old_decision = "visual_color_reference_only" if old_passed and old_colored_ratio >= 0.8 else "not_ready"

    report = {
        "summary_version": 1,
        "inputs": {
            "conceptseg_qa": file_info(args.conceptseg_qa),
            "conceptseg_alignment": file_info(args.conceptseg_alignment),
            "conceptseg_intersection": file_info(args.conceptseg_intersection),
            "old_route_validation": file_info(args.old_route_validation),
            "old_route_summary": file_info(args.old_route_summary),
        },
        "conceptseg_r1": {
            "decision": concept_decision,
            "role": "constrained_second_stage_fine_object_candidate_generator",
            "items": concept_items,
            "success_count": concept_success,
            "success_ratio": ratio(concept_success, concept_items),
            "mode_counts": concept_qa.get("mode_counts", {}),
            "concept_summary": concept_qa.get("concept_summary", {}),
            "aligned_targets": aligned_targets,
            "aligned_objects": concept_align.get("object_count"),
            "concept_match_count": concept_align.get("concept_match_count"),
            "semantically_discriminative_targets": sem_discriminative,
            "accepted_intersection_candidates": accepted,
            "targets_with_accepted_candidates": targets_with_accept,
            "accepted_target_ratio": ratio(targets_with_accept, target_count),
            "accepted_concept_counts": concept_intersection.get("accepted_concept_counts", {}),
            "promotion_policy": [
                "Do not use ConceptSeg-R1 as dense semantic source.",
                "Use only after strict intersection with existing SAM2/Qwen instance masks.",
                "Apply 3D connected-component filtering before point-level fusion.",
            ],
        },
        "old_route": {
            "decision": old_decision,
            "role": "fixed_visual_color_reference_only",
            "passed": old_passed,
            "colored_ratio": old_colored_ratio,
            "sections": old_validation.get("sections"),
            "source_points": old_validation.get("source_points"),
            "fused_points": old_validation.get("fused_points"),
            "color_frames": old_validation.get("color_frames"),
            "sample_mode": old_validation.get("sample_mode"),
            "sample_radius": old_validation.get("sample_radius"),
            "fusion_mode": old_validation.get("fusion_mode"),
            "summary_output": old_summary.get("output"),
            "promotion_policy": [
                "Do not revive transforms.json + project_world_points semantic route.",
                "Use as color/geometric sanity reference for scanner-native projection.",
                "Reconstruct a reproducible runner only from img_pos.txt + cam_in_ex.txt + Tcl + Til.",
            ],
        },
        "next_actions": [
            "Integrate accepted ConceptSeg candidates only into fine-object split/refine review, not dense labels.",
            "Use hybrid strict-surface object PLY as current main viewer artifact.",
            "Keep old-route smoke as a fixed reference until a scanner-native reproducible runner is rebuilt.",
        ],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    concept = report["conceptseg_r1"]
    old = report["old_route"]
    lines = [
        "# Side Track Readiness",
        "",
        "## ConceptSeg-R1",
        "",
        f"- decision: `{concept['decision']}`",
        f"- role: `{concept['role']}`",
        f"- success: `{concept['success_count']}` / `{concept['items']}`",
        f"- accepted intersection candidates: `{concept['accepted_intersection_candidates']}`",
        f"- targets with accepted candidates: `{concept['targets_with_accepted_candidates']}`",
        f"- accepted target ratio: `{concept['accepted_target_ratio']:.4f}`",
        f"- semantically discriminative targets: `{concept['semantically_discriminative_targets']}`",
        f"- accepted concept counts: `{concept['accepted_concept_counts']}`",
        "",
        "## Old Route",
        "",
        f"- decision: `{old['decision']}`",
        f"- role: `{old['role']}`",
        f"- validation passed: `{old['passed']}`",
        f"- colored ratio: `{old['colored_ratio']:.4f}`",
        f"- sections/source/fused points: `{old['sections']}` / `{old['source_points']}` / `{old['fused_points']}`",
        f"- color frames: `{old['color_frames']}`",
        f"- fusion mode: `{old['fusion_mode']}`",
        "",
        "## Next Actions",
        "",
    ]
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--conceptseg-qa", type=Path, default=root / "server_conceptseg_fine_object_runlist_v008_outputs_all/conceptseg_fine_object_all_qa.json")
    parser.add_argument("--conceptseg-alignment", type=Path, default=root / "server_conceptseg_fine_object_alignment_v008/conceptseg_target_object_alignment_report.json")
    parser.add_argument("--conceptseg-intersection", type=Path, default=root / "server_conceptseg_instance_intersection_v008/conceptseg_instance_intersection_report.json")
    parser.add_argument("--old-route-validation", type=Path, default=root / "server_old_route_smoke/old_route_reference_validation.json")
    parser.add_argument("--old-route-summary", type=Path, default=root / "server_old_route_smoke/world_colorize_summary.json")
    parser.add_argument("--output-json", type=Path, default=root / "route_status_20260610/side_track_readiness_20260611.json")
    parser.add_argument("--output-md", type=Path, default=root / "route_status_20260610/side_track_readiness_20260611.md")
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
