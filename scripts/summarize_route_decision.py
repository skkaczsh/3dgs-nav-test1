#!/usr/bin/env python3
"""Summarize current dense semantic route decisions from validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exists_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    main = summary["main_route"]
    concept = summary["conceptseg_side_track"]
    old = summary["old_route_side_track"]
    next_steps = summary["next_steps"]
    lines = [
        "# Dense Semantic Route Decision",
        "",
        "## Decision",
        "",
        f"- Main route status: `{main['decision']}`.",
        f"- ConceptSeg-R1 status: `{concept['decision']}`.",
        f"- Old route status: `{old['decision']}`.",
        "",
        "## Main Route Evidence",
        "",
        f"- Dataset manifest passed: `{main['dataset_manifest_passed']}`",
        f"- Output validation passed: `{main['output_validation_passed']}`",
        f"- Frame range: `{main['frame_range'][0]}-{main['frame_range'][1]}`",
        f"- Semantic combo: `{main['semantic_combo']}`",
        f"- Projection route: `{main['projection_route']}`",
        f"- Target count: `{main['target_count']}`",
        f"- Object count: `{main['object_count']}`",
        f"- Object ambiguous ratio: `{main['object_ambiguous_ratio']:.4f}`",
        f"- Surface-first changed ratio: `{main['surface_first_changed_ratio']:.4f}`",
        f"- Residual surface assignment ratio: `{main['residual_surface_assigned_ratio']:.4f}`",
        f"- Residual surface unassigned points: `{main['residual_surface_unassigned_points']}`",
        f"- Residual absorption sweep best ratio: `{main['residual_absorption_sweep_best_ratio']:.4f}`",
        f"- Residual miss reasons: `{main['residual_surface_miss_reason_counts']}`",
        f"- Residual candidate coverage best ratio: `{main['residual_candidate_coverage_best_ratio']:.4f}`",
        f"- Surface seed augmented best ratio: `{main['residual_candidate_coverage_augmented_best_ratio']:.4f}`",
        "",
        "## ConceptSeg-R1 Evidence",
        "",
        f"- Candidate runs: `{concept['candidate_count']}`",
        f"- Aligned targets: `{concept['aligned_target_count']}`",
        f"- Concept matches: `{concept['concept_match_count']}`",
        f"- Semantically discriminative targets: `{concept['semantically_discriminative_target_count']}`",
        f"- Instance-intersection accepted candidates: `{concept['intersection_accepted_candidate_count']}`",
        f"- Instance-intersection target coverage: `{concept['intersection_target_coverage']}`",
        f"- Conclusion: {concept['conclusion']}",
        "",
        "## Old Route Evidence",
        "",
        f"- Reference validation passed: `{old['validation_passed']}`",
        f"- Colored ratio: `{old['colored_ratio']:.4f}`",
        f"- PLY vertices: `{old['ply_vertex_count']}`",
        f"- RGB fields present: `{old['ply_has_rgb']}`",
        f"- Conclusion: {old['conclusion']}",
        "",
        "## Next Steps",
        "",
    ]
    lines.extend(f"- {step}" for step in next_steps)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    parser.add_argument("--conceptseg-alignment", type=Path, required=True)
    parser.add_argument("--conceptseg-intersection", type=Path, required=True)
    parser.add_argument("--old-route-validation", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    dataset = read_json(args.dataset_manifest)
    output_validation = read_json(args.output_validation)
    concept_align = read_json(args.conceptseg_alignment)
    concept_inter = read_json(args.conceptseg_intersection)
    old_validation = read_json(args.old_route_validation)

    metrics = dataset.get("metrics", {})
    frame_range = dataset.get("dataset", {}).get("frame_range", [None, None])
    target_status_counts = concept_inter.get("target_status_counts", {})
    inter_target_coverage = f"{target_status_counts.get('has_intersection_candidate', 0)} / {concept_inter.get('target_count', 0)}"
    summary = {
        "summary_version": 1,
        "inputs": {
            "dataset_manifest": exists_info(args.dataset_manifest),
            "output_validation": exists_info(args.output_validation),
            "conceptseg_alignment": exists_info(args.conceptseg_alignment),
            "conceptseg_intersection": exists_info(args.conceptseg_intersection),
            "old_route_validation": exists_info(args.old_route_validation),
        },
        "main_route": {
            "decision": "continue_as_authoritative_route",
            "dataset_manifest_passed": bool(dataset.get("passed", False)),
            "output_validation_passed": bool(output_validation.get("passed", False)),
            "frame_range": frame_range,
            "semantic_combo": dataset.get("dataset", {}).get("semantic_combo"),
            "projection_route": dataset.get("dataset", {}).get("projection_route"),
            "target_count": metrics.get("target_count"),
            "object_count": metrics.get("object_count"),
            "object_ambiguous_ratio": float(metrics.get("object_ambiguous_ratio", 0.0)),
            "surface_first_changed_ratio": float(metrics.get("surface_first_changed_ratio", 0.0)),
            "residual_surface_assigned_ratio": float(metrics.get("residual_surface_assigned_ratio", 0.0)),
            "residual_surface_unassigned_points": metrics.get("residual_surface_unassigned_points"),
            "residual_absorption_sweep_best_ratio": float(metrics.get("residual_absorption_sweep_best_ratio", 0.0)),
            "residual_surface_miss_reason_counts": metrics.get("residual_surface_miss_reason_counts", {}),
            "residual_candidate_coverage_best_ratio": float(metrics.get("residual_candidate_coverage_best_ratio", 0.0)),
            "residual_candidate_coverage_augmented_best_ratio": float(metrics.get("residual_candidate_coverage_augmented_best_ratio", 0.0)),
        },
        "conceptseg_side_track": {
            "decision": "keep_as_conservative_fine_object_refinement_only",
            "candidate_count": concept_align.get("item_count"),
            "aligned_target_count": concept_align.get("target_count"),
            "concept_match_count": concept_align.get("concept_match_count"),
            "semantically_discriminative_target_count": concept_align.get("semantically_discriminative_target_count"),
            "intersection_accepted_candidate_count": concept_inter.get("accepted_candidate_count"),
            "intersection_target_coverage": inter_target_coverage,
            "conclusion": (
                "Useful for a small subset of local fine-object mask refinements after strict instance-mask "
                "intersection; not suitable for dense semantic generation or target-level classification."
            ),
        },
        "old_route_side_track": {
            "decision": "keep_as_fixed_visual_color_reference_only",
            "validation_passed": bool(old_validation.get("passed", False)),
            "colored_ratio": float(old_validation.get("colored_ratio", 0.0)),
            "ply_vertex_count": old_validation.get("ply_vertex_count"),
            "ply_has_rgb": bool(old_validation.get("ply_has_rgb", False)),
            "conclusion": "Validated as an RGB visual sanity reference; no reusable production runner found.",
        },
        "next_steps": [
            "Do not expand ConceptSeg to all frames; first integrate only accepted intersection candidates into fine-object split/refine QA.",
            "Do not revive deprecated transforms.json/project_world_points semantic projection.",
            "For main route, continue from object/residual refinement: stable surface layer first, then fine-object 3D connected components.",
            "Before extending beyond 0-999 frames, validate the current reviewed package visually in the PLY viewer/CloudCompare.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
