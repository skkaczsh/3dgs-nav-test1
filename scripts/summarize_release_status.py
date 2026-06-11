#!/usr/bin/env python3
"""Summarize the current dataset release status across main and side routes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def nested(data: dict[str, Any], *keys: str, default=None):
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.dataset_manifest)
    route = read_json(args.route_decision)
    side = read_json(args.side_track_readiness)
    concept_plan = read_json(args.conceptseg_integration_plan)
    concept_3d = read_json(args.conceptseg_3d_refinement_report)
    old_route = read_json(args.old_route_validation)
    acceptance = read_json(args.delivery_acceptance)
    package = read_json(args.package_manifest)

    metrics = manifest.get("metrics", {})
    main_passed = bool(manifest.get("passed")) and bool(acceptance.get("passed"))
    old_passed = bool(old_route.get("passed"))
    concept_components = int(concept_3d.get("component_count", 0) or 0)
    concept_points = int(concept_3d.get("component_points", 0) or 0)

    release_status = "ready_for_visual_review" if main_passed else "not_ready"
    return {
        "status_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release": {
            "name": "dense_semantic_0000_0999_review_package",
            "status": release_status,
            "package": info(args.package_tgz),
            "package_manifest": info(args.package_manifest),
            "manual_gate": "visual_acceptance_in_ply_viewer_or_cloudcompare",
        },
        "inputs": {
            "dataset_manifest": info(args.dataset_manifest),
            "route_decision": info(args.route_decision),
            "side_track_readiness": info(args.side_track_readiness),
            "conceptseg_integration_plan": info(args.conceptseg_integration_plan),
            "conceptseg_3d_refinement_report": info(args.conceptseg_3d_refinement_report),
            "old_route_validation": info(args.old_route_validation),
            "delivery_acceptance": info(args.delivery_acceptance),
        },
        "main_route": {
            "decision": nested(route, "main_route", "decision", default="continue_as_authoritative_route"),
            "status": "authoritative_review_candidate" if main_passed else "not_ready",
            "projection_route": nested(manifest, "dataset", "projection_route"),
            "semantic_combo": nested(manifest, "dataset", "semantic_combo"),
            "frame_range": nested(manifest, "dataset", "frame_range"),
            "readiness_ratios": metrics.get("readiness_ratios", {}),
            "target_count": metrics.get("target_count"),
            "target_points": metrics.get("target_points"),
            "object_count": metrics.get("object_count"),
            "object_merge_ratio": metrics.get("object_merge_ratio"),
            "object_ambiguous_ratio": metrics.get("object_ambiguous_ratio"),
            "strict_surface_effect": {
                "wall_points_base": metrics.get("surface_fusion_wall_points_base"),
                "wall_points_strict": metrics.get("surface_fusion_wall_points_strict"),
                "ambiguous_points_base": metrics.get("surface_fusion_ambiguous_points_base"),
                "ambiguous_points_strict": metrics.get("surface_fusion_ambiguous_points_strict"),
            },
            "remaining_bottlenecks": {
                "residual_surface_unassigned_points": metrics.get("residual_surface_unassigned_points"),
                "residual_surface_miss_reason_counts": metrics.get("residual_surface_miss_reason_counts", {}),
                "residual_candidate_coverage_augmented_best_ratio": metrics.get("residual_candidate_coverage_augmented_best_ratio"),
            },
            "recommended_viewer_inputs": manifest.get("recommended_viewer_inputs", []),
        },
        "conceptseg_r1": {
            "decision": "review_only_3d_refinement_candidates",
            "side_track_decision": nested(side, "conceptseg_r1", "decision"),
            "integration_decision": concept_plan.get("decision"),
            "accepted_candidates": nested(concept_plan, "summary", "accepted_candidate_count"),
            "accepted_targets": nested(concept_plan, "summary", "accepted_target_count"),
            "accepted_objects": nested(concept_plan, "summary", "accepted_object_count"),
            "accepted_concepts": nested(concept_plan, "summary", "accepted_concept_counts", default={}),
            "three_d_status_counts": concept_3d.get("candidate_status_counts", {}),
            "three_d_component_count": concept_components,
            "three_d_component_points": concept_points,
            "three_d_component_concepts": concept_3d.get("component_concept_counts", {}),
            "promotion_policy": [
                "Do not overwrite dense object labels automatically.",
                "Use only strict instance-intersection candidates.",
                "Require 3D connected-component coherence before manual/VLM split review.",
            ],
        },
        "old_route": {
            "decision": "visual_color_reference_only" if old_passed else "not_ready",
            "side_track_decision": nested(side, "old_route", "decision"),
            "validation_passed": old_passed,
            "colored_ratio": old_route.get("colored_ratio"),
            "ply_vertex_count": old_route.get("ply_vertex_count"),
            "ply_has_rgb": old_route.get("ply_has_rgb"),
            "promotion_policy": [
                "Do not use deprecated transforms.json/project_world_points semantic projection.",
                "Use old-route smoke only as a fixed RGB/geometric sanity reference.",
                "Future reusable old-route runner must use img_pos.txt + cam_in_ex.txt + Tcl + Til.",
            ],
        },
        "acceptance": {
            "delivery_acceptance_passed": acceptance.get("passed"),
            "manifest_required_file_count": nested(read_json(args.manifest_validation), "required_file_count"),
            "package_packaged_file_count": nested(read_json(args.package_validation), "packaged_file_count"),
            "package_large_file_count": nested(read_json(args.package_validation), "large_file_count"),
            "package_passed": package.get("passed"),
        },
        "next_actions": [
            "Visual QA the hybrid strict-surface PLY first; it is the current primary artifact.",
            "Inspect ConceptSeg 3D components only as local fine-object split proposals.",
            "Resolve remaining residual surface bottlenecks before expanding beyond 0-999.",
            "Keep old route as a reference until a scanner-native reproducible runner is implemented.",
        ],
    }


def render_markdown(status: dict[str, Any]) -> str:
    main = status["main_route"]
    concept = status["conceptseg_r1"]
    old = status["old_route"]
    acc = status["acceptance"]
    lines = [
        "# Dense Semantic Release Status",
        "",
        f"- release: `{status['release']['name']}`",
        f"- status: `{status['release']['status']}`",
        f"- manual gate: `{status['release']['manual_gate']}`",
        f"- package: `{status['release']['package']['path']}`",
        "",
        "## Main Route",
        "",
        f"- decision: `{main['decision']}`",
        f"- status: `{main['status']}`",
        f"- projection route: `{main['projection_route']}`",
        f"- semantic combo: `{main['semantic_combo']}`",
        f"- frame range: `{main['frame_range']}`",
        f"- readiness ratios: `{main['readiness_ratios']}`",
        f"- targets/points: `{main['target_count']}` / `{main['target_points']}`",
        f"- objects: `{main['object_count']}`",
        f"- object merge/ambiguous ratio: `{main['object_merge_ratio']}` / `{main['object_ambiguous_ratio']}`",
        f"- strict surface wall points base/strict: `{main['strict_surface_effect']['wall_points_base']}` / `{main['strict_surface_effect']['wall_points_strict']}`",
        f"- strict surface ambiguous points base/strict: `{main['strict_surface_effect']['ambiguous_points_base']}` / `{main['strict_surface_effect']['ambiguous_points_strict']}`",
        f"- residual unassigned points: `{main['remaining_bottlenecks']['residual_surface_unassigned_points']}`",
        f"- residual miss reasons: `{main['remaining_bottlenecks']['residual_surface_miss_reason_counts']}`",
        "",
        "## ConceptSeg-R1",
        "",
        f"- decision: `{concept['decision']}`",
        f"- accepted candidates/targets/objects: `{concept['accepted_candidates']}` / `{concept['accepted_targets']}` / `{concept['accepted_objects']}`",
        f"- accepted concepts: `{concept['accepted_concepts']}`",
        f"- 3D status counts: `{concept['three_d_status_counts']}`",
        f"- 3D components/points: `{concept['three_d_component_count']}` / `{concept['three_d_component_points']}`",
        f"- 3D component concepts: `{concept['three_d_component_concepts']}`",
        "",
        "## Old Route",
        "",
        f"- decision: `{old['decision']}`",
        f"- validation passed: `{old['validation_passed']}`",
        f"- colored ratio: `{old['colored_ratio']}`",
        f"- PLY vertices: `{old['ply_vertex_count']}`",
        f"- RGB present: `{old['ply_has_rgb']}`",
        "",
        "## Acceptance",
        "",
        f"- delivery acceptance passed: `{acc['delivery_acceptance_passed']}`",
        f"- manifest required files: `{acc['manifest_required_file_count']}`",
        f"- package files/large files: `{acc['package_packaged_file_count']}` / `{acc['package_large_file_count']}`",
        "",
        "## Viewer Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in main["recommended_viewer_inputs"])
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in status["next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--dataset-manifest", type=Path, default=root / "route_status_20260610/dataset_delivery_manifest_0000_0999.json")
    parser.add_argument("--manifest-validation", type=Path, default=root / "route_status_20260610/dataset_delivery_manifest_0000_0999_validation.json")
    parser.add_argument("--route-decision", type=Path, default=root / "route_status_20260610/dense_semantic_route_decision_20260611.json")
    parser.add_argument("--side-track-readiness", type=Path, default=root / "route_status_20260610/side_track_readiness_20260611.json")
    parser.add_argument("--conceptseg-integration-plan", type=Path, default=root / "route_status_20260610/conceptseg_integration_plan_20260611.json")
    parser.add_argument("--conceptseg-3d-refinement-report", type=Path, default=root / "server_conceptseg_3d_refinement_v008/conceptseg_3d_refinement_report.json")
    parser.add_argument("--old-route-validation", type=Path, default=root / "server_old_route_smoke/old_route_reference_validation.json")
    parser.add_argument("--delivery-acceptance", type=Path, default=root / "route_status_20260610/delivery_acceptance_20260611.json")
    parser.add_argument("--package-manifest", type=Path, default=root / "dataset_delivery_0000_0999/package_manifest.json")
    parser.add_argument("--package-validation", type=Path, default=root / "dataset_delivery_0000_0999_validation.json")
    parser.add_argument("--package-tgz", type=Path, default=root / "dataset_delivery_0000_0999.tgz")
    parser.add_argument("--output-json", type=Path, default=root / "route_status_20260610/dense_semantic_release_status_20260611.json")
    parser.add_argument("--output-md", type=Path, default=root / "route_status_20260610/dense_semantic_release_status_20260611.md")
    args = parser.parse_args()

    status = build_status(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(status), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md), "status": status["release"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
