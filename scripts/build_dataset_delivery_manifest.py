#!/usr/bin/env python3
"""Build a delivery manifest for the 0-999 dense semantic dataset run."""

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


def file_entry(path: Path, role: str, required: bool = True, remote_path: str = "") -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "remote_path": remote_path,
        "required": required,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def check_threshold(name: str, value: float | None, threshold: float, op: str) -> dict[str, Any]:
    if value is None:
        passed = False
    elif op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    else:
        raise ValueError(op)
    return {"name": name, "value": value, "op": op, "threshold": threshold, "passed": passed}


def nested(data: dict, *keys: str, default=None):
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    readiness = read_json(args.dataset_readiness)
    validation = read_json(args.output_validation)
    target_qa = read_json(args.target_object_qa)
    object_pipeline_qa = read_json(args.object_pipeline_qa)
    surface = read_json(args.surface_first_report)
    residual_assignment = read_json(args.residual_assignment_report)
    residual_sweep = read_json(args.residual_absorption_sweep)
    residual_miss_reasons = read_json(args.residual_miss_reasons)
    residual_candidate_coverage = read_json(args.residual_candidate_coverage)
    surface_seed_candidates = read_json(args.surface_seed_candidates)
    surface_seed_promotion = read_json(args.surface_seed_promotion)
    residual_candidate_coverage_augmented = read_json(args.residual_candidate_coverage_augmented)
    surface_fusion_bottleneck = read_json(args.surface_fusion_bottleneck)
    surface_fusion_bottleneck_strict = read_json(args.surface_fusion_bottleneck_strict)
    surface_consolidation = read_json(args.surface_consolidation_report)
    surface_hybrid_consolidation = read_json(args.surface_hybrid_report)
    concept = read_json(args.conceptseg_qa)
    route_decision = read_json(args.route_decision)
    release_status = read_json(args.release_status)
    infra_readiness = read_json(args.infra_readiness)
    parallel_execution_queue = read_json(args.parallel_execution_queue)
    visual_acceptance = read_json(args.visual_acceptance)
    visual_acceptance_validation = read_json(args.visual_acceptance_validation)
    next_increment_readiness = read_json(args.next_increment_readiness)
    concept_align = read_json(args.conceptseg_alignment)
    concept_intersection = read_json(args.conceptseg_intersection)
    concept_integration = read_json(args.conceptseg_integration_plan)
    concept_3d_refinement = read_json(args.conceptseg_3d_refinement_report)
    side_track_readiness = read_json(args.side_track_readiness)
    old_route = read_json(args.old_route_summary)
    old_route_validation = read_json(args.old_route_validation)
    delivery_acceptance = read_json(args.delivery_acceptance)
    fine_targets = read_json(args.fine_targets_report)
    fine_tracklets = read_json(args.fine_tracklet_report)
    long_assoc = read_json(args.long_assoc_report)
    reviewed_merge = read_json(args.reviewed_merge_qa)

    ratios = readiness.get("ratios", {})
    target_objects = target_qa.get("objects", {})
    checks = [
        check_threshold("complete_camera_frames", ratios.get("complete_camera_frames"), 1.0, ">="),
        check_threshold("color_ply", ratios.get("color_ply"), 0.95, ">="),
        check_threshold("sky_masks", ratios.get("sky_masks"), 0.95, ">="),
        check_threshold("sam2_masks", ratios.get("sam2_masks"), 0.95, ">="),
        check_threshold("completion_semantic_images", ratios.get("completion_semantic_images"), 0.90, ">="),
        check_threshold("target_frame_ok_ratio", nested(target_qa, "frames", "ok_count", default=0) / max(nested(target_qa, "frames", "count", default=1), 1), 0.95, ">="),
        check_threshold("ambiguous_ratio", target_objects.get("ambiguous_ratio"), 0.35, "<="),
        check_threshold("surface_first_changed_ratio", surface.get("changed_ratio"), 0.0, ">="),
        check_threshold("residual_surface_assigned_ratio", residual_assignment.get("assigned_ratio"), 0.0, ">="),
        check_threshold("conceptseg_items", concept.get("items"), 40, ">="),
        check_threshold("conceptseg_instance_targets", nested(concept_intersection, "target_status_counts", "has_intersection_candidate", default=0), 1, ">="),
        check_threshold("old_route_colored_ratio", old_route.get("colored_ratio"), 0.80, ">="),
        check_threshold("old_route_reference_validation", 1.0 if old_route_validation.get("passed") else 0.0, 1.0, ">="),
        check_threshold("delivery_acceptance", 1.0 if delivery_acceptance.get("passed") else 0.0, 1.0, ">="),
        check_threshold("infra_readiness", 1.0 if infra_readiness.get("passed") else 0.0, 1.0, ">="),
        check_threshold("fine_targets", fine_targets.get("targets"), 3000, ">="),
        check_threshold("fine_tracklet_merge_ratio", fine_tracklets.get("merge_ratio"), 0.85, ">="),
        check_threshold("long_assoc_objects", long_assoc.get("objects"), 100, "<="),
    ]

    files = [
        file_entry(args.dataset_readiness, "dataset_readiness_report"),
        file_entry(args.output_validation, "strict_output_validation"),
        file_entry(args.target_object_qa, "target_object_qa"),
        file_entry(args.object_pipeline_qa, "object_pipeline_qa_summary"),
        file_entry(args.objects_jsonl, "target_object_objects_jsonl"),
        file_entry(args.object_points_ply, "target_object_full_ply"),
        file_entry(args.object_points_stride_ply, "target_object_preview_ply"),
        file_entry(
            args.surface_first_voxel_ply,
            "surface_first_subcluster_preview_ply",
            remote_path="/root/epfs/new_route_stage1_skymask/surface_first_subcluster_qa_0000_0999/object_points_surface_first_subcluster_voxel004.ply",
        ),
        file_entry(args.surface_first_report, "surface_first_subcluster_report"),
        file_entry(args.surface_first_preview, "surface_first_subcluster_xy_preview"),
        file_entry(args.residual_assignment_report, "residual_surface_assignment_report", required=True),
        file_entry(args.residual_assignment_preview, "residual_surface_assignment_xy_preview", required=True),
        file_entry(args.residual_absorption_sweep, "residual_absorption_sweep_report", required=True),
        file_entry(args.residual_miss_reasons, "residual_surface_miss_reasons_report", required=True),
        file_entry(args.residual_candidate_coverage, "residual_candidate_surface_coverage_report", required=True),
        file_entry(args.surface_seed_candidates, "surface_seed_candidates_report", required=True),
        file_entry(args.surface_seed_promotion, "surface_seed_promotion_report", required=True),
        file_entry(args.residual_candidate_coverage_augmented, "residual_candidate_surface_coverage_augmented_report", required=True),
        file_entry(args.surface_fusion_bottleneck, "surface_target_fusion_bottleneck_report", required=True),
        file_entry(args.surface_fusion_bottleneck_strict, "surface_target_fusion_bottleneck_strict_report", required=True),
        file_entry(args.strict_surface_stride_ply, "strict_surface_fusion_preview_ply", required=True),
        file_entry(args.strict_surface_preview, "strict_surface_fusion_xy_preview", required=True),
        file_entry(args.strict_surface_fusion_report, "strict_surface_fusion_report", required=True),
        file_entry(args.surface_consolidated_stride_ply, "surface_consolidated_preview_ply", required=True),
        file_entry(args.surface_consolidated_preview, "surface_consolidated_xy_preview", required=True),
        file_entry(args.surface_consolidation_report, "surface_consolidation_report", required=True),
        file_entry(args.surface_consolidation_mapping, "surface_consolidation_mapping", required=True),
        file_entry(args.surface_consolidation_sweep, "surface_consolidation_sweep_report", required=True),
        file_entry(args.surface_hybrid_stride_ply, "surface_hybrid_consolidated_preview_ply", required=True),
        file_entry(args.surface_hybrid_preview, "surface_hybrid_consolidated_xy_preview", required=True),
        file_entry(args.surface_hybrid_report, "surface_hybrid_consolidation_report", required=True),
        file_entry(args.surface_hybrid_mapping, "surface_hybrid_consolidation_mapping", required=True),
        file_entry(args.conceptseg_qa, "conceptseg_problem40_structured_qa", required=False),
        file_entry(args.conceptseg_contact_sheet, "conceptseg_problem40_contact_sheet", required=False),
        file_entry(args.route_decision, "dense_semantic_route_decision", required=True),
        file_entry(args.route_decision_md, "dense_semantic_route_decision_markdown", required=True),
        file_entry(args.release_status, "dense_semantic_release_status", required=True),
        file_entry(args.release_status_md, "dense_semantic_release_status_markdown", required=True),
        file_entry(args.infra_readiness, "infra_readiness", required=True),
        file_entry(args.infra_readiness_md, "infra_readiness_markdown", required=True),
        file_entry(args.parallel_execution_queue, "parallel_execution_queue", required=True),
        file_entry(args.parallel_execution_queue_md, "parallel_execution_queue_markdown", required=True),
        file_entry(args.visual_acceptance, "visual_acceptance_review", required=True),
        file_entry(args.visual_acceptance_md, "visual_acceptance_review_markdown", required=True),
        file_entry(args.visual_acceptance_html, "visual_acceptance_review_html", required=True),
        file_entry(args.visual_acceptance_validation, "visual_acceptance_review_validation", required=True),
        file_entry(args.next_increment_readiness, "next_increment_readiness", required=True),
        file_entry(args.next_increment_readiness_md, "next_increment_readiness_markdown", required=True),
        file_entry(args.conceptseg_alignment, "conceptseg_fine_object_alignment", required=False),
        file_entry(args.conceptseg_intersection, "conceptseg_instance_intersection", required=False),
        file_entry(args.conceptseg_instance_accepted_sheet, "conceptseg_instance_accepted_sheet", required=False),
        file_entry(args.conceptseg_integration_plan, "conceptseg_integration_plan", required=True),
        file_entry(args.conceptseg_integration_plan_md, "conceptseg_integration_plan_markdown", required=True),
        file_entry(args.conceptseg_accepted_integration_candidates, "conceptseg_accepted_integration_candidates", required=True),
        file_entry(args.conceptseg_3d_refinement_report, "conceptseg_3d_refinement_report", required=True),
        file_entry(args.conceptseg_3d_components_jsonl, "conceptseg_3d_components_jsonl", required=True),
        file_entry(args.conceptseg_3d_components_ply, "conceptseg_3d_components_ply", required=True),
        file_entry(args.conceptseg_3d_components_preview, "conceptseg_3d_components_xy_preview", required=True),
        file_entry(args.side_track_readiness, "side_track_readiness", required=True),
        file_entry(args.side_track_readiness_md, "side_track_readiness_markdown", required=True),
        file_entry(args.old_route_summary, "old_route_color_smoke_summary", required=False),
        file_entry(args.old_route_preview, "old_route_color_smoke_preview", required=False),
        file_entry(args.old_route_validation, "old_route_reference_validation", required=False),
        file_entry(args.delivery_acceptance, "delivery_acceptance_report", required=True),
        file_entry(args.fine_targets_report, "v008_fine_targets_report"),
        file_entry(args.fine_targets_jsonl, "v008_fine_targets_jsonl"),
        file_entry(args.fine_tracklet_report, "v008_fine_tracklet_report"),
        file_entry(args.fine_tracklets_jsonl, "v008_fine_tracklets_jsonl"),
        file_entry(args.long_assoc_report, "v008_long_association_report"),
        file_entry(args.long_objects_jsonl, "v008_long_objects_jsonl"),
        file_entry(args.reviewed_merge_qa, "v008_reviewed_merge_qa"),
        file_entry(args.reviewed_long_objects_jsonl, "v008_reviewed_long_objects_jsonl"),
    ]
    file_failures = [f for f in files if f["required"] and not f["exists"]]

    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "new_route_dense_semantic_0000_0999",
            "frame_range": [0, 999],
            "camera_count": 3,
            "semantic_combo": "sam2_prompt_v3_sky_label_merge_completion",
            "projection_route": "img_pos.txt + cam_in_ex.txt + Tcl + Til",
            "status": "ready_for_reviewed_side_tracks" if validation.get("passed") else "not_ready",
        },
        "metrics": {
            "readiness_counts": readiness.get("counts", {}),
            "readiness_ratios": ratios,
            "target_count": nested(target_qa, "targets", "count"),
            "target_points": nested(target_qa, "targets", "points"),
            "target_small_residual_ratio": nested(target_qa, "targets", "small_residual_ratio"),
            "object_count": target_objects.get("count"),
            "object_merge_ratio": target_objects.get("merge_ratio"),
            "object_ambiguous_ratio": target_objects.get("ambiguous_ratio"),
            "object_pipeline_recommendations": object_pipeline_qa.get("recommendations", []),
            "surface_first_changed_ratio": surface.get("changed_ratio"),
            "surface_first_after_counts": surface.get("after_counts", {}),
            "residual_surface_assigned_ratio": residual_assignment.get("assigned_ratio"),
            "residual_surface_assigned_points": residual_assignment.get("assigned_points"),
            "residual_surface_unassigned_points": (
                residual_assignment.get("residual_points", 0) - residual_assignment.get("assigned_points", 0)
                if isinstance(residual_assignment.get("residual_points"), int)
                and isinstance(residual_assignment.get("assigned_points"), int)
                else None
            ),
            "residual_surface_by_label": residual_assignment.get("by_label", {}),
            "residual_surface_assigned_by_label": residual_assignment.get("assigned_by_label", {}),
            "residual_absorption_sweep_best_ratio": max(
                (row.get("assigned_ratio", 0.0) for row in residual_sweep.get("configs", [])),
                default=0.0,
            ),
            "residual_absorption_sweep_best_unassigned": (
                sorted(
                    max(
                        residual_sweep.get("configs", []),
                        key=lambda row: row.get("assigned_ratio", 0.0),
                        default={},
                    ).get("unassigned_by_label", {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
            ),
            "residual_surface_miss_reason_counts": residual_miss_reasons.get("reason_counts", {}),
            "residual_surface_miss_reason_by_label": residual_miss_reasons.get("reason_by_label", {}),
            "residual_candidate_coverage_best_ratio": max(
                (row.get("matched_surface_ratio", 0.0) for row in residual_candidate_coverage.get("configs", [])),
                default=0.0,
            ),
            "residual_candidate_coverage_best_reasons": max(
                residual_candidate_coverage.get("configs", []),
                key=lambda row: row.get("matched_surface_ratio", 0.0),
                default={},
            ).get("reason_counts", {}),
            "surface_seed_candidate_count": surface_seed_candidates.get("candidate_count"),
            "surface_seed_candidate_points": surface_seed_candidates.get("candidate_points"),
            "surface_seed_promoted_count": surface_seed_promotion.get("promoted_count"),
            "surface_seed_promoted_points": surface_seed_promotion.get("promoted_points"),
            "residual_candidate_coverage_augmented_best_ratio": max(
                (row.get("matched_surface_ratio", 0.0) for row in residual_candidate_coverage_augmented.get("configs", [])),
                default=0.0,
            ),
            "surface_fusion_wall_points_base": nested(surface_fusion_bottleneck, "objects", "by_label", "wall", "point_count"),
            "surface_fusion_wall_points_strict": nested(surface_fusion_bottleneck_strict, "objects", "by_label", "wall", "point_count"),
            "surface_fusion_ambiguous_points_base": nested(surface_fusion_bottleneck, "objects", "by_label", "ambiguous", "point_count"),
            "surface_fusion_ambiguous_points_strict": nested(surface_fusion_bottleneck_strict, "objects", "by_label", "ambiguous", "point_count"),
            "surface_consolidation_input_objects": surface_consolidation.get("input_objects"),
            "surface_consolidation_output_objects": surface_consolidation.get("output_objects"),
            "surface_consolidation_merged_reduction": surface_consolidation.get("merged_object_reduction"),
            "surface_hybrid_consolidation_input_objects": surface_hybrid_consolidation.get("input_objects"),
            "surface_hybrid_consolidation_output_objects": surface_hybrid_consolidation.get("output_objects"),
            "surface_hybrid_consolidation_merged_reduction": surface_hybrid_consolidation.get("merged_object_reduction"),
            "conceptseg_items": concept.get("items"),
            "conceptseg_mode_counts": concept.get("mode_counts", {}),
            "route_decision": nested(route_decision, "main_route", "decision"),
            "release_status": nested(release_status, "release", "status"),
            "release_manual_gate": nested(release_status, "release", "manual_gate"),
            "infra_readiness_passed": infra_readiness.get("passed"),
            "infra_all_reachable": infra_readiness.get("all_reachable"),
            "infra_all_required_paths_ok": infra_readiness.get("all_required_paths_ok"),
            "infra_local_delivery_artifacts_ok": infra_readiness.get("local_delivery_artifacts_ok"),
            "infra_servers": [
                {
                    "name": server.get("name"),
                    "reachable": server.get("reachable"),
                    "role": server.get("role"),
                    "gpus": server.get("gpus", []),
                    "disk": server.get("disk", []),
                }
                for server in infra_readiness.get("servers", [])
            ],
            "parallel_queue_task_count": len(parallel_execution_queue.get("queue", [])),
            "parallel_queue_gates": parallel_execution_queue.get("gates", {}),
            "visual_acceptance_status": visual_acceptance.get("status"),
            "visual_acceptance_allow_next_increment": visual_acceptance.get("allow_next_increment"),
            "visual_acceptance_validation_passed": visual_acceptance_validation.get("passed"),
            "visual_acceptance_pending_required": visual_acceptance_validation.get("pending_required", []),
            "next_increment_status": next_increment_readiness.get("status"),
            "next_increment_ratios": next_increment_readiness.get("ratios", {}),
            "next_increment_next_steps": next_increment_readiness.get("next_steps", []),
            "conceptseg_decision": nested(route_decision, "conceptseg_side_track", "decision"),
            "conceptseg_fine_candidates": concept_align.get("item_count"),
            "conceptseg_semantically_discriminative_targets": concept_align.get("semantically_discriminative_target_count"),
            "conceptseg_instance_accepted_candidates": concept_intersection.get("accepted_candidate_count"),
            "conceptseg_instance_target_status_counts": concept_intersection.get("target_status_counts", {}),
            "conceptseg_integration_decision": concept_integration.get("decision"),
            "conceptseg_integration_role": concept_integration.get("role"),
            "conceptseg_integration_accepted_targets": nested(concept_integration, "summary", "accepted_target_count"),
            "conceptseg_integration_accepted_objects": nested(concept_integration, "summary", "accepted_object_count"),
            "conceptseg_integration_accepted_source_labels": nested(concept_integration, "summary", "accepted_source_label_counts", default={}),
            "conceptseg_3d_refinement_status_counts": concept_3d_refinement.get("candidate_status_counts", {}),
            "conceptseg_3d_refinement_components": concept_3d_refinement.get("component_count"),
            "conceptseg_3d_refinement_component_points": concept_3d_refinement.get("component_points"),
            "conceptseg_3d_refinement_concepts": concept_3d_refinement.get("component_concept_counts", {}),
            "side_track_conceptseg_decision": nested(side_track_readiness, "conceptseg_r1", "decision"),
            "side_track_conceptseg_accepted_target_ratio": nested(side_track_readiness, "conceptseg_r1", "accepted_target_ratio"),
            "side_track_old_route_decision": nested(side_track_readiness, "old_route", "decision"),
            "old_route_decision": nested(route_decision, "old_route_side_track", "decision"),
            "old_route_colored_ratio": old_route.get("colored_ratio"),
            "old_route_reference_passed": old_route_validation.get("passed"),
            "delivery_acceptance_passed": delivery_acceptance.get("passed"),
            "fine_targets": fine_targets.get("targets"),
            "fine_target_points": fine_targets.get("target_points"),
            "fine_target_small_residual_points": fine_targets.get("small_residual_points"),
            "fine_tracklets": fine_tracklets.get("tracklets"),
            "fine_tracklet_merge_ratio": fine_tracklets.get("merge_ratio"),
            "long_assoc_objects": long_assoc.get("objects"),
            "long_assoc_merge_ratio": long_assoc.get("merge_ratio"),
            "reviewed_output_objects": reviewed_merge.get("output_object_count"),
            "reviewed_accepted_merge_count": reviewed_merge.get("accepted_merge_count"),
            "reviewed_merge_passed": reviewed_merge.get("passed"),
        },
        "checks": checks,
        "passed": bool(validation.get("passed")) and not file_failures and all(c["passed"] for c in checks),
        "file_failures": file_failures,
        "files": files,
        "recommended_viewer_inputs": [
            str(args.surface_hybrid_stride_ply),
            str(args.surface_consolidated_stride_ply),
            str(args.strict_surface_stride_ply),
            str(args.surface_first_voxel_ply),
            str(args.object_points_stride_ply),
            str(args.conceptseg_3d_components_ply),
        ],
        "next_actions": [
            "Use surface-first subcluster PLY for visual QA of surface contamination.",
            "Use residual surface assignment report to separate absorbable surface noise from unresolved fine residuals.",
            "Use strict surface-label fusion to prevent wall targets from being absorbed into floor objects.",
            "Use v008 frame-target/tracklet/long-association artifacts for fine-object dataset evolution.",
            "Use reviewed long objects as the current fine-object object baseline.",
            "Keep ConceptSeg-R1 as constrained second-stage candidate generator.",
            "Use ConceptSeg-R1 only after strict instance-mask intersection; current accepted target coverage is low.",
            "Use ConceptSeg integration plan as review-only split/refine candidates; do not overwrite dense labels automatically.",
            "Use ConceptSeg 3D refinement components as visual QA proposals only; current accepted component coverage is small.",
            "Keep old route as visual color reference only.",
            "Use the infra readiness report before launching more 1000+ frame or model side-track jobs.",
        ],
    }
    return manifest


def render_markdown(manifest: dict[str, Any]) -> str:
    dataset = manifest["dataset"]
    metrics = manifest["metrics"]
    lines = [
        "# Dataset Delivery Manifest",
        "",
        f"- dataset: `{dataset['name']}`",
        f"- status: `{dataset['status']}`",
        f"- frame range: `{dataset['frame_range'][0]}-{dataset['frame_range'][1]}`",
        f"- semantic combo: `{dataset['semantic_combo']}`",
        f"- passed: `{manifest['passed']}`",
        "",
        "## Metrics",
        "",
        f"- readiness ratios: `{metrics.get('readiness_ratios')}`",
        f"- targets: `{metrics.get('target_count')}`",
        f"- target points: `{metrics.get('target_points')}`",
        f"- small residual ratio: `{metrics.get('target_small_residual_ratio')}`",
        f"- objects: `{metrics.get('object_count')}`",
        f"- object merge ratio: `{metrics.get('object_merge_ratio')}`",
        f"- object ambiguous ratio: `{metrics.get('object_ambiguous_ratio')}`",
        f"- surface-first changed ratio: `{metrics.get('surface_first_changed_ratio')}`",
        f"- residual surface assigned ratio: `{metrics.get('residual_surface_assigned_ratio')}`",
        f"- residual surface unassigned points: `{metrics.get('residual_surface_unassigned_points')}`",
        f"- surface fusion wall points base/strict: `{metrics.get('surface_fusion_wall_points_base')}` / `{metrics.get('surface_fusion_wall_points_strict')}`",
        f"- surface fusion ambiguous points base/strict: `{metrics.get('surface_fusion_ambiguous_points_base')}` / `{metrics.get('surface_fusion_ambiguous_points_strict')}`",
        f"- surface consolidation objects input/output/reduced: `{metrics.get('surface_consolidation_input_objects')}` / `{metrics.get('surface_consolidation_output_objects')}` / `{metrics.get('surface_consolidation_merged_reduction')}`",
        f"- hybrid surface consolidation objects input/output/reduced: `{metrics.get('surface_hybrid_consolidation_input_objects')}` / `{metrics.get('surface_hybrid_consolidation_output_objects')}` / `{metrics.get('surface_hybrid_consolidation_merged_reduction')}`",
        f"- ConceptSeg modes: `{metrics.get('conceptseg_mode_counts')}`",
        f"- route decision: `{metrics.get('route_decision')}`",
        f"- release status: `{metrics.get('release_status')}`",
        f"- release manual gate: `{metrics.get('release_manual_gate')}`",
        f"- infra readiness passed/all reachable/paths ok: `{metrics.get('infra_readiness_passed')}` / `{metrics.get('infra_all_reachable')}` / `{metrics.get('infra_all_required_paths_ok')}`",
        f"- parallel queue tasks/gates: `{metrics.get('parallel_queue_task_count')}` / `{metrics.get('parallel_queue_gates')}`",
        f"- visual acceptance status/allow next increment: `{metrics.get('visual_acceptance_status')}` / `{metrics.get('visual_acceptance_allow_next_increment')}`",
        f"- next increment status: `{metrics.get('next_increment_status')}`",
        f"- ConceptSeg decision: `{metrics.get('conceptseg_decision')}`",
        f"- side-track ConceptSeg decision: `{metrics.get('side_track_conceptseg_decision')}`",
        f"- side-track ConceptSeg accepted target ratio: `{metrics.get('side_track_conceptseg_accepted_target_ratio')}`",
        f"- ConceptSeg instance accepted candidates: `{metrics.get('conceptseg_instance_accepted_candidates')}`",
        f"- ConceptSeg instance target status: `{metrics.get('conceptseg_instance_target_status_counts')}`",
        f"- ConceptSeg integration decision: `{metrics.get('conceptseg_integration_decision')}`",
        f"- ConceptSeg integration accepted targets/objects: `{metrics.get('conceptseg_integration_accepted_targets')}` / `{metrics.get('conceptseg_integration_accepted_objects')}`",
        f"- ConceptSeg integration accepted source labels: `{metrics.get('conceptseg_integration_accepted_source_labels')}`",
        f"- ConceptSeg 3D refinement status: `{metrics.get('conceptseg_3d_refinement_status_counts')}`",
        f"- ConceptSeg 3D refinement components/points: `{metrics.get('conceptseg_3d_refinement_components')}` / `{metrics.get('conceptseg_3d_refinement_component_points')}`",
        f"- ConceptSeg 3D refinement concepts: `{metrics.get('conceptseg_3d_refinement_concepts')}`",
        f"- side-track old route decision: `{metrics.get('side_track_old_route_decision')}`",
        f"- old route decision: `{metrics.get('old_route_decision')}`",
        f"- old route colored ratio: `{metrics.get('old_route_colored_ratio')}`",
        f"- old route reference passed: `{metrics.get('old_route_reference_passed')}`",
        f"- delivery acceptance passed: `{metrics.get('delivery_acceptance_passed')}`",
        f"- fine targets: `{metrics.get('fine_targets')}`",
        f"- fine tracklets: `{metrics.get('fine_tracklets')}`",
        f"- long-association objects: `{metrics.get('long_assoc_objects')}`",
        f"- reviewed fine objects: `{metrics.get('reviewed_output_objects')}`",
        "",
        "## Recommended Viewer Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in manifest["recommended_viewer_inputs"])
    lines.extend(["", "## Required Files", ""])
    for row in manifest["files"]:
        if row["required"]:
            lines.append(f"- `{row['role']}` exists=`{row['exists']}` bytes=`{row['bytes']}`: `{row['path']}`")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in manifest["next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--output-dir", type=Path, default=root / "route_status_20260610")
    parser.add_argument("--dataset-readiness", type=Path, default=root / "route_status_20260610/server_dataset_readiness_0000_0999.json")
    parser.add_argument("--output-validation", type=Path, default=root / "route_status_20260610/server_resume_output_validation.json")
    parser.add_argument("--target-object-qa", type=Path, default=root / "server_resume_target_object_fusion_0000_0999/reports/target_object_qa.json")
    parser.add_argument("--object-pipeline-qa", type=Path, default=root / "route_status_20260610/object_pipeline_qa_summary_latest.json")
    parser.add_argument("--objects-jsonl", type=Path, default=root / "server_resume_target_object_fusion_0000_0999/objects/objects.jsonl")
    parser.add_argument("--object-points-ply", type=Path, default=root / "server_resume_target_object_fusion_0000_0999/objects/object_points_latest.ply")
    parser.add_argument("--object-points-stride-ply", type=Path, default=root / "server_resume_target_object_fusion_0000_0999/objects/object_points_latest_stride10.ply")
    parser.add_argument("--surface-first-report", type=Path, default=root / "server_surface_first_subcluster_qa_0000_0999/surface_first_subcluster_report.json")
    parser.add_argument("--surface-first-voxel-ply", type=Path, default=root / "server_surface_first_subcluster_qa_0000_0999/object_points_surface_first_subcluster_voxel004.ply")
    parser.add_argument("--surface-first-preview", type=Path, default=root / "server_surface_first_subcluster_qa_0000_0999/object_points_surface_first_subcluster_xy.png")
    parser.add_argument("--residual-assignment-report", type=Path, default=root / "server_residual_surface_assignment_0000_0999/assignment_report.json")
    parser.add_argument("--residual-assignment-preview", type=Path, default=root / "server_residual_surface_assignment_0000_0999/residual_surface_assigned_xy.png")
    parser.add_argument("--residual-absorption-sweep", type=Path, default=root / "server_residual_surface_assignment_0000_0999/residual_absorption_sweep_20260611.json")
    parser.add_argument("--residual-miss-reasons", type=Path, default=root / "server_residual_surface_assignment_0000_0999/residual_surface_miss_reasons_20260611.json")
    parser.add_argument("--residual-candidate-coverage", type=Path, default=root / "server_residual_surface_assignment_0000_0999/residual_candidate_surface_coverage_20260611.json")
    parser.add_argument("--surface-seed-candidates", type=Path, default=root / "route_status_20260610/surface_seed_candidates_20260611.json")
    parser.add_argument("--surface-seed-promotion", type=Path, default=root / "route_status_20260610/surface_seed_promotion_20260611.json")
    parser.add_argument("--residual-candidate-coverage-augmented", type=Path, default=root / "route_status_20260610/residual_candidate_surface_coverage_augmented_20260611.json")
    parser.add_argument("--surface-fusion-bottleneck", type=Path, default=root / "route_status_20260610/surface_target_fusion_bottleneck_20260611.json")
    parser.add_argument("--surface-fusion-bottleneck-strict", type=Path, default=root / "route_status_20260610/surface_target_fusion_bottleneck_strict_20260611.json")
    parser.add_argument("--strict-surface-stride-ply", type=Path, default=root / "server_strict_surface_fusion_0000_0999/objects/object_points_strict_surface_stride10.ply")
    parser.add_argument("--strict-surface-preview", type=Path, default=root / "server_strict_surface_fusion_0000_0999/objects/object_points_strict_surface_stride10_xy.png")
    parser.add_argument("--strict-surface-fusion-report", type=Path, default=root / "server_strict_surface_fusion_0000_0999/objects/fusion_report.json")
    parser.add_argument("--surface-consolidated-stride-ply", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated/object_points_strict_surface_consolidated_stride10.ply")
    parser.add_argument("--surface-consolidated-preview", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated/object_points_strict_surface_consolidated_stride10_xy.png")
    parser.add_argument("--surface-consolidation-report", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated/surface_consolidation_report.json")
    parser.add_argument("--surface-consolidation-mapping", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated/object_mapping.jsonl")
    parser.add_argument("--surface-consolidation-sweep", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated/surface_consolidation_sweep_20260611.json")
    parser.add_argument("--surface-hybrid-stride-ply", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated_hybrid/object_points_strict_surface_hybrid_stride10.ply")
    parser.add_argument("--surface-hybrid-preview", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated_hybrid/object_points_strict_surface_hybrid_stride10_xy.png")
    parser.add_argument("--surface-hybrid-report", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated_hybrid/surface_consolidation_report.json")
    parser.add_argument("--surface-hybrid-mapping", type=Path, default=root / "server_strict_surface_fusion_0000_0999/surface_consolidated_hybrid/object_mapping.jsonl")
    parser.add_argument("--conceptseg-qa", type=Path, default=root / "server_conceptseg_problem40/conceptseg_problem40_structured_qa.json")
    parser.add_argument("--conceptseg-contact-sheet", type=Path, default=root / "server_conceptseg_problem40/conceptseg_problem40_contact_sheet.jpg")
    parser.add_argument("--route-decision", type=Path, default=root / "route_status_20260610/dense_semantic_route_decision_20260611.json")
    parser.add_argument("--route-decision-md", type=Path, default=root / "route_status_20260610/dense_semantic_route_decision_20260611.md")
    parser.add_argument("--release-status", type=Path, default=root / "route_status_20260610/dense_semantic_release_status_20260611.json")
    parser.add_argument("--release-status-md", type=Path, default=root / "route_status_20260610/dense_semantic_release_status_20260611.md")
    parser.add_argument("--infra-readiness", type=Path, default=root / "route_status_20260610/infra_readiness_20260611.json")
    parser.add_argument("--infra-readiness-md", type=Path, default=root / "route_status_20260610/infra_readiness_20260611.md")
    parser.add_argument("--parallel-execution-queue", type=Path, default=root / "route_status_20260610/parallel_execution_queue_20260611.json")
    parser.add_argument("--parallel-execution-queue-md", type=Path, default=root / "route_status_20260610/parallel_execution_queue_20260611.md")
    parser.add_argument("--visual-acceptance", type=Path, default=root / "route_status_20260610/visual_acceptance_review_20260611.json")
    parser.add_argument("--visual-acceptance-md", type=Path, default=root / "route_status_20260610/visual_acceptance_review_20260611.md")
    parser.add_argument("--visual-acceptance-html", type=Path, default=root / "route_status_20260610/visual_acceptance_review_20260611.html")
    parser.add_argument("--visual-acceptance-validation", type=Path, default=root / "route_status_20260610/visual_acceptance_review_20260611_validation.json")
    parser.add_argument("--next-increment-readiness", type=Path, default=root / "route_status_20260610/next_increment_readiness_1000_1999.json")
    parser.add_argument("--next-increment-readiness-md", type=Path, default=root / "route_status_20260610/next_increment_readiness_1000_1999.md")
    parser.add_argument("--conceptseg-alignment", type=Path, default=root / "server_conceptseg_fine_object_alignment_v008/conceptseg_target_object_alignment_report.json")
    parser.add_argument("--conceptseg-intersection", type=Path, default=root / "server_conceptseg_instance_intersection_v008/conceptseg_instance_intersection_report.json")
    parser.add_argument("--conceptseg-instance-accepted-sheet", type=Path, default=root / "server_conceptseg_instance_intersection_v008/conceptseg_instance_accepted_sheet.jpg")
    parser.add_argument("--conceptseg-integration-plan", type=Path, default=root / "route_status_20260610/conceptseg_integration_plan_20260611.json")
    parser.add_argument("--conceptseg-integration-plan-md", type=Path, default=root / "route_status_20260610/conceptseg_integration_plan_20260611.md")
    parser.add_argument("--conceptseg-accepted-integration-candidates", type=Path, default=root / "route_status_20260610/conceptseg_accepted_integration_candidates_20260611.jsonl")
    parser.add_argument("--conceptseg-3d-refinement-report", type=Path, default=root / "server_conceptseg_3d_refinement_v008/conceptseg_3d_refinement_report.json")
    parser.add_argument("--conceptseg-3d-components-jsonl", type=Path, default=root / "server_conceptseg_3d_refinement_v008/conceptseg_3d_components.jsonl")
    parser.add_argument("--conceptseg-3d-components-ply", type=Path, default=root / "server_conceptseg_3d_refinement_v008/conceptseg_3d_components.ply")
    parser.add_argument("--conceptseg-3d-components-preview", type=Path, default=root / "server_conceptseg_3d_refinement_v008/conceptseg_3d_components_xy.png")
    parser.add_argument("--side-track-readiness", type=Path, default=root / "route_status_20260610/side_track_readiness_20260611.json")
    parser.add_argument("--side-track-readiness-md", type=Path, default=root / "route_status_20260610/side_track_readiness_20260611.md")
    parser.add_argument("--old-route-summary", type=Path, default=root / "server_old_route_smoke/world_colorize_summary.json")
    parser.add_argument("--old-route-preview", type=Path, default=root / "server_old_route_smoke/old_route_world_color_smoke_s8_v010_best_chroma_xy.png")
    parser.add_argument("--old-route-validation", type=Path, default=root / "server_old_route_smoke/old_route_reference_validation.json")
    parser.add_argument("--delivery-acceptance", type=Path, default=root / "route_status_20260610/delivery_acceptance_20260611.json")
    parser.add_argument("--fine-targets-report", type=Path, default=root / "server_frame_fine_target_object_v008/frame_fine_targets_0000_0999_v008_sweep/v0.16_m3/frame_fine_targets_report.json")
    parser.add_argument("--fine-targets-jsonl", type=Path, default=root / "server_frame_fine_target_object_v008/frame_fine_targets_0000_0999_v008_sweep/v0.16_m3/targets_all.jsonl")
    parser.add_argument("--fine-tracklet-report", type=Path, default=root / "server_frame_fine_long_assoc_v008/frame_fine_tracklets_0000_0999_v008_v016_m3_gap60_v2/tracklet_report.json")
    parser.add_argument("--fine-tracklets-jsonl", type=Path, default=root / "server_frame_fine_long_assoc_v008/frame_fine_tracklets_0000_0999_v008_v016_m3_gap60_v2/tracklets.jsonl")
    parser.add_argument("--long-assoc-report", type=Path, default=root / "server_frame_fine_long_assoc_v008/frame_fine_tracklet_long_assoc_0000_0999_v008_gap60_v2_samecand_loose/long_association_report.json")
    parser.add_argument("--long-objects-jsonl", type=Path, default=root / "server_frame_fine_long_assoc_v008/frame_fine_tracklet_long_assoc_0000_0999_v008_gap60_v2_samecand_loose/long_objects.jsonl")
    parser.add_argument("--reviewed-merge-qa", type=Path, default=root / "server_frame_fine_cross_candidate_review_pack_v008_v2/vlm_review_qwen_compact_applied/qa_reviewed_merge_report.json")
    parser.add_argument("--reviewed-long-objects-jsonl", type=Path, default=root / "server_frame_fine_cross_candidate_review_pack_v008_v2/vlm_review_qwen_compact_applied/review_merged_long_objects.jsonl")
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dataset_delivery_manifest_0000_0999.json"
    md_path = args.output_dir / "dataset_delivery_manifest_0000_0999.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(manifest), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "passed": manifest["passed"]}, indent=2))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
