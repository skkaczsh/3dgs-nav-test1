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
    concept = read_json(args.conceptseg_qa)
    route_decision = read_json(args.route_decision)
    concept_align = read_json(args.conceptseg_alignment)
    concept_intersection = read_json(args.conceptseg_intersection)
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
        file_entry(args.conceptseg_qa, "conceptseg_problem40_structured_qa", required=False),
        file_entry(args.conceptseg_contact_sheet, "conceptseg_problem40_contact_sheet", required=False),
        file_entry(args.route_decision, "dense_semantic_route_decision", required=True),
        file_entry(args.route_decision_md, "dense_semantic_route_decision_markdown", required=True),
        file_entry(args.conceptseg_alignment, "conceptseg_fine_object_alignment", required=False),
        file_entry(args.conceptseg_intersection, "conceptseg_instance_intersection", required=False),
        file_entry(args.conceptseg_instance_accepted_sheet, "conceptseg_instance_accepted_sheet", required=False),
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
            "conceptseg_items": concept.get("items"),
            "conceptseg_mode_counts": concept.get("mode_counts", {}),
            "route_decision": nested(route_decision, "main_route", "decision"),
            "conceptseg_decision": nested(route_decision, "conceptseg_side_track", "decision"),
            "conceptseg_fine_candidates": concept_align.get("item_count"),
            "conceptseg_semantically_discriminative_targets": concept_align.get("semantically_discriminative_target_count"),
            "conceptseg_instance_accepted_candidates": concept_intersection.get("accepted_candidate_count"),
            "conceptseg_instance_target_status_counts": concept_intersection.get("target_status_counts", {}),
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
            str(args.surface_first_voxel_ply),
            str(args.object_points_stride_ply),
        ],
        "next_actions": [
            "Use surface-first subcluster PLY for visual QA of surface contamination.",
            "Use residual surface assignment report to separate absorbable surface noise from unresolved fine residuals.",
            "Use v008 frame-target/tracklet/long-association artifacts for fine-object dataset evolution.",
            "Use reviewed long objects as the current fine-object object baseline.",
            "Keep ConceptSeg-R1 as constrained second-stage candidate generator.",
            "Use ConceptSeg-R1 only after strict instance-mask intersection; current accepted target coverage is low.",
            "Keep old route as visual color reference only.",
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
        f"- ConceptSeg modes: `{metrics.get('conceptseg_mode_counts')}`",
        f"- route decision: `{metrics.get('route_decision')}`",
        f"- ConceptSeg decision: `{metrics.get('conceptseg_decision')}`",
        f"- ConceptSeg instance accepted candidates: `{metrics.get('conceptseg_instance_accepted_candidates')}`",
        f"- ConceptSeg instance target status: `{metrics.get('conceptseg_instance_target_status_counts')}`",
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
    parser.add_argument("--conceptseg-qa", type=Path, default=root / "server_conceptseg_problem40/conceptseg_problem40_structured_qa.json")
    parser.add_argument("--conceptseg-contact-sheet", type=Path, default=root / "server_conceptseg_problem40/conceptseg_problem40_contact_sheet.jpg")
    parser.add_argument("--route-decision", type=Path, default=root / "route_status_20260610/dense_semantic_route_decision_20260611.json")
    parser.add_argument("--route-decision-md", type=Path, default=root / "route_status_20260610/dense_semantic_route_decision_20260611.md")
    parser.add_argument("--conceptseg-alignment", type=Path, default=root / "server_conceptseg_fine_object_alignment_v008/conceptseg_target_object_alignment_report.json")
    parser.add_argument("--conceptseg-intersection", type=Path, default=root / "server_conceptseg_instance_intersection_v008/conceptseg_instance_intersection_report.json")
    parser.add_argument("--conceptseg-instance-accepted-sheet", type=Path, default=root / "server_conceptseg_instance_intersection_v008/conceptseg_instance_accepted_sheet.jpg")
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
