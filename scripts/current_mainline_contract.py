"""Shared contract for the current dense semantic mainline.

Keep small, dependency-free constants here so launchers, validators, and review
index builders cannot drift into different definitions of rejected artifacts.
"""

from __future__ import annotations

from pathlib import Path


REJECTED_ARTIFACT_SUBSTRINGS: tuple[str, ...] = (
    "objects_v12_teacher_v20_grid6_unknown_absorb",
    "objects_v14_teacher_v20_grid6_geometry_guard_wall_recall",
    "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor",
    "objects_v16_teacher_v20_grid6_geometry_guard_surface_recall",
)

FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS: tuple[str, ...] = (
    "frame_object_points_stride10.ply",
    *REJECTED_ARTIFACT_SUBSTRINGS,
)

# Backward-compatible name for review tools.  Use
# FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS when validating dense production inputs.
FORBIDDEN_ARTIFACT_SUBSTRINGS = REJECTED_ARTIFACT_SUBSTRINGS

REQUIRED_ACTIVE_BASELINE_IDS: tuple[str, ...] = (
    "pure_surface_visibility_full_0000_6180",
    "full_scene_objects_refined_v20",
    "objects_v9_teacher_v20_semantic",
    "objects_v17_teacher_v20_surface_preserve_guard",
)

REQUIRED_DENSE_SOURCE_IDS: tuple[str, ...] = (
    "raw_opt_las_local",
    "dense_las_voxel003_canonical",
)

REQUIRED_REJECTED_ARTIFACT_IDS: tuple[str, ...] = (
    "objects_v12_teacher_v20_grid6_unknown_absorb",
    "objects_v14_teacher_v20_grid6_geometry_guard_wall_recall",
    "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor",
    "objects_v16_teacher_v20_grid6_geometry_guard_surface_recall",
    "v23_mimo_rich_highctx_global_relabel",
    "old_transforms_json_project_world_points_route",
    "single_frame_keyframe_pairing_route",
    "raw_sam_png_vote_on_patches",
)

APPROVED_MAINLINE_RUNNER_PATHS: tuple[str, ...] = (
    "scripts/run_dense_patch_object_refinement_v7.py",
    "scripts/run_scan_train_dense_patch_object_refinement_v7.sh",
    "scripts/run_object_semantic_evidence_fusion.py",
    "scripts/run_semantic_evidence_pipeline.py",
    "scripts/run_validated_semantic_viewer_export.py",
)

PROTECTED_PRODUCTION_GUARD_SCRIPT_PATHS: tuple[str, ...] = (
    "scripts/run_object_semantic_evidence_fusion.py",
    "scripts/run_validated_semantic_viewer_export.py",
    "scripts/run_semantic_evidence_pipeline.py",
    "scripts/run_dense_patch_object_refinement_v7.py",
    "scripts/rewrite_viewer_ply_semantics.py",
    "scripts/transfer_teacher_semantics_to_objects.py",
    "scripts/accumulate_semantic_png_votes_to_objects.py",
)

PROTECTED_SEMANTIC_CONTRACT_SCRIPT_PATHS: tuple[str, ...] = (
    "scripts/analyze_residual_absorbability.py",
    "scripts/apply_geometry_conflict_relabels.py",
    "scripts/apply_priority_guard_to_full_scene.py",
    "scripts/apply_surface_trust_guard_to_ply.py",
    "scripts/apply_visual_promotion_geometry_guard.py",
    "scripts/build_parking_dataset_manifest.py",
    "scripts/build_spatial_partition_objects.py",
    "scripts/export_frame_target_objects_for_viewer.py",
    "scripts/project_semantic.py",
    "scripts/qa_object_voxel_overlap.py",
    "scripts/qa_viewer_candidate.py",
    "scripts/transfer_teacher_semantics_to_objects.py",
)

PROTECTED_GEOMETRY_INPUT_CONTRACT_SCRIPT_PATHS: tuple[str, ...] = (
    "scripts/accumulate_semantic_png_votes_to_objects.py",
    "scripts/transfer_teacher_semantics_to_objects.py",
)


def forbidden_artifact_match(value: str | Path) -> str | None:
    text = str(value)
    for forbidden in REJECTED_ARTIFACT_SUBSTRINGS:
        if forbidden in text:
            return forbidden
    return None


def forbidden_production_input_match(value: str | Path) -> str | None:
    text = str(value)
    for forbidden in FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS:
        if forbidden in text:
            return forbidden
    return None


def reject_forbidden_production_input(value: str | Path) -> None:
    forbidden = forbidden_production_input_match(value)
    if forbidden:
        raise ValueError(f"forbidden input path contains {forbidden}: {value}")
