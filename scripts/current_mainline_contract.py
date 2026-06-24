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
