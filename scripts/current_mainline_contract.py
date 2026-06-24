"""Shared contract for the current dense semantic mainline.

Keep small, dependency-free constants here so launchers, validators, and review
index builders cannot drift into different definitions of rejected artifacts.
"""

from __future__ import annotations

from pathlib import Path


FORBIDDEN_ARTIFACT_SUBSTRINGS: tuple[str, ...] = (
    "frame_object_points_stride10.ply",
    "objects_v12_teacher_v20_grid6_unknown_absorb",
    "objects_v14_teacher_v20_grid6_geometry_guard_wall_recall",
    "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor",
    "objects_v16_teacher_v20_grid6_geometry_guard_surface_recall",
)


def forbidden_artifact_match(value: str | Path) -> str | None:
    text = str(value)
    for forbidden in FORBIDDEN_ARTIFACT_SUBSTRINGS:
        if forbidden in text:
            return forbidden
    return None

