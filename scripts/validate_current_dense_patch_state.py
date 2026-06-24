#!/usr/bin/env python3
"""Validate the dense raw-data Patch/Object state file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SCHEMA = "current-dense-patch-state/v1"
REQUIRED_TOP_LEVEL = {
    "schema",
    "updated_at",
    "dataset",
    "authoritative_source",
    "derived_dense_input",
    "current_patch_baseline",
    "current_object_baseline",
    "stage_contract",
    "forbidden_inputs",
    "next_action",
}

REQUIRED_FORBIDDEN_PATTERNS = {
    "frame_object_points_stride10.ply",
    "objects_v12_teacher_v20_grid6_unknown_absorb",
    "objects_v14_teacher_v20_grid6_geometry_guard_wall_recall",
    "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor",
    "objects_v16_teacher_v20_grid6_geometry_guard_surface_recall",
}

REQUIRED_STAGE_RULES = {
    "dense_source",
    "patch_generation",
    "patch_boundary_optimization",
    "object_building",
    "semantic_evidence",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("dense patch state must be a JSON object")
    return data


def iter_local_paths(data: Any) -> list[str]:
    found: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "local_paths" and isinstance(value, list):
                found.extend(str(item) for item in value)
            else:
                found.extend(iter_local_paths(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(iter_local_paths(item))
    return found


def validate(path: Path) -> dict[str, Any]:
    data = load_json(path)
    errors: list[str] = []
    warnings: list[str] = []

    missing_top = sorted(REQUIRED_TOP_LEVEL - set(data))
    errors.extend(f"missing_top_level={key}" for key in missing_top)

    if data.get("schema") != REQUIRED_SCHEMA:
        errors.append(f"unexpected_schema={data.get('schema')!r}")

    forbidden = {str(item.get("pattern")) for item in data.get("forbidden_inputs", []) if isinstance(item, dict)}
    for pattern in sorted(REQUIRED_FORBIDDEN_PATTERNS - forbidden):
        errors.append(f"missing_forbidden_pattern={pattern}")

    stages = {str(item.get("stage")) for item in data.get("stage_contract", []) if isinstance(item, dict)}
    for stage in sorted(REQUIRED_STAGE_RULES - stages):
        errors.append(f"missing_stage_contract={stage}")

    all_paths = iter_local_paths(data)
    for item in all_paths:
        if item.startswith("/") and not Path(item).exists():
            errors.append(f"missing_local_path={item}")

    source = data.get("authoritative_source", {})
    if isinstance(source, dict):
        if source.get("type") != "las":
            errors.append("authoritative_source_not_las")
        if int(source.get("known_point_count", 0)) < 90_000_000:
            errors.append("authoritative_source_point_count_too_low")

    derived = data.get("derived_dense_input", {})
    if isinstance(derived, dict):
        if abs(float(derived.get("voxel_size_m", 0.0)) - 0.03) > 1e-9:
            errors.append("derived_dense_input_not_voxel003")
        if int(derived.get("known_voxel_count", 0)) < 10_000_000:
            errors.append("derived_dense_input_voxel_count_too_low")

    patch = data.get("current_patch_baseline", {})
    if isinstance(patch, dict):
        if "v6" not in str(patch.get("id", "")):
            warnings.append("current_patch_baseline_is_not_v6")
        metrics = patch.get("metrics", {})
        if isinstance(metrics, dict) and int(metrics.get("output_patch_count", 0)) <= 0:
            errors.append("current_patch_output_patch_count_missing")

    return {
        "passed": not errors,
        "path": str(path),
        "checked_local_path_count": len(all_paths),
        "stage_contract_count": len(stages),
        "forbidden_input_count": len(forbidden),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        default="docs/current_dense_patch_state.json",
        help="Path to current_dense_patch_state.json",
    )
    args = parser.parse_args()
    report = validate(Path(args.state))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
