#!/usr/bin/env python3
"""Validate the RTX 5070Ti parking candidate review manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_STATUS = "candidate_ready_for_visual_review"
EXPECTED_ROUTE_PARTS = (
    "guarded_v2",
    "ground artifact guard",
    "strict surface fusion",
    "object surface relabel",
)
EXPECTED_CANDIDATE = "ground_guard_object_relabel"
REQUIRED_COMMAND_SNIPPETS = (
    "run_rtx5070_parking_candidate_surface_route.sh",
    "pull_rtx5070_parking_candidate_surface_route.sh",
)
REQUIRED_FILE_ROLES = {
    "candidate_viewer_ply",
    "candidate_viewer_objects_jsonl",
    "candidate_viewer_export_report",
    "candidate_frame_local_qa_report",
    "surface_refinement_full_risk_compare_json",
    "surface_refinement_full_risk_compare_markdown",
    "candidate_object_relabel_report",
    "candidate_geometry_refine_summary",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str, value: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail, "value": value})


def validate_manifest(manifest: dict[str, Any], *, check_disk: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    dataset = manifest.get("dataset") or {}
    metrics = manifest.get("metrics") or {}
    files = manifest.get("files") or []
    commands = manifest.get("commands") or {}
    route = str(dataset.get("route") or "")
    candidate = str(dataset.get("candidate_name") or "")

    add_check(checks, "manifest_passed_flag", manifest.get("passed") is True, "builder marked manifest as passed", manifest.get("passed"))
    add_check(checks, "status_ready_for_review", manifest.get("status") == EXPECTED_STATUS, "manifest status is the expected review handoff state", manifest.get("status"))
    add_check(
        checks,
        "route_is_current_candidate",
        all(part in route for part in EXPECTED_ROUTE_PARTS),
        "route contains all current candidate stages",
        route,
    )
    add_check(checks, "candidate_name_current", candidate == EXPECTED_CANDIDATE, "candidate name matches current reviewed route", candidate)

    command_blob = "\n".join(str(value) for value in commands.values())
    add_check(
        checks,
        "commands_include_rebuild_and_pull_scripts",
        all(snippet in command_blob for snippet in REQUIRED_COMMAND_SNIPPETS),
        "manifest records fixed rebuild and pull commands",
        sorted(commands),
    )

    file_roles = {str(row.get("role") or "") for row in files}
    missing_roles = sorted(REQUIRED_FILE_ROLES - file_roles)
    add_check(checks, "required_roles_present", not missing_roles, "all expected artifact roles are listed", missing_roles)

    missing_required: list[str] = []
    zero_required: list[str] = []
    stale_disk_status: list[str] = []
    for row in files:
        role = str(row.get("role") or "")
        required = bool(row.get("required"))
        path = Path(str(row.get("path") or ""))
        manifest_exists = bool(row.get("exists"))
        manifest_bytes = as_int(row.get("bytes"), 0) or 0
        if required and not manifest_exists:
            missing_required.append(role)
        if required and manifest_bytes <= 0:
            zero_required.append(role)
        if check_disk and required:
            if not path.exists():
                stale_disk_status.append(f"{role}:missing_on_disk:{path}")
            elif path.stat().st_size <= 0:
                stale_disk_status.append(f"{role}:zero_on_disk:{path}")

    add_check(checks, "required_files_marked_present", not missing_required, "all required files are marked present in manifest", missing_required)
    add_check(checks, "required_files_nonzero", not zero_required, "all required files have nonzero manifest byte counts", zero_required)
    if check_disk:
        add_check(checks, "required_files_exist_on_disk", not stale_disk_status, "required file paths still exist on this machine", stale_disk_status)

    builder_failed_checks = [row.get("name") for row in manifest.get("checks", []) if not row.get("passed")]
    add_check(checks, "embedded_builder_checks_passed", not builder_failed_checks, "all builder checks passed", builder_failed_checks)

    output_vertices = as_int(nested(metrics, "viewer", "output_vertices"), 0) or 0
    missing_target_points = as_int(nested(metrics, "viewer", "missing_target_points"), -1)
    object_count = as_int(nested(metrics, "viewer", "object_count_with_points"), 0) or 0
    label_counts = nested(metrics, "viewer", "label_counts", default={}) or {}
    add_check(checks, "viewer_has_points", output_vertices > 0, "viewer PLY has exported points", output_vertices)
    add_check(checks, "viewer_has_objects", object_count > 0, "viewer export has objects with points", object_count)
    add_check(checks, "viewer_missing_target_points_zero", missing_target_points == 0, "viewer has no lost target point mapping", missing_target_points)
    add_check(checks, "viewer_has_surface_and_fine_labels", {"ground", "wall", "car", "railing"}.issubset(label_counts), "viewer label counts include core review labels", label_counts)

    all_risk_counts = nested(metrics, "qa", "all_risk_reason_counts", default={}) or {}
    add_check(checks, "qa_full_risk_counts_present", bool(all_risk_counts), "QA includes full-object risk counts", all_risk_counts)

    baseline_risk = as_int(nested(metrics, "comparison", "baseline_all_candidate_count"))
    candidate_risk = as_int(nested(metrics, "comparison", "candidate_all_candidate_count"))
    deltas = nested(metrics, "comparison", "candidate_deltas_from_baseline", default={}) or {}
    add_check(
        checks,
        "candidate_all_risk_not_worse",
        baseline_risk is not None and candidate_risk is not None and candidate_risk <= baseline_risk,
        "candidate all-risk count is no worse than strict-surface baseline",
        {"baseline": baseline_risk, "candidate": candidate_risk},
    )
    add_check(
        checks,
        "surface_risk_deltas_improve",
        as_int(deltas.get("ground_has_large_height_span"), 1) <= 0
        and as_int(deltas.get("wall_too_flat_low_height"), 0) < 0
        and as_int(deltas.get("wall_normal_too_up"), 0) < 0,
        "candidate improves or preserves key surface-risk deltas",
        deltas,
    )

    geometry_missing = as_int(nested(metrics, "geometry_refine", "missing_target_points"), -1)
    add_check(checks, "geometry_refine_missing_points_zero", geometry_missing == 0, "geometry refinement preserved target point mapping", geometry_missing)

    for row in checks:
        if not row["passed"]:
            errors.append(f"{row['name']}: {row['detail']} value={row.get('value')}")

    if manifest.get("git_head") in ("", None):
        warnings.append("manifest git_head is empty")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "check_disk": check_disk,
        "checks": checks,
        "summary": {
            "status": manifest.get("status"),
            "candidate": candidate,
            "output_vertices": output_vertices,
            "object_count_with_points": object_count,
            "baseline_all_candidate_count": baseline_risk,
            "candidate_all_candidate_count": candidate_risk,
            "surface_risk_deltas": deltas,
            "required_file_count": sum(1 for row in files if row.get("required")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo / "server_parking_priority_s10" / "parking_candidate_manifest_rtx5070" / "manifest.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-disk-check", action="store_true")
    args = parser.parse_args()

    result = validate_manifest(read_json(args.manifest), check_disk=not args.no_disk_check)
    result["manifest"] = str(args.manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
