#!/usr/bin/env python3
"""Validate the current dense semantic mainline operator state.

This is a health check, not a promotion command.  A candidate may still be
blocked on manual visual acceptance while the mainline itself is healthy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_current_dense_review_index
from scripts import gate_current_dense_mainline_promotion
from scripts import plan_current_dense_promotion
from scripts.gate_cache_contract import resolve_relative_path, stale_gate_reasons
from scripts import validate_approved_mainline_runners
from scripts import validate_current_dense_patch_state
from scripts import validate_current_project_architecture
from scripts import validate_geometry_input_contract_usage
from scripts import validate_pointcloud_supervised_baseline_smoke
from scripts import validate_pointcloud_supervised_smoke_manifest
from scripts import validate_production_input_guard_usage
from scripts import validate_production_inputs
from scripts import validate_semantic_contract_usage
from scripts.current_mainline_contract import REQUIRED_AUTHORITATIVE_SOURCE_ID, REQUIRED_DERIVED_DENSE_INPUT_ID


DEFAULT_ARCHITECTURE = REPO_ROOT / "docs" / "current_project_architecture.json"
DEFAULT_DENSE_STATE = REPO_ROOT / "docs" / "current_dense_patch_state.json"
DEFAULT_PROMOTION_GATE = REPO_ROOT / "docs" / "current_dense_promotion_gate.json"
DEFAULT_QA = REPO_ROOT / "docs" / "current_dense_mainline_qa.json"
DEFAULT_SUPERVISED_SMOKE = REPO_ROOT / "docs" / "pointcloud_supervised_baseline_smoke_20260708.json"
DEFAULT_SUPERVISED_MANIFEST = REPO_ROOT / "docs" / "pointcloud_supervised_baseline_smoke_manifest_20260708.json"

ALLOWED_VISUAL_PENDING_REASONS = (
    "visual_status_not_accepted=",
    "visual_required_checks_not_accepted=",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def recompute_promotion_gate(data: dict[str, Any]) -> dict[str, Any] | None:
    qa_json = resolve_relative_path(data.get("qa_json"), REPO_ROOT)
    if qa_json is None:
        return None
    visual_acceptance = resolve_relative_path(data.get("visual_acceptance"), REPO_ROOT)
    thresholds = data.get("thresholds") or {}
    args = argparse.Namespace(
        qa_json=qa_json,
        visual_acceptance=visual_acceptance,
        output=Path("/dev/null"),
        min_accepted_delta=number(thresholds.get("min_accepted_delta"), default=1.0),
        max_output_object_delta=number(thresholds.get("max_output_object_delta"), default=0.0),
        max_overlap_delta=number(thresholds.get("max_overlap_delta"), default=0.0),
        max_unknown_point_delta=number(thresholds.get("max_unknown_point_delta"), default=0.0),
        no_require_visual_acceptance=not bool(thresholds.get("require_visual_acceptance", True)),
    )
    return gate_current_dense_mainline_promotion.evaluate(args)


def validate_promotion_gate(path: Path) -> dict[str, Any]:
    data = read_json(path)
    errors: list[str] = []
    warnings: list[str] = []
    if data.get("schema") != "current-dense-promotion-gate/v1":
        errors.append(f"unexpected_promotion_gate_schema={data.get('schema')!r}")
    if data.get("candidate") != "v8_object_refinement":
        errors.append(f"unexpected_promotion_candidate={data.get('candidate')!r}")
    metrics = data.get("metrics") or {}
    if number(metrics.get("accepted_delta")) < 1.0:
        errors.append("promotion_gate_accepted_delta_not_positive")
    if number(metrics.get("output_object_delta"), default=1.0) > 0.0:
        errors.append("promotion_gate_output_object_count_regressed")
    if number(metrics.get("overlap_delta"), default=1.0) > 0.0:
        errors.append("promotion_gate_overlap_regressed")
    if number(metrics.get("unknown_point_delta"), default=1.0) > 0.0:
        errors.append("promotion_gate_unknown_spike")
    if metrics.get("nonzero_surface_delta"):
        errors.append(f"promotion_gate_surface_delta={metrics.get('nonzero_surface_delta')}")

    recomputed = recompute_promotion_gate(data)
    errors.extend(stale_gate_reasons(data, recomputed, prefix="promotion_gate"))

    reasons = [str(item) for item in data.get("reasons", [])]
    non_visual_reasons = [
        item
        for item in reasons
        if not any(item.startswith(prefix) for prefix in ALLOWED_VISUAL_PENDING_REASONS)
    ]
    if non_visual_reasons:
        errors.extend(f"promotion_gate_non_visual_failure={item}" for item in non_visual_reasons)
    elif data.get("status") == "fail" and reasons:
        warnings.append("promotion_candidate_waiting_for_visual_acceptance")
    elif data.get("status") not in {"pass", "fail"}:
        errors.append(f"unexpected_promotion_gate_status={data.get('status')!r}")
    return {
        "passed": not errors,
        "path": str(path),
        "status": data.get("status"),
        "candidate": data.get("candidate"),
        "metrics": metrics,
        "errors": errors,
        "warnings": warnings,
    }


def validate_promotion_plan(state_path: Path, qa_path: Path, gate_path: Path) -> dict[str, Any]:
    plan = plan_current_dense_promotion.build_plan(read_json(state_path), read_json(qa_path), read_json(gate_path))
    errors: list[str] = []
    warnings: list[str] = []
    if plan.get("schema") != "current-dense-promotion-plan/v1":
        errors.append(f"unexpected_promotion_plan_schema={plan.get('schema')!r}")
    if plan.get("candidate") != "v8_object_refinement":
        errors.append(f"unexpected_promotion_plan_candidate={plan.get('candidate')!r}")
    proposed = plan.get("proposed_object_baseline", {})
    if not isinstance(proposed, dict):
        errors.append("promotion_plan_missing_proposed_object_baseline")
        proposed = {}
    if proposed.get("id") != "v8_object_refinement":
        errors.append(f"promotion_plan_proposed_id_mismatch={proposed.get('id')!r}")
    if proposed.get("status") != "promoted_dense_object_geometry_baseline":
        errors.append(f"promotion_plan_unexpected_status={proposed.get('status')!r}")
    for path in proposed.get("local_paths", []):
        if "stride10" in str(path):
            errors.append(f"promotion_plan_stride10_in_production_path={path}")
    qa_paths = proposed.get("qa_only_paths", [])
    if not any("stride10" in str(path) for path in qa_paths):
        errors.append("promotion_plan_missing_stride10_qa_path")

    plan_errors = [str(item) for item in plan.get("errors", [])]
    if plan.get("passed"):
        warnings.append("promotion_plan_ready_to_apply_after_gate_pass")
    elif plan_errors == ["promotion_gate_not_passed=fail"]:
        warnings.append("promotion_plan_waiting_for_gate_pass")
    else:
        errors.extend(f"promotion_plan_error={item}" for item in plan_errors)
    return {
        "passed": not errors,
        "candidate": plan.get("candidate"),
        "gate_status": plan.get("gate_status"),
        "proposed_object_baseline_id": proposed.get("id"),
        "errors": errors,
        "warnings": warnings,
    }


def validate_state_consistency(architecture_path: Path, dense_state_path: Path) -> dict[str, Any]:
    architecture = read_json(architecture_path)
    dense_state = read_json(dense_state_path)
    errors: list[str] = []
    warnings: list[str] = []

    arch_dataset = architecture.get("dataset")
    state_dataset = dense_state.get("dataset")
    if arch_dataset != state_dataset:
        errors.append(f"dataset_mismatch={arch_dataset}!={state_dataset}")

    arch_dense_sources = {
        str(item.get("id")): item
        for item in architecture.get("dense_sources", [])
        if isinstance(item, dict)
    }
    if REQUIRED_AUTHORITATIVE_SOURCE_ID not in arch_dense_sources:
        errors.append(f"architecture_missing_authoritative_source={REQUIRED_AUTHORITATIVE_SOURCE_ID}")
    if REQUIRED_DERIVED_DENSE_INPUT_ID not in arch_dense_sources:
        errors.append(f"architecture_missing_derived_dense_input={REQUIRED_DERIVED_DENSE_INPUT_ID}")

    state_source_id = dense_state.get("authoritative_source", {}).get("id")
    state_derived_id = dense_state.get("derived_dense_input", {}).get("id")
    if state_source_id != REQUIRED_AUTHORITATIVE_SOURCE_ID:
        errors.append(f"dense_state_authoritative_source_mismatch={state_source_id}")
    if state_derived_id != REQUIRED_DERIVED_DENSE_INPUT_ID:
        errors.append(f"dense_state_derived_input_mismatch={state_derived_id}")

    arch_raw_paths = set(str(item) for item in arch_dense_sources.get(REQUIRED_AUTHORITATIVE_SOURCE_ID, {}).get("local_paths", []))
    state_raw_paths = set(str(item) for item in dense_state.get("authoritative_source", {}).get("local_paths", []))
    if not arch_raw_paths & state_raw_paths:
        errors.append("authoritative_source_path_disjoint")

    arch_dense_paths = set(str(item) for item in arch_dense_sources.get(REQUIRED_DERIVED_DENSE_INPUT_ID, {}).get("remote_paths", []))
    state_dense_paths = set(str(item) for item in dense_state.get("derived_dense_input", {}).get("remote_paths", []))
    if not arch_dense_paths & state_dense_paths:
        errors.append("derived_dense_input_path_disjoint")

    return {
        "schema": "current-mainline-state-consistency/v1",
        "passed": not errors,
        "architecture": str(architecture_path),
        "dense_patch_state": str(dense_state_path),
        "dataset": arch_dataset,
        "errors": errors,
        "warnings": warnings,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    architecture = validate_current_project_architecture.validate(args.architecture)
    dense_state = validate_current_dense_patch_state.validate(args.dense_patch_state)
    approved_runner_usage = validate_approved_mainline_runners.validate(args.dense_patch_state)
    review_allowlist = build_current_dense_review_index.validate_artifact_allowlist()
    semantic_contract_usage = validate_semantic_contract_usage.validate()
    geometry_input_contract_usage = validate_geometry_input_contract_usage.validate()
    production_input_guard_usage = validate_production_input_guard_usage.validate()
    production_input_allowlist = validate_production_inputs.validate_dense_allowlist(args.dense_patch_state)
    supervised_baseline_smoke = validate_pointcloud_supervised_baseline_smoke.validate(args.supervised_smoke)
    supervised_smoke_manifest = validate_pointcloud_supervised_smoke_manifest.validate(args.supervised_manifest)
    state_consistency = validate_state_consistency(args.architecture, args.dense_patch_state)
    promotion_gate = validate_promotion_gate(args.promotion_gate)
    promotion_plan = validate_promotion_plan(args.dense_patch_state, args.qa_json, args.promotion_gate)

    checks = {
        "architecture": architecture,
        "dense_patch_state": dense_state,
        "approved_runner_usage": approved_runner_usage,
        "review_artifact_allowlist": review_allowlist,
        "semantic_contract_usage": semantic_contract_usage,
        "geometry_input_contract_usage": geometry_input_contract_usage,
        "production_input_guard_usage": production_input_guard_usage,
        "production_input_allowlist": production_input_allowlist,
        "supervised_baseline_smoke": supervised_baseline_smoke,
        "supervised_smoke_manifest": supervised_smoke_manifest,
        "state_consistency": state_consistency,
        "promotion_gate_health": promotion_gate,
        "promotion_plan_health": promotion_plan,
    }
    errors: list[str] = []
    warnings: list[str] = []
    for name, report in checks.items():
        if not report.get("passed"):
            errors.extend(f"{name}:{item}" for item in report.get("errors", []))
        warnings.extend(f"{name}:{item}" for item in report.get("warnings", []))

    return {
        "schema": "current-mainline-healthcheck/v1",
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--dense-patch-state", type=Path, default=DEFAULT_DENSE_STATE)
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument("--promotion-gate", type=Path, default=DEFAULT_PROMOTION_GATE)
    parser.add_argument("--supervised-smoke", type=Path, default=DEFAULT_SUPERVISED_SMOKE)
    parser.add_argument("--supervised-manifest", type=Path, default=DEFAULT_SUPERVISED_MANIFEST)
    args = parser.parse_args()
    report = validate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
