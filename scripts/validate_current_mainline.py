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
from scripts.gate_cache_contract import resolve_relative_path, stale_gate_reasons
from scripts import validate_current_dense_patch_state
from scripts import validate_current_project_architecture
from scripts import validate_geometry_input_contract_usage
from scripts import validate_semantic_contract_usage


DEFAULT_ARCHITECTURE = REPO_ROOT / "docs" / "current_project_architecture.json"
DEFAULT_DENSE_STATE = REPO_ROOT / "docs" / "current_dense_patch_state.json"
DEFAULT_PROMOTION_GATE = REPO_ROOT / "docs" / "current_dense_promotion_gate.json"

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


def validate(args: argparse.Namespace) -> dict[str, Any]:
    architecture = validate_current_project_architecture.validate(args.architecture)
    dense_state = validate_current_dense_patch_state.validate(args.dense_patch_state)
    review_allowlist = build_current_dense_review_index.validate_artifact_allowlist()
    semantic_contract_usage = validate_semantic_contract_usage.validate()
    geometry_input_contract_usage = validate_geometry_input_contract_usage.validate()
    promotion_gate = validate_promotion_gate(args.promotion_gate)

    checks = {
        "architecture": architecture,
        "dense_patch_state": dense_state,
        "review_artifact_allowlist": review_allowlist,
        "semantic_contract_usage": semantic_contract_usage,
        "geometry_input_contract_usage": geometry_input_contract_usage,
        "promotion_gate_health": promotion_gate,
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
    parser.add_argument("--promotion-gate", type=Path, default=DEFAULT_PROMOTION_GATE)
    args = parser.parse_args()
    report = validate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
