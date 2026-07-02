#!/usr/bin/env python3
"""Validate the dense raw-data Patch/Object state file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.current_mainline_contract import APPROVED_MAINLINE_RUNNER_PATHS, FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS


REQUIRED_SCHEMA = "current-dense-patch-state/v1"
REQUIRED_TOP_LEVEL = {
    "schema",
    "updated_at",
    "dataset",
    "authoritative_source",
    "derived_dense_input",
    "current_patch_baseline",
    "current_object_baseline",
    "remote_executable_baseline",
    "latest_remote_run",
    "current_promotion_candidate",
    "current_qa_report",
    "stage_contract",
    "approved_runners",
    "forbidden_inputs",
    "next_action",
}

REQUIRED_FORBIDDEN_PATTERNS = set(FORBIDDEN_PRODUCTION_INPUT_SUBSTRINGS)

REQUIRED_STAGE_RULES = {
    "dense_source",
    "patch_generation",
    "patch_boundary_optimization",
    "object_building",
    "semantic_evidence",
}

REQUIRED_APPROVED_RUNNERS = set(APPROVED_MAINLINE_RUNNER_PATHS)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("dense patch state must be a JSON object")
    return data


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
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

    approved_runners = {str(item.get("path")) for item in data.get("approved_runners", []) if isinstance(item, dict)}
    for runner in sorted(REQUIRED_APPROVED_RUNNERS - approved_runners):
        errors.append(f"missing_approved_runner={runner}")
    for runner in sorted(approved_runners):
        runner_path = REPO_ROOT / runner
        if not runner_path.exists():
            errors.append(f"approved_runner_missing_file={runner}")

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

    remote = data.get("remote_executable_baseline", {})
    if isinstance(remote, dict):
        remote_paths = [str(item) for item in remote.get("remote_paths", [])]
        if not any(item.endswith("_cpp_region_grower_input.bin") for item in remote_paths):
            errors.append("remote_baseline_missing_region_input")
        if not any(item.endswith("_labels.bin") for item in remote_paths):
            errors.append("remote_baseline_missing_patch_labels")
        metrics = remote.get("metrics", {})
        if isinstance(metrics, dict):
            if int(metrics.get("r4_region_voxel_count", 0)) < 10_000_000:
                errors.append("remote_baseline_voxel_count_too_low")
            if int(metrics.get("attach_v4_output_patch_count", 0)) <= 0:
                errors.append("remote_baseline_patch_count_missing")

    latest = data.get("latest_remote_run", {})
    if isinstance(latest, dict):
        if latest.get("status") != "completed":
            errors.append("latest_remote_run_not_completed")
        object_metrics = latest.get("object_metrics", {})
        if isinstance(object_metrics, dict):
            if int(object_metrics.get("output_object_count", 0)) <= 0:
                errors.append("latest_remote_run_missing_output_objects")
            if int(object_metrics.get("accepted_candidate_rows", 0)) <= 0:
                errors.append("latest_remote_run_no_accepted_candidates")
        candidate_metrics = latest.get("candidate_metrics", {})
        if isinstance(candidate_metrics, dict):
            if int(candidate_metrics.get("structural_multimaterial_candidates", 0)) <= 0:
                errors.append("latest_remote_run_no_structural_candidates")
        if latest.get("promotion_status") == "diagnostic_not_promoted":
            if "verify_latest_remote_dense_run.py" not in str(latest.get("verification_command", "")):
                errors.append("latest_diagnostic_missing_verification_command")
            if "Keep v8 as the current visual-promotion candidate" not in str(latest.get("interpretation", "")):
                errors.append("latest_diagnostic_missing_v8_candidate_interpretation")

    promotion_candidate = data.get("current_promotion_candidate", {})
    if isinstance(promotion_candidate, dict):
        candidate_id = str(promotion_candidate.get("id", ""))
        qa_candidate_id = str(promotion_candidate.get("qa_candidate_id", candidate_id))
        if not candidate_id:
            errors.append("current_promotion_candidate_missing_id")
        if not qa_candidate_id:
            errors.append("current_promotion_candidate_missing_qa_candidate_id")
        if candidate_id != "v8_object_refinement":
            errors.append(f"unexpected_current_promotion_candidate={candidate_id}")
        if promotion_candidate.get("status") != "awaiting_required_visual_checks":
            errors.append(f"unexpected_current_promotion_candidate_status={promotion_candidate.get('status')}")
        if "visual checks" not in str(promotion_candidate.get("reason", "")):
            errors.append("current_promotion_candidate_reason_missing_visual_gate")
        if isinstance(latest, dict) and latest.get("id") != promotion_candidate.get("source_run_id"):
            if latest.get("promotion_status") != "diagnostic_not_promoted":
                errors.append("latest_differs_from_candidate_but_not_diagnostic")
        for key in ("gate_json", "visual_acceptance_json", "qa_json"):
            value = promotion_candidate.get(key)
            if not value:
                errors.append(f"current_promotion_candidate_missing_{key}")
                continue
            candidate_path = Path(str(value))
            if not candidate_path.is_absolute():
                candidate_path = path.parent / ".." / candidate_path
            candidate_path = candidate_path.resolve()
            if not candidate_path.exists():
                errors.append(f"current_promotion_candidate_path_missing={value}")
                continue
            linked_json = read_optional_json(candidate_path)
            if not isinstance(linked_json, dict):
                continue
            if key == "gate_json":
                if linked_json.get("candidate") != candidate_id:
                    errors.append(
                        f"promotion_candidate_gate_mismatch={linked_json.get('candidate')}!={candidate_id}"
                    )
                if linked_json.get("status") != "fail":
                    warnings.append(f"promotion_candidate_gate_status={linked_json.get('status')}")
            elif key == "visual_acceptance_json":
                if linked_json.get("accepted_candidate") != candidate_id:
                    errors.append(
                        "promotion_candidate_visual_acceptance_mismatch="
                        f"{linked_json.get('accepted_candidate')}!={candidate_id}"
                    )
                if linked_json.get("status") != "pending":
                    warnings.append(f"promotion_candidate_visual_status={linked_json.get('status')}")
            elif key == "qa_json":
                object_refinement = linked_json.get("object_refinement", {})
                if isinstance(object_refinement, dict) and object_refinement.get("candidate") != qa_candidate_id:
                    errors.append(
                        f"promotion_candidate_qa_mismatch={object_refinement.get('candidate')}!={qa_candidate_id}"
                    )

    next_action = data.get("next_action", {})
    if isinstance(next_action, dict):
        for key in ("runner", "remote_runner"):
            value = str(next_action.get(key, ""))
            if not value:
                errors.append(f"next_action_missing_{key}")
            elif value not in approved_runners:
                errors.append(f"next_action_unapproved_{key}={value}")

    qa = data.get("current_qa_report", {})
    if isinstance(qa, dict):
        qa_json_data: dict[str, Any] | None = None
        for key in (
            "json_path",
            "markdown_path",
            "review_index_html",
            "promotion_gate_json",
            "visual_acceptance_markdown",
        ):
            value = qa.get(key)
            if not value:
                errors.append(f"current_qa_report_missing_{key}")
                continue
            path_value = Path(str(value))
            if not path_value.is_absolute():
                path_value = path.parent / ".." / path_value
            if not path_value.resolve().exists():
                errors.append(f"current_qa_report_path_missing={value}")
            elif key == "json_path":
                qa_json_data = read_optional_json(path_value.resolve())
        findings = qa.get("key_findings", {})
        if isinstance(findings, dict):
            if findings.get("v17_label_point_delta_vs_v9_all_zero") is not True:
                errors.append("current_qa_report_surface_guard_not_stable")
            if float(findings.get("v8_mixed_object_voxel_ratio_delta_vs_v7", 1.0)) > 0:
                errors.append("current_qa_report_overlap_regressed")
        gate_status = qa.get("promotion_gate_status")
        if gate_status not in {
            "awaiting_visual_acceptance",
            "awaiting_required_visual_checks",
            "accepted",
            "rejected",
        }:
            errors.append(f"unexpected_promotion_gate_status={gate_status}")
        if "update_current_dense_visual_acceptance.py" not in str(qa.get("visual_acceptance_update_command", "")):
            errors.append("missing_visual_acceptance_update_command")
        if "gate_current_dense_mainline_promotion.py" not in str(qa.get("visual_acceptance_gate_command", "")):
            errors.append("missing_visual_acceptance_gate_command")
        if isinstance(qa_json_data, dict) and isinstance(latest, dict):
            v8_metrics = (
                qa_json_data.get("object_refinement", {})
                .get("metrics", {})
                .get("v8", {})
            )
            latest_objects = latest.get("object_metrics", {})
            if isinstance(v8_metrics, dict) and isinstance(latest_objects, dict):
                latest_accepted = int(latest_objects.get("accepted_candidate_rows", 0))
                v8_accepted = int(v8_metrics.get("accepted_candidate_rows", 0))
                latest_output_objects = int(latest_objects.get("output_object_count", 0))
                v8_output_objects = int(v8_metrics.get("output_object_count", 0))
                latest_is_weaker = latest_accepted < v8_accepted or latest_output_objects > v8_output_objects
                if latest_is_weaker and latest.get("promotion_status") != "diagnostic_not_promoted":
                    errors.append("latest_weaker_than_v8_but_not_diagnostic")

    return {
        "passed": not errors,
        "path": str(path),
        "checked_local_path_count": len(all_paths),
        "stage_contract_count": len(stages),
        "approved_runner_count": len(approved_runners),
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
