#!/usr/bin/env python3
"""Print the current approved semantic/dense-patch mainline.

This is a read-only operator entry point.  It deliberately does not infer a
new route from latest files on disk; it reads the checked-in state files so
failed diagnostic runs do not silently become defaults.
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

from scripts import validate_current_mainline
from scripts import validate_production_inputs


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return REPO_ROOT / path


def load(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def summarize(
    architecture: dict[str, Any],
    dense_patch: dict[str, Any],
    *,
    architecture_path: Path | None = None,
    dense_patch_state_path: Path | None = None,
) -> dict[str, Any]:
    active = [
        {
            "id": item.get("id"),
            "status": item.get("status"),
            "description": item.get("description"),
        }
        for item in architecture.get("active_baselines", [])
        if isinstance(item, dict)
    ]
    rejected = [
        {
            "id": item.get("id"),
            "reason": item.get("reason"),
        }
        for item in architecture.get("rejected_artifacts", [])
        if isinstance(item, dict)
    ]
    production_input_allowlist = {}
    if dense_patch_state_path is not None:
        production_input_allowlist = validate_production_inputs.validate_dense_allowlist(resolve_path(dense_patch_state_path))
    state_consistency = {}
    if architecture_path is not None and dense_patch_state_path is not None:
        state_consistency = validate_current_mainline.validate_state_consistency(
            resolve_path(architecture_path),
            resolve_path(dense_patch_state_path),
        )

    promotion_candidate = dense_patch.get("current_promotion_candidate", {})
    qa_report = dense_patch.get("current_qa_report", {})

    return {
        "dataset": architecture.get("dataset"),
        "decision": architecture.get("current_diagnosis", {}).get("decision"),
        "active_baselines": active,
        "dense_authoritative_source": dense_patch.get("authoritative_source", {}),
        "dense_patch_baseline": dense_patch.get("current_patch_baseline", {}),
        "dense_object_baseline": dense_patch.get("current_object_baseline", {}),
        "remote_executable_baseline": dense_patch.get("remote_executable_baseline", {}),
        "latest_remote_run": dense_patch.get("latest_remote_run", {}),
        "current_promotion_candidate": promotion_candidate,
        "current_qa_report": qa_report,
        "approved_runners": dense_patch.get("approved_runners", []),
        "next_action": dense_patch.get("next_action", {}),
        "forbidden_inputs": dense_patch.get("forbidden_inputs", []),
        "rejected_semantic_artifacts": rejected,
        "production_input_allowlist": production_input_allowlist,
        "state_consistency": state_consistency,
    }


def format_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"dataset: {summary.get('dataset')}")
    lines.append(f"decision: {summary.get('decision')}")
    lines.append("")
    lines.append("active baselines:")
    for item in summary.get("active_baselines", []):
        lines.append(f"- {item.get('id')} [{item.get('status')}]")
    lines.append("")
    source = summary.get("dense_authoritative_source", {})
    lines.append("dense source:")
    lines.append(f"- {source.get('id')} ({source.get('type')})")
    for path in source.get("local_paths", []):
        lines.append(f"  local: {path}")
    lines.append("")
    patch = summary.get("dense_patch_baseline", {})
    patch_metrics = patch.get("metrics", {})
    lines.append("current dense patch baseline:")
    lines.append(f"- {patch.get('id')} [{patch.get('status')}]")
    lines.append(f"  output_patch_count: {patch_metrics.get('output_patch_count')}")
    lines.append(f"  preview_points_stride10: {patch_metrics.get('preview_points_stride10')}")
    lines.append("")
    obj = summary.get("dense_object_baseline", {})
    obj_metrics = obj.get("metrics", {})
    lines.append("current object baseline:")
    lines.append(f"- {obj.get('id')} [{obj.get('status')}]")
    lines.append(f"  output_object_count: {obj_metrics.get('output_object_count')}")
    lines.append("")
    remote = summary.get("remote_executable_baseline", {})
    remote_metrics = remote.get("metrics", {})
    lines.append("remote executable baseline:")
    lines.append(f"- {remote.get('id')} [{remote.get('status')}] on {remote.get('host')}")
    lines.append(f"  r4_region_voxel_count: {remote_metrics.get('r4_region_voxel_count')}")
    lines.append(f"  attach_v4_output_patch_count: {remote_metrics.get('attach_v4_output_patch_count')}")
    lines.append("")
    latest = summary.get("latest_remote_run", {})
    latest_obj = latest.get("object_metrics", {})
    latest_cand = latest.get("candidate_metrics", {})
    lines.append("latest remote run:")
    lines.append(f"- {latest.get('id')} [{latest.get('status')}]")
    if latest.get("promotion_status"):
        lines.append(f"  promotion_status: {latest.get('promotion_status')}")
    lines.append(f"  candidates: {latest_cand.get('candidate_count')}")
    lines.append(f"  accepted_candidate_rows: {latest_obj.get('accepted_candidate_rows')}")
    lines.append(f"  output_object_count: {latest_obj.get('output_object_count')}")
    if latest.get("interpretation"):
        lines.append(f"  interpretation: {latest.get('interpretation')}")
    lines.append("")
    promotion_candidate = summary.get("current_promotion_candidate", {})
    if promotion_candidate:
        lines.append("current promotion candidate:")
        lines.append(f"- {promotion_candidate.get('id')} [{promotion_candidate.get('status')}]")
        if promotion_candidate.get("qa_candidate_id"):
            lines.append(f"  qa_candidate_id: {promotion_candidate.get('qa_candidate_id')}")
        if promotion_candidate.get("source_run_id"):
            lines.append(f"  source_run_id: {promotion_candidate.get('source_run_id')}")
        lines.append(f"  gate_json: {promotion_candidate.get('gate_json')}")
        lines.append(f"  visual_acceptance_json: {promotion_candidate.get('visual_acceptance_json')}")
        lines.append(f"  qa_json: {promotion_candidate.get('qa_json')}")
        if promotion_candidate.get("reason"):
            lines.append(f"  reason: {promotion_candidate.get('reason')}")
        lines.append("")
    qa = summary.get("current_qa_report", {})
    if qa:
        lines.append("current QA / promotion gate:")
        lines.append(f"- qa_report: {qa.get('markdown_path')}")
        lines.append(f"  review_index: {qa.get('review_index_url') or qa.get('review_index_html')}")
        lines.append(f"  promotion_gate_status: {qa.get('promotion_gate_status')}")
        lines.append(f"  visual_acceptance: {qa.get('visual_acceptance_expected_path')}")
        if qa.get("visual_acceptance_update_command"):
            lines.append(f"  update_command: {qa.get('visual_acceptance_update_command')}")
        if qa.get("visual_acceptance_gate_command"):
            lines.append(f"  gate_command: {qa.get('visual_acceptance_gate_command')}")
        lines.append("  promotion_plan_command: python3 scripts/plan_current_dense_promotion.py")
        for reason in qa.get("promotion_gate_current_reasons", qa.get("blocked_by", [])):
            lines.append(f"  blocked_by: {reason}")
        allowlist = qa.get("review_artifact_allowlist", {})
        if allowlist:
            lines.append(
                f"  review_allowlist: passed={allowlist.get('passed')} "
                f"artifacts={allowlist.get('artifact_ids')}"
            )
        rejected = qa.get("rejected_guard_diagnostics", {})
        if rejected:
            lines.append(
                f"  rejected_guard_baseline: {rejected.get('baseline')} "
                f"unknown_points={rejected.get('baseline_unknown_points')}"
            )
            for row in rejected.get("variants", []):
                top_reasons = ", ".join(
                    f"{item.get('reason')}={item.get('count')}"
                    for item in row.get("top_reasons", [])[:2]
                )
                lines.append(
                    f"  rejected_guard: {row.get('id')} "
                    f"unknown_delta_vs_v9={row.get('unknown_delta_vs_v9')} "
                    f"top={top_reasons}"
                )
        lines.append("")
    allowlist = summary.get("production_input_allowlist", {})
    if allowlist:
        lines.append("production input allowlist:")
        lines.append(
            f"- passed={allowlist.get('passed')} "
            f"allowed_count={allowlist.get('allowed_count')} "
            f"output_artifact_count={allowlist.get('output_artifact_count')}"
        )
        for error in allowlist.get("errors", []):
            lines.append(f"  error: {error}")
        for warning in allowlist.get("warnings", []):
            lines.append(f"  warning: {warning}")
        lines.append("")
    state_consistency = summary.get("state_consistency", {})
    if state_consistency:
        lines.append("state consistency:")
        lines.append(
            f"- passed={state_consistency.get('passed')} "
            f"dataset={state_consistency.get('dataset')}"
        )
        lines.append(f"  architecture: {state_consistency.get('architecture')}")
        lines.append(f"  dense_patch_state: {state_consistency.get('dense_patch_state')}")
        for error in state_consistency.get("errors", []):
            lines.append(f"  error: {error}")
        for warning in state_consistency.get("warnings", []):
            lines.append(f"  warning: {warning}")
        lines.append("")
    next_action = summary.get("next_action", {})
    lines.append("next action:")
    lines.append(f"- {next_action.get('id')}: {next_action.get('description')}")
    if next_action.get("runner"):
        lines.append(f"  runner: {next_action.get('runner')}")
    if next_action.get("remote_runner"):
        lines.append(f"  remote_runner: {next_action.get('remote_runner')}")
    if next_action.get("current_blocker"):
        lines.append(f"  blocker: {next_action.get('current_blocker')}")
    for item in next_action.get("success_criteria", []):
        lines.append(f"  gate: {item}")
    lines.append("")
    lines.append("approved runners:")
    for item in summary.get("approved_runners", []):
        lines.append(f"- {item.get('path')} [{item.get('stage')}]")
        if item.get("scope"):
            lines.append(f"  scope: {item.get('scope')}")
    lines.append("")
    lines.append("forbidden inputs:")
    for item in summary.get("forbidden_inputs", []):
        lines.append(f"- {item.get('pattern')}: {item.get('reason')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=Path, default=Path("docs/current_project_architecture.json"))
    parser.add_argument("--dense-patch-state", type=Path, default=Path("docs/current_dense_patch_state.json"))
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--json", action="store_true", help="Alias for --format json")
    args = parser.parse_args()

    output_format = "json" if args.json else args.format
    summary = summarize(
        load(args.architecture),
        load(args.dense_patch_state),
        architecture_path=args.architecture,
        dense_patch_state_path=args.dense_patch_state,
    )
    if output_format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
