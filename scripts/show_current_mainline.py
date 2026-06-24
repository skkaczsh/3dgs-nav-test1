#!/usr/bin/env python3
"""Print the current approved semantic/dense-patch mainline.

This is a read-only operator entry point.  It deliberately does not infer a
new route from latest files on disk; it reads the checked-in state files so
failed diagnostic runs do not silently become defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def summarize(architecture: dict[str, Any], dense_patch: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "dataset": architecture.get("dataset"),
        "decision": architecture.get("current_diagnosis", {}).get("decision"),
        "active_baselines": active,
        "dense_authoritative_source": dense_patch.get("authoritative_source", {}),
        "dense_patch_baseline": dense_patch.get("current_patch_baseline", {}),
        "dense_object_baseline": dense_patch.get("current_object_baseline", {}),
        "remote_executable_baseline": dense_patch.get("remote_executable_baseline", {}),
        "latest_remote_run": dense_patch.get("latest_remote_run", {}),
        "current_qa_report": dense_patch.get("current_qa_report", {}),
        "next_action": dense_patch.get("next_action", {}),
        "forbidden_inputs": dense_patch.get("forbidden_inputs", []),
        "rejected_semantic_artifacts": rejected,
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
    lines.append(f"  candidates: {latest_cand.get('candidate_count')}")
    lines.append(f"  accepted_candidate_rows: {latest_obj.get('accepted_candidate_rows')}")
    lines.append(f"  output_object_count: {latest_obj.get('output_object_count')}")
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
        for reason in qa.get("promotion_gate_current_reasons", []):
            lines.append(f"  blocked_by: {reason}")
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
    lines.append("forbidden inputs:")
    for item in summary.get("forbidden_inputs", []):
        lines.append(f"- {item.get('pattern')}: {item.get('reason')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=Path, default=Path("docs/current_project_architecture.json"))
    parser.add_argument("--dense-patch-state", type=Path, default=Path("docs/current_dense_patch_state.json"))
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    summary = summarize(load(args.architecture), load(args.dense_patch_state))
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
