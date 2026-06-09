#!/usr/bin/env python3
"""Summarize the current dense semantic route status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def file_state(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def build_status(args: argparse.Namespace) -> dict:
    connectivity = read_json(args.connectivity)
    stage = read_json(args.stage_summary)
    delivery = read_json(args.delivery_manifest)
    old_route = read_json(args.old_route_summary)
    conceptseg = file_state(args.conceptseg_report)
    offline_qa_report = getattr(args, "offline_qa_report", None)
    offline_qa = read_json(offline_qa_report) if offline_qa_report else {}
    resume_command_validation_report = getattr(args, "resume_command_validation", None)
    resume_command_validation = read_json(resume_command_validation_report) if resume_command_validation_report else {}
    resume_output_validation_report = getattr(args, "resume_output_validation", None)
    resume_output_validation = read_json(resume_output_validation_report) if resume_output_validation_report else {}
    return {
        "connectivity": connectivity,
        "offline_qa": {
            "report": str(offline_qa_report) if offline_qa_report else "",
            "passed": offline_qa.get("passed"),
            "git_head": offline_qa.get("git_head"),
            "timestamp": offline_qa.get("timestamp"),
            "checks": offline_qa.get("checks", []),
        },
        "server_resume_command_plan": {
            "validation_report": str(resume_command_validation_report) if resume_command_validation_report else "",
            "passed": resume_command_validation.get("passed"),
            "errors": resume_command_validation.get("errors", []),
            "warnings": resume_command_validation.get("warnings", []),
            "phase_ids": resume_command_validation.get("phase_ids", []),
            "required_local_scripts": resume_command_validation.get("required_local_scripts", []),
        },
        "server_resume_outputs": {
            "validation_report": str(resume_output_validation_report) if resume_output_validation_report else "",
            "passed": resume_output_validation.get("passed"),
            "blockers": resume_output_validation.get("blockers", []),
            "next_gate": resume_output_validation.get("next_gate"),
            "checks": resume_output_validation.get("checks", []),
        },
        "main_route": {
            "stage_summary": str(args.stage_summary),
            "stage_status": stage.get("stage_status", {}),
            "review_items": stage.get("review_items"),
            "manual_workflow_pending": stage.get("manual_workflow_pending", {}),
            "manual_merge_qa": stage.get("manual_merge_qa", {}),
        },
        "delivery": {
            "manifest": str(args.delivery_manifest),
            "file_count": delivery.get("file_count"),
            "missing": delivery.get("missing", []),
            "zip": str(args.delivery_zip),
            "zip_exists": args.delivery_zip.exists(),
        },
        "new_model_side_track": {
            "name": "ConceptSeg-R1",
            "report": conceptseg,
            "status": "side_track_only",
            "reason": "smoke/review evidence has not beaten sam2_prompt_v3_sky_label_merge_completion",
        },
        "old_route_side_track": {
            "summary": str(args.old_route_summary),
            "exists": args.old_route_summary.exists(),
            "status": "visual_reference_only",
            "summary_payload": old_route,
        },
        "next_actions": [
            "Restore server connectivity or bind address before Qwen/server-side work.",
            "Run scripts/resume_server_qwen_review.sh when scan-train is reachable.",
            "Use delivery HTML/manual CSV fallback if Qwen remains unavailable.",
            "Apply reviewed merges only through run_manual_merge_review_workflow.py so QA runs automatically.",
            "Keep ConceptSeg-R1 and old route as side tracks until they pass the same reviewed-object QA gates.",
        ],
    }


def render_markdown(status: dict) -> str:
    conn = status["connectivity"]
    offline_qa = status.get("offline_qa", {})
    resume_plan = status.get("server_resume_command_plan", {})
    resume_outputs = status.get("server_resume_outputs", {})
    stage = status["main_route"]["stage_status"]
    delivery = status["delivery"]
    qa = status["main_route"].get("manual_merge_qa", {})
    lines = [
        "# Dense Semantic Route Status",
        "",
        "## Server Connectivity",
        "",
        f"- all reachable: `{conn.get('all_reachable')}`",
    ]
    for host in conn.get("hosts", []):
        cfg = host.get("ssh_config", {})
        tcp = host.get("tcp", {})
        lines.append(
            f"- `{host.get('host')}` {cfg.get('hostname')}:{cfg.get('port')} "
            f"reachable=`{tcp.get('reachable')}` bind_present=`{host.get('bind_address_present_locally')}` error=`{tcp.get('error')}`"
        )
    lines.extend(
        [
            "",
            "## Offline QA",
            "",
            f"- passed: `{offline_qa.get('passed')}`",
            f"- git head: `{offline_qa.get('git_head')}`",
            f"- timestamp: `{offline_qa.get('timestamp')}`",
            f"- checks: `{offline_qa.get('checks')}`",
            "",
            "## Server Resume Plan",
            "",
            f"- validation passed: `{resume_plan.get('passed')}`",
            f"- phase order: `{resume_plan.get('phase_ids')}`",
            f"- errors: `{resume_plan.get('errors')}`",
            f"- warnings: `{resume_plan.get('warnings')}`",
            "",
            "## Server Resume Outputs",
            "",
            f"- validation passed: `{resume_outputs.get('passed')}`",
            f"- blockers: `{resume_outputs.get('blockers')}`",
            f"- next gate: `{resume_outputs.get('next_gate')}`",
            "",
            "## Main Route",
            "",
            f"- review pack ready: `{stage.get('review_pack_ready')}`",
            f"- contact sheets ready: `{stage.get('contact_sheets_ready')}`",
            f"- manual HTML ready: `{stage.get('manual_html_ready')}`",
            f"- pending apply safe: `{stage.get('pending_apply_safe')}`",
            f"- Qwen review ready: `{stage.get('qwen_review_ready')}`",
            f"- QA passed: `{qa.get('passed')}`",
            f"- input/output points: `{qa.get('input_point_count')}` / `{qa.get('output_point_count')}`",
            "",
            "## Delivery",
            "",
            f"- manifest: `{delivery.get('manifest')}`",
            f"- zip: `{delivery.get('zip')}` exists=`{delivery.get('zip_exists')}`",
            f"- file count: `{delivery.get('file_count')}`",
            f"- missing: `{delivery.get('missing')}`",
            "",
            "## Side Tracks",
            "",
            f"- ConceptSeg-R1: `{status['new_model_side_track']['status']}`. {status['new_model_side_track']['reason']}.",
            f"- Old route: `{status['old_route_side_track']['status']}`.",
            "",
            "## Next Actions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in status["next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectivity", type=Path, required=True)
    parser.add_argument("--stage-summary", type=Path, required=True)
    parser.add_argument("--delivery-manifest", type=Path, required=True)
    parser.add_argument("--delivery-zip", type=Path, required=True)
    parser.add_argument("--conceptseg-report", type=Path, required=True)
    parser.add_argument("--old-route-summary", type=Path, required=True)
    parser.add_argument("--offline-qa-report", type=Path, default=None)
    parser.add_argument("--resume-command-validation", type=Path, default=None)
    parser.add_argument("--resume-output-validation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    status = build_status(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dense_semantic_route_status.json"
    md_path = args.output_dir / "dense_semantic_route_status.md"
    json_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(status), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
