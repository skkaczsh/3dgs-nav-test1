#!/usr/bin/env python3
"""Validate the generated server-resume command plan offline."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


EXPECTED_PHASE_ORDER = [
    "connectivity",
    "main_qwen_review",
    "main_semantic_refresh",
    "main_object_fusion",
    "main_output_validation",
    "new_model_side_track",
    "old_route_side_track",
]

REQUIRED_LOCAL_SCRIPTS = {
    "diagnose_connectivity": "scripts/diagnose_server_connectivity.py",
    "qwen_review": "scripts/resume_server_qwen_review.sh",
    "semantic_completion_sharded": "scripts/run_remote_server_semantic_completion_sharded.sh",
    "dataset_readiness": "scripts/run_server_dataset_readiness.sh",
    "target_object_fusion": "scripts/run_remote_server_target_object_fusion.sh",
    "strict_output_validation": "scripts/validate_server_resume_outputs.py",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def command_tokens(command: str) -> list[str]:
    tokens = shlex.split(command)
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        key, _, _ = tokens[0].partition("=")
        if not key.replace("_", "").isalnum():
            break
        tokens = tokens[1:]
    return tokens


def validate_plan(plan: dict, repo_root: Path, shell_path: Path | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    phases = plan.get("phases", [])
    phase_ids = [phase.get("id") for phase in phases]

    if phase_ids != EXPECTED_PHASE_ORDER:
        errors.append(f"phase_order={phase_ids}")

    readiness = plan.get("readiness", {})
    if readiness.get("blockers"):
        errors.append(f"readiness_blockers={readiness.get('blockers')}")
    if readiness.get("ready_for_server_probe") is not True:
        errors.append("ready_for_server_probe_not_true")

    commands_by_name = {}
    for phase in phases:
        phase_id = phase.get("id", "")
        for row in phase.get("commands", []):
            name = row.get("name", "")
            commands_by_name[name] = row
            required = bool(row.get("required", False))
            if phase_id.startswith("main_") or phase_id == "connectivity":
                if not required:
                    errors.append(f"main_phase_command_not_required={phase_id}:{name}")
            if phase_id.endswith("side_track") and required:
                errors.append(f"side_track_command_required={phase_id}:{name}")

    for name, script in REQUIRED_LOCAL_SCRIPTS.items():
        row = commands_by_name.get(name)
        if not row:
            errors.append(f"missing_command={name}")
            continue
        if script not in row.get("command", ""):
            errors.append(f"command_missing_script={name}:{script}")
        if not (repo_root / script).exists():
            errors.append(f"missing_local_script={script}")
        tokens = command_tokens(row.get("command", ""))
        if not tokens:
            errors.append(f"empty_command={name}")
            continue
        if tokens[0] not in {"python3", "bash"}:
            errors.append(f"unexpected_required_executor={name}:{tokens[0]}")

    qwen = commands_by_name.get("qwen_review", {}).get("command", "")
    if "CONCURRENCY=4" not in qwen:
        errors.append("qwen_concurrency_not_4")
    semantic = commands_by_name.get("semantic_completion_sharded", {}).get("command", "")
    if "PATCH_SCENE_PROMPTS=1" not in semantic:
        errors.append("scene_prompt_patch_not_enabled")
    if "SHARDS=4" not in semantic:
        errors.append("semantic_shards_not_4")
    strict_validation = commands_by_name.get("strict_output_validation", {}).get("command", "")
    if "--strict" not in strict_validation:
        errors.append("strict_output_validation_not_strict")

    if shell_path is not None:
        if not shell_path.exists():
            errors.append(f"missing_shell_plan={shell_path}")
        else:
            shell = shell_path.read_text(encoding="utf-8")
            for name, script in REQUIRED_LOCAL_SCRIPTS.items():
                if script not in shell:
                    errors.append(f"shell_missing_script={name}:{script}")
            if "[optional] conceptseg_status" not in shell:
                warnings.append("shell_missing_optional_conceptseg_echo")
            if "[optional] old_route_smoke_status" not in shell:
                warnings.append("shell_missing_optional_old_route_echo")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "phase_ids": phase_ids,
        "required_local_scripts": sorted(REQUIRED_LOCAL_SCRIPTS.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands.json"))
    parser.add_argument("--shell-plan", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands.sh"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = validate_plan(load_json(args.plan_json), args.repo_root, args.shell_plan)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
