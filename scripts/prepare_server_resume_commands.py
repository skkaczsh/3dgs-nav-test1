#!/usr/bin/env python3
"""Generate an ordered server-resume command plan without probing servers."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


DEFAULT_READINESS = Path("/Users/skkac/Work/SCAN/route_status_20260610/server_resume_readiness.json")
DEFAULT_OUTPUT_JSON = Path("/Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands.json")
DEFAULT_OUTPUT_SH = Path("/Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands.sh")


def load_readiness(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path), "ready_for_server_probe": False, "blockers": ["missing_readiness_report"]}
    return json.loads(path.read_text(encoding="utf-8"))


def command(name: str, description: str, cmd: str, required: bool = True) -> dict:
    return {
        "name": name,
        "description": description,
        "command": cmd,
        "required": required,
    }


def build_plan(args: argparse.Namespace) -> dict:
    readiness = load_readiness(args.readiness)
    blockers = list(readiness.get("blockers", []))
    if not readiness.get("ready_for_server_probe"):
        blockers.append("readiness_not_ready_for_server_probe")

    bind_prefix = f"BIND_ADDRESS={shlex.quote(args.bind_address)} " if args.bind_address else ""
    qwen_cmd = (
        f"{bind_prefix}SERVER={shlex.quote(args.server)} CONCURRENCY={args.qwen_concurrency} "
        "bash scripts/resume_server_qwen_review.sh"
    )
    semantic_cmd = (
        f"PATCH_SCENE_PROMPTS=1 SHARDS={args.semantic_shards} "
        "bash scripts/run_server_semantic_completion_sharded.sh"
    )
    fusion_cmd = (
        f"MIN_MERGE_CONFIDENCE={args.min_merge_confidence} "
        "bash scripts/run_server_target_object_fusion.sh"
    )

    phases = [
        {
            "id": "connectivity",
            "title": "Connectivity Gate",
            "run_when": "after returning to the server LAN or VPN",
            "commands": [
                command(
                    "diagnose_connectivity",
                    "Verify SSH config, local bind address, and TCP reachability before any remote work.",
                    "python3 scripts/diagnose_server_connectivity.py --output /Users/skkac/Work/SCAN/server_connectivity_diagnosis_20260610_latest.json",
                ),
            ],
        },
        {
            "id": "main_qwen_review",
            "title": "Main Route Qwen Review",
            "run_when": "connectivity passed",
            "commands": [
                command(
                    "qwen_review",
                    "Run compact Qwen merge review at the empirically preferred concurrency.",
                    qwen_cmd,
                ),
            ],
        },
        {
            "id": "main_semantic_refresh",
            "title": "Scene-Aware 2D Semantic Refresh",
            "run_when": "Qwen review is complete, or when regenerating semantic artifacts is explicitly needed",
            "commands": [
                command(
                    "semantic_completion_sharded",
                    "Patch rooftop point-cloud semantic prompts and complete missing SAM2+Qwen semantic artifacts.",
                    semantic_cmd,
                ),
            ],
        },
        {
            "id": "main_object_fusion",
            "title": "Target/Object Fusion",
            "run_when": "scene-aware semantic artifacts are ready",
            "commands": [
                command(
                    "target_object_fusion",
                    "Rebuild targets and objects with VLM quality gating preserved.",
                    fusion_cmd,
                ),
            ],
        },
        {
            "id": "new_model_side_track",
            "title": "New Model Side Track",
            "run_when": "main route GPU demand is low",
            "commands": [
                command(
                    "conceptseg_status",
                    "Inspect ConceptSeg-R1 files and GPU state without promoting it to the main path.",
                    "ssh scan-train 'nvidia-smi; ls -lah /root/epfs/model_side_tracks/ConceptSeg-R1'",
                    required=False,
                ),
            ],
        },
        {
            "id": "old_route_side_track",
            "title": "Old Route Side Track",
            "run_when": "main route GPU demand is low and visual-reference comparison is needed",
            "commands": [
                command(
                    "old_route_smoke_status",
                    "Inspect local old-route smoke output; keep it visual-reference-only until it passes reviewed object QA gates.",
                    "ls -lah /Users/skkac/Work/SCAN/server_old_route_smoke",
                    required=False,
                ),
            ],
        },
    ]

    return {
        "readiness": {
            "path": str(args.readiness),
            "ready_for_server_probe": readiness.get("ready_for_server_probe"),
            "blockers": blockers,
            "offline_qa_git_head": readiness.get("offline_qa", {}).get("git_head"),
        },
        "defaults": {
            "server": args.server,
            "bind_address": args.bind_address,
            "qwen_concurrency": args.qwen_concurrency,
            "semantic_shards": args.semantic_shards,
            "min_merge_confidence": args.min_merge_confidence,
        },
        "phases": phases,
    }


def shell_quote_command(cmd: str) -> str:
    return cmd.replace("'", "'\"'\"'")


def render_shell(plan: dict) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated server resume plan. Run only after returning to the server LAN/VPN.",
        "",
    ]
    blockers = plan["readiness"].get("blockers", [])
    if blockers:
        lines.extend(
            [
                "echo 'Readiness blockers are present; inspect server_resume_commands.json first.' >&2",
                f"echo 'blockers: {', '.join(blockers)}' >&2",
                "exit 1",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    for phase in plan["phases"]:
        lines.append(f"echo '[phase] {phase['title']}'")
        for row in phase["commands"]:
            if row["required"]:
                lines.append(row["command"])
            else:
                lines.append(f"echo '[optional] {row['name']}: {shell_quote_command(row['description'])}'")
                lines.append(f"echo '  {shell_quote_command(row['command'])}'")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-shell", type=Path, default=DEFAULT_OUTPUT_SH)
    parser.add_argument("--server", default="scan-train")
    parser.add_argument("--bind-address", default="192.168.0.3")
    parser.add_argument("--qwen-concurrency", type=int, default=4)
    parser.add_argument("--semantic-shards", type=int, default=4)
    parser.add_argument("--min-merge-confidence", type=float, default=0.5)
    args = parser.parse_args()

    plan = build_plan(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_shell.write_text(render_shell(plan), encoding="utf-8")
    args.output_shell.chmod(0o755)
    print(json.dumps({"json": str(args.output_json), "shell": str(args.output_shell), "blockers": plan["readiness"]["blockers"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
