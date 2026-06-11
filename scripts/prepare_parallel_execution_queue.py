#!/usr/bin/env python3
"""Build the next execution queue from current readiness reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def server_by_name(infra: dict[str, Any], name: str) -> dict[str, Any]:
    for server in infra.get("servers", []):
        if server.get("name") == name:
            return server
    return {}


def endpoint_env(server: dict[str, Any]) -> str:
    endpoint = server.get("endpoint", {})
    host = endpoint.get("host", "")
    port = endpoint.get("port", "")
    if not host or not port:
        return ""
    return f"SSH_HOST={host} SSH_PORT={port} SSH_USER=root"


def gpu_summary(server: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": gpu.get("index"),
            "name": gpu.get("name"),
            "memory_used_mib": gpu.get("memory_used_mib"),
            "memory_total_mib": gpu.get("memory_total_mib"),
            "memory_used_ratio": gpu.get("memory_used_ratio"),
            "utilization_gpu_percent": gpu.get("utilization_gpu_percent"),
        }
        for gpu in server.get("gpus", [])
    ]


def make_queue(args: argparse.Namespace) -> dict[str, Any]:
    infra = read_json(args.infra_readiness)
    release = read_json(args.release_status)
    acceptance = read_json(args.delivery_acceptance)
    route_decision = read_json(args.route_decision)
    visual_acceptance = read_json(args.visual_acceptance)
    train = server_by_name(infra, "scan-train")
    vlm = server_by_name(infra, "scan-vlm")

    visual_accepted = bool(visual_acceptance.get("allow_next_increment"))
    visual_gate_open = release.get("release", {}).get("status") == "ready_for_visual_review" and not visual_accepted
    package_ok = bool(acceptance.get("passed"))
    infra_ok = bool(infra.get("passed"))
    main_authoritative = route_decision.get("main_route", {}).get("decision") == "continue_as_authoritative_route"

    queue: list[dict[str, Any]] = []
    if infra_ok and package_ok and main_authoritative:
        queue.append(
            {
                "id": "visual_review_current_0000_0999",
                "track": "main",
                "server": "local",
                "priority": 0,
                "status": "ready",
                "reason": "Current 0-999 package is built; manual visual acceptance is the next gate.",
                "commands": [
                    "cd /Users/skkac/Work/SCAN/new_route",
                    "python3 scripts/serve_review_package.py --root /Users/skkac/Work/SCAN --host 127.0.0.1 --port 8765",
                ],
                "review_urls": [
                    "http://127.0.0.1:8765/dataset_delivery_0000_0999/qa_index.html",
                    "http://127.0.0.1:8765/new_route/tools/semantic_ply_viewer.html",
                ],
                "acceptance_record": str(args.visual_acceptance),
                "gate": "visual_acceptance_in_ply_viewer_or_cloudcompare",
            }
        )
    if infra_ok and train.get("reachable"):
        queue.append(
            {
                "id": "main_route_next_increment_plan",
                "track": "main",
                "server": "scan-train",
                "priority": 1,
                "status": "blocked_by_visual_gate" if visual_gate_open else "ready_after_gate",
                "reason": "Do not extend beyond 0-999 until current package is visually accepted.",
                "gpu_summary": gpu_summary(train),
                "commands_after_gate": [
                    "cd /Users/skkac/Work/SCAN/new_route",
                    (
                        f"{endpoint_env(train)} SERVER=scan-train START_FRAME=1000 END_FRAME=1999 "
                        "bash scripts/run_remote_server_target_object_fusion.sh"
                    ),
                ],
            }
        )
    if infra_ok and train.get("reachable"):
        queue.append(
            {
                "id": "conceptseg_review_only_expansion",
                "track": "new_model_side_track",
                "server": "scan-train",
                "priority": 2,
                "status": "ready_when_gpu1_idle",
                "reason": "ConceptSeg is useful only as conservative review/split proposals; keep it off the dense path.",
                "gpu_summary": gpu_summary(train),
                "commands": [
                    (
                        "ssh -F /dev/null -p 31909 root@10.0.8.114 "
                        "'tmux new-session -Ad -s conceptseg_side_track'"
                    ),
                    (
                        "ssh -F /dev/null -p 31909 root@10.0.8.114 "
                        "'tmux send-keys -t conceptseg_side_track "
                        "\"cd /root/epfs/new_route_scripts && CUDA_VISIBLE_DEVICES=1 LIMIT=-1 "
                        "OUTPUT_DIR=/root/epfs/new_route_stage1_skymask/conceptseg_fine_object_runlist_v008_outputs_next "
                        "bash ./run_server_conceptseg_fine_object_runlist.sh\" C-m'"
                    ),
                ],
                "constraints": [
                    "Use EPFS cache only.",
                    "Do not promote ConceptSeg to dense semantic generation.",
                    "Stop if GPU1 is needed for main-route work.",
                ],
            }
        )
    if infra_ok and vlm.get("reachable"):
        queue.append(
            {
                "id": "qwen_review_capacity_note",
                "track": "vlm_side_track",
                "server": "scan-vlm",
                "priority": 3,
                "status": "capacity_available",
                "reason": "L20 is reachable and mostly idle; Qwen concurrency should default to 4 when review work resumes.",
                "gpu_summary": gpu_summary(vlm),
                "constraints": [
                    "Avoid root-backed Hugging Face/cache writes.",
                    "Use /root/epfs for all model/cache/output paths.",
                ],
            }
        )
    queue.append(
        {
            "id": "old_route_reference_rebuild",
            "track": "old_route_side_track",
            "server": "scan-train",
            "priority": 4,
            "status": "deferred_until_runner_rebuilt",
            "reason": "Old route is validated only as color reference; no reusable production runner is currently available.",
            "commands": [
                "cd /Users/skkac/Work/SCAN/new_route",
                "python3 scripts/validate_old_route_reference.py",
            ],
        }
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "infra_readiness": str(args.infra_readiness),
            "release_status": str(args.release_status),
            "delivery_acceptance": str(args.delivery_acceptance),
            "route_decision": str(args.route_decision),
            "visual_acceptance": str(args.visual_acceptance),
        },
        "gates": {
            "infra_passed": infra_ok,
            "delivery_acceptance_passed": package_ok,
            "main_route_authoritative": main_authoritative,
            "visual_acceptance_status": visual_acceptance.get("status", "missing"),
            "visual_acceptance_all_required_accepted": visual_accepted,
            "visual_gate_open": visual_gate_open,
        },
        "queue": queue,
    }


def render_markdown(queue: dict[str, Any]) -> str:
    lines = [
        "# Parallel Execution Queue",
        "",
        f"- generated at: `{queue['generated_at']}`",
        f"- gates: `{queue['gates']}`",
        "",
        "## Tasks",
        "",
    ]
    for item in queue["queue"]:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- track: `{item['track']}`",
                f"- server: `{item['server']}`",
                f"- priority: `{item['priority']}`",
                f"- status: `{item['status']}`",
                f"- reason: {item['reason']}",
            ]
        )
        commands = item.get("commands") or item.get("commands_after_gate") or []
        if commands:
            lines.extend(["- commands:", ""])
            lines.append("```bash")
            lines.extend(commands)
            lines.append("```")
        if item.get("constraints"):
            lines.append("- constraints:")
            lines.extend(f"  - {constraint}" for constraint in item["constraints"])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infra-readiness", type=Path, default=ROOT / "route_status_20260610/infra_readiness_20260611.json")
    parser.add_argument("--release-status", type=Path, default=ROOT / "route_status_20260610/dense_semantic_release_status_20260611.json")
    parser.add_argument("--delivery-acceptance", type=Path, default=ROOT / "route_status_20260610/delivery_acceptance_20260611.json")
    parser.add_argument("--route-decision", type=Path, default=ROOT / "route_status_20260610/dense_semantic_route_decision_20260611.json")
    parser.add_argument("--visual-acceptance", type=Path, default=ROOT / "route_status_20260610/visual_acceptance_review_20260611.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "route_status_20260610")
    args = parser.parse_args()

    queue = make_queue(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "parallel_execution_queue_20260611.json"
    md_path = args.output_dir / "parallel_execution_queue_20260611.md"
    json_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(queue), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "tasks": len(queue["queue"])}, indent=2))


if __name__ == "__main__":
    main()
