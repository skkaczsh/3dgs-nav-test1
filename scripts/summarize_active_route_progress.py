#!/usr/bin/env python3
"""Summarize active route progress and side-track readiness.

This script is read-only. It consolidates local route reports and lightweight
remote checks so the current execution boundary is explicit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout}s",
            "passed": False,
        }
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
    }


def ssh_json(host: str, port: int, script: str, timeout: int = 20) -> dict[str, Any]:
    result = run(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "ConnectTimeout=8",
            "-p",
            str(port),
            f"root@{host}",
            f"python3 - <<'PY'\n{script}\nPY",
        ],
        timeout=timeout,
    )
    if not result["passed"]:
        return {"reachable": False, "stderr": result["stderr"][-2000:]}
    try:
        data = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"reachable": True, "parse_error": True, "stdout": result["stdout"][-2000:]}
    data["reachable"] = True
    return data


def remote_status(host: str, port: int) -> dict[str, Any]:
    return ssh_json(
        host,
        port,
        r'''
import json
import subprocess
from pathlib import Path

def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True, check=False)

gpu_proc = run([
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
])
gpus = []
for line in gpu_proc.stdout.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) == 5:
        used = int(parts[2])
        total = int(parts[3])
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_used_mib": used,
            "memory_total_mib": total,
            "memory_used_ratio": used / total if total else None,
            "utilization_gpu_percent": int(parts[4]),
        })

qwen = run(["curl", "-s", "--max-time", "5", "http://127.0.0.1:8001/v1/models"])
try:
    qwen_json = json.loads(qwen.stdout) if qwen.stdout.strip() else {}
except Exception:
    qwen_json = {"parse_error": True, "stdout": qwen.stdout[:1000]}

preflight = Path("/root/epfs/new_route_stage1_skymask/preflight_color_1000_1002")
color_rows = []
for ply in sorted(preflight.glob("frame_*.ply")):
    total = 0
    colored = 0
    with ply.open() as f:
        for line in f:
            if line.strip() == "end_header":
                break
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                total += 1
                if sum(int(v) for v in parts[3:6]) > 0:
                    colored += 1
    color_rows.append({
        "file": str(ply),
        "total_points": total,
        "colored_points": colored,
        "colored_ratio": colored / max(total, 1),
    })

print(json.dumps({
    "gpus": gpus,
    "qwen_localhost_8001": {
        "reachable": qwen.returncode == 0 and bool(qwen.stdout.strip()),
        "model_count": len(qwen_json.get("data") or qwen_json.get("models") or []),
        "raw_keys": sorted(qwen_json.keys()) if isinstance(qwen_json, dict) else [],
    },
    "color_preflight_1000_1002": color_rows,
}, indent=2))
''',
    )


def render_markdown(report: dict[str, Any]) -> str:
    next_inc = report.get("next_increment", {})
    route = report.get("route_decision", {}).get("main_route", {})
    visual = report.get("visual_acceptance", {})
    concept = report.get("conceptseg", {})
    old_route = report.get("old_route", {})
    lines = [
        "# Active Route Progress",
        "",
        f"- generated at: `{report.get('generated_at')}`",
        f"- visual gate: `{visual.get('status')}` / allow next increment `{visual.get('allow_next_increment')}`",
        f"- next increment status: `{next_inc.get('status')}`",
        f"- next increment ratios: `{next_inc.get('ratios')}`",
        "",
        "## Main Route",
        "",
        f"- decision: `{route.get('decision')}`",
        f"- semantic combo: `{route.get('semantic_combo')}`",
        f"- projection route: `{route.get('projection_route')}`",
        f"- target/object count: `{route.get('target_count')}` / `{route.get('object_count')}`",
        f"- residual unassigned surface points: `{route.get('residual_surface_unassigned_points')}`",
        "",
        "## Next Increment 1000-1999",
        "",
    ]
    for step in next_inc.get("next_steps", []):
        lines.append(f"- next step: {step}")
    for row in report.get("scan_train", {}).get("color_preflight_1000_1002", []):
        lines.append(
            f"- color preflight `{Path(row['file']).name}`: "
            f"`{row['colored_points']}/{row['total_points']}` "
            f"(`{row['colored_ratio']:.4f}`)"
        )
    lines.extend(
        [
            "",
            "## Side Tracks",
            "",
            f"- old route: passed `{old_route.get('passed')}`, colored ratio `{old_route.get('colored_ratio')}`, role `visual_color_reference_only`",
            f"- ConceptSeg: decision `{concept.get('decision')}`, accepted candidate count `{concept.get('summary', {}).get('accepted_candidate_count')}`, accepted target count `{concept.get('summary', {}).get('accepted_target_count')}`",
            "",
            "## Servers",
            "",
        ]
    )
    for name in ("scan_train", "scan_vlm"):
        server = report.get(name, {})
        lines.append(f"### {name}")
        lines.append(f"- reachable: `{server.get('reachable')}`")
        for gpu in server.get("gpus", []):
            lines.append(
                f"- gpu{gpu.get('index')} {gpu.get('name')}: "
                f"mem `{gpu.get('memory_used_mib')}/{gpu.get('memory_total_mib')}` MiB, "
                f"util `{gpu.get('utilization_gpu_percent')}`%"
            )
        qwen = server.get("qwen_localhost_8001", {})
        if qwen:
            lines.append(f"- qwen localhost:8001 reachable: `{qwen.get('reachable')}`, models `{qwen.get('model_count')}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-dir", type=Path, default=ROOT / "route_status_20260610")
    parser.add_argument("--output", type=Path, default=ROOT / "route_status_20260610/active_route_progress_20260611.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "route_status_20260610/active_route_progress_20260611.md")
    args = parser.parse_args()

    route_decision = read_json(args.status_dir / "dense_semantic_route_decision_20260611.json")
    visual_acceptance = read_json(args.status_dir / "visual_acceptance_review_20260611.json")
    next_increment = read_json(args.status_dir / "next_increment_readiness_1000_1999.json")
    conceptseg = read_json(args.status_dir / "conceptseg_integration_plan_20260611.json")
    old_route = read_json(ROOT / "server_old_route_smoke/world_colorize_summary.json")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_decision": route_decision,
        "visual_acceptance": visual_acceptance,
        "next_increment": next_increment,
        "conceptseg": conceptseg,
        "old_route": {
            "passed": True,
            "colored_ratio": old_route.get("colored_ratio"),
            "ply_vertex_count": old_route.get("ply_vertex_count"),
            "sample_mode": old_route.get("sample_mode"),
            "fusion_mode": old_route.get("fusion_mode"),
        },
        "scan_train": remote_status("10.0.8.114", 31909),
        "scan_vlm": remote_status("10.0.8.114", 31079),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown)}, indent=2))


if __name__ == "__main__":
    main()
