#!/usr/bin/env python3
"""Monitor remote dense-semantic production jobs across scan-train and scan-vlm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")


def run_ssh_python(
    port: int,
    script: str,
    timeout: int = 20,
    connect_timeout: int = 6,
    bind_address: str = "",
) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
    ]
    if bind_address:
        cmd.extend(["-o", f"BindAddress={bind_address}"])
    cmd.extend([
        "-p",
        str(port),
        "root@10.0.8.114",
        "python3",
        "-",
    ])
    try:
        proc = subprocess.run(cmd, input=script, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"passed": False, "timeout": True, "stdout": exc.stdout or "", "stderr": exc.stderr or ""}
    return {
        "passed": proc.returncode == 0,
        "timeout": False,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


REMOTE_SCRIPT = r'''
import csv
import datetime
import json
import re
import socket
import subprocess
from pathlib import Path


def run(cmd):
    try:
        return subprocess.run(cmd, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, "", "")


out = Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_1000_1999")
sam = Path("/root/epfs/new_route_stage1_skymask/sam_masks_1000_1999_combined")
combos = [
    "sam2_qwen",
    "sam2_sky_label_merge_qwen_review",
    "sam2_prompt_v3_sky_label_merge",
    "sam2_prompt_v3_sky_label_merge_completion",
]

tmux = run(["tmux", "ls"])
processes = run([
    "pgrep",
    "-af",
    "pure_sam_mask_generator|run_server_semantic_completion_sharded|run_eval.py|review_merged_labels_prompt_v2.py|complete_unknown_regions.py|llama-server",
])
gpu_proc = run([
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader",
])
gpu = []
for row in csv.reader(gpu_proc.stdout.splitlines()):
    if not row:
        continue
    gpu.append({
        "index": int(row[0].strip()),
        "name": row[1].strip(),
        "memory_used_mib": int(row[2].split()[0]),
        "memory_total_mib": int(row[3].split()[0]),
        "utilization_gpu_percent": int(row[4].split()[0]),
    })

counts = {"sam_masks": len(list(sam.glob("*_sam_masks.json")))}
for combo in combos:
    counts[combo] = len(list((out / "images").glob(f"cam*_*/{combo}/semantic.png")))
counts["label_records"] = len(list((out / "images").glob("cam*_*/sam2_prompt_v3_sky_label_merge_completion/label_records.json")))


def parse_log_stats(log_roots):
    stats = {}
    for root in log_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.glob("*.log")):
            key = path.stem
            row = stats.setdefault(key, {"lines": 0, "parse_true": 0, "parse_false": 0, "last_nonempty": ""})
            try:
                for line in path.read_text(errors="replace").splitlines():
                    text = line.strip()
                    if not text:
                        continue
                    row["lines"] += 1
                    row["last_nonempty"] = text[-240:]
                    if re.search(r"\bparse=True\b", text):
                        row["parse_true"] += 1
                    if re.search(r"\bparse=False\b", text):
                        row["parse_false"] += 1
            except OSError as exc:
                row["last_nonempty"] = f"read_error: {exc}"
    return stats


log_stats = parse_log_stats(sorted(out.glob("_sharded_work*/logs")))

print(json.dumps({
    "hostname": socket.gethostname(),
    "date_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "tmux": tmux.stdout.splitlines() if tmux.returncode == 0 else [],
    "processes": processes.stdout.splitlines() if processes.returncode == 0 else [],
    "gpu": gpu,
    "counts": counts,
    "log_stats": log_stats,
}, ensure_ascii=False))
'''


def collect(bind_address: str = "") -> dict[str, Any]:
    servers = []
    for name, port in [("scan-train", 31909), ("scan-vlm", 31079)]:
        result = run_ssh_python(port, REMOTE_SCRIPT, bind_address=bind_address)
        row: dict[str, Any] = {
            "name": name,
            "port": port,
            "bind_address": bind_address,
            "reachable": result["passed"],
        }
        if result["passed"]:
            try:
                row.update(json.loads(result["stdout"]))
            except json.JSONDecodeError as exc:
                row["parse_error"] = str(exc)
                row["stdout"] = result["stdout"][-2000:]
        else:
            row["stderr"] = result.get("stderr", "")[-2000:]
            row["stdout"] = result.get("stdout", "")[-2000:]
        servers.append(row)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bind_address": bind_address,
        "range": {"start": 1000, "end": 1999, "images": 3000},
        "servers": servers,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Remote Production Monitor",
        "",
        f"- generated at: `{report.get('generated_at')}`",
        f"- bind address: `{report.get('bind_address') or ''}`",
        f"- range: `{report.get('range')}`",
        "",
    ]
    for server in report.get("servers", []):
        lines.extend([f"## {server.get('name')}", ""])
        lines.append(f"- reachable: `{server.get('reachable')}`")
        lines.append(f"- hostname: `{server.get('hostname')}`")
        lines.append(f"- date utc: `{server.get('date_utc')}`")
        counts = server.get("counts", {})
        if counts:
            lines.append("- counts: `" + json.dumps(counts, ensure_ascii=False) + "`")
        log_stats = server.get("log_stats") or {}
        if log_stats:
            sam2_true = sum(v.get("parse_true", 0) for k, v in log_stats.items() if k.startswith("sam2_qwen"))
            sam2_false = sum(v.get("parse_false", 0) for k, v in log_stats.items() if k.startswith("sam2_qwen"))
            lines.append(f"- sam2_qwen parse stats: `true={sam2_true}, false={sam2_false}`")
            active_logs = {
                k: v.get("last_nonempty", "")
                for k, v in sorted(log_stats.items())
                if v.get("last_nonempty") and (k.startswith("review") or k.startswith("completion") or k.startswith("sam2_qwen"))
            }
            for key, value in list(active_logs.items())[:12]:
                lines.append(f"- log {key}: `{value}`")
        gpu = server.get("gpu", [])
        for row in gpu:
            lines.append(
                f"- gpu{row.get('index')} {row.get('name')}: mem `{row.get('memory_used_mib')}/{row.get('memory_total_mib')}` MiB, util `{row.get('utilization_gpu_percent')}`%"
            )
        sessions = server.get("tmux", [])
        if sessions:
            lines.append("- tmux sessions:")
            lines.extend(f"  - `{x}`" for x in sessions)
        processes = server.get("processes", [])
        if processes:
            lines.append(f"- matching processes: `{len(processes)}`")
        if server.get("parse_error"):
            lines.append(f"- parse error: `{server.get('parse_error')}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "route_status_20260610/remote_production_monitor_20260611.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "route_status_20260610/remote_production_monitor_20260611.md")
    parser.add_argument(
        "--bind-address",
        default=os.environ.get("BIND_ADDRESS", ""),
        help="Optional local source address for SSH, for example the Wi-Fi IP when Ethernet has the default route.",
    )
    args = parser.parse_args()

    report = collect(bind_address=args.bind_address)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
