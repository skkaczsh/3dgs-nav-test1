#!/usr/bin/env python3
"""Check server readiness for dense semantic dataset continuation.

This intentionally uses direct SSH endpoints instead of local SSH aliases. The
aliases have had stale BindAddress values after network changes, while the
direct 10.0.8.114 routes are the verified connection path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")


@dataclass(frozen=True)
class Server:
    name: str
    host: str
    port: int
    role: str
    required_paths: tuple[str, ...]
    optional_paths: tuple[str, ...] = ()


SERVERS = (
    Server(
        name="scan-train",
        host="10.0.8.114",
        port=31909,
        role="main_and_side_track_gpu",
        required_paths=(
            "/root/epfs/new_route_data",
            "/root/epfs/new_route_stage1_skymask",
            "/root/epfs/manifold_3dgs_project",
        ),
        optional_paths=("/root/epfs/model_side_tracks/ConceptSeg-R1",),
    ),
    Server(
        name="scan-vlm",
        host="10.0.8.114",
        port=31079,
        role="vlm_l20_side_track",
        required_paths=(
            "/root/epfs/new_route_data",
            "/root/epfs/new_route_stage1_skymask",
            "/root/epfs/manifold_3dgs_project",
        ),
    ),
)


LOCAL_ARTIFACTS = (
    ROOT / "dataset_delivery_0000_0999.tgz",
    ROOT / "dataset_delivery_0000_0999/package_manifest.json",
    ROOT / "route_status_20260610/dataset_delivery_manifest_0000_0999.json",
    ROOT / "route_status_20260610/dense_semantic_release_status_20260611.json",
    ROOT / "route_status_20260610/delivery_acceptance_20260611.json",
)


def run(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout}s",
            "timeout": True,
            "passed": False,
        }
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "timeout": False,
        "passed": proc.returncode == 0,
    }


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        index, name, mem_used, mem_total, util = parts[:5]
        try:
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_used_mib": int(mem_used),
                    "memory_total_mib": int(mem_total),
                    "memory_used_ratio": round(int(mem_used) / max(int(mem_total), 1), 4),
                    "utilization_gpu_percent": int(util),
                }
            )
        except ValueError:
            continue
    return rows


def parse_df(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 6:
            rows.append(
                {
                    "filesystem": parts[0],
                    "size": parts[1],
                    "used": parts[2],
                    "avail": parts[3],
                    "use_percent": parts[4],
                    "mounted_on": parts[5],
                }
            )
    return rows


def remote_script(server: Server) -> str:
    all_paths = list(server.required_paths + server.optional_paths)
    path_checks = "\n".join(
        f'if [ -e "{path}" ]; then echo "PATH_OK {path}"; else echo "PATH_MISSING {path}"; fi'
        for path in all_paths
    )
    return f"""
set -u
echo "HOSTNAME $(hostname)"
echo "DATE_UTC $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "NVIDIA_BEGIN"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo "NVIDIA_END"
echo "DF_BEGIN"
df -h / /root/epfs || true
echo "DF_END"
echo "PATHS_BEGIN"
{path_checks}
echo "PATHS_END"
"""


def section(text: str, begin: str, end: str) -> str:
    if begin not in text or end not in text:
        return ""
    return text.split(begin, 1)[1].split(end, 1)[0].strip()


def check_server(server: Server, timeout: int) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        f"ConnectTimeout={min(timeout, 10)}",
        "-o",
        "StrictHostKeyChecking=no",
        "-p",
        str(server.port),
        f"root@{server.host}",
        remote_script(server),
    ]
    result = run(cmd, timeout=timeout)
    stdout = result["stdout"]
    path_rows = []
    for line in section(stdout, "PATHS_BEGIN", "PATHS_END").splitlines():
        status, _, path = line.partition(" ")
        if status.startswith("PATH_") and path:
            path_rows.append({"path": path, "exists": status == "PATH_OK"})
    required_set = set(server.required_paths)
    required_ok = all(row["exists"] for row in path_rows if row["path"] in required_set)
    return {
        "name": server.name,
        "role": server.role,
        "endpoint": {"host": server.host, "port": server.port, "ssh": f"root@{server.host}:{server.port}"},
        "reachable": result["passed"],
        "hostname": stdout.split("HOSTNAME ", 1)[1].splitlines()[0] if "HOSTNAME " in stdout else "",
        "date_utc": stdout.split("DATE_UTC ", 1)[1].splitlines()[0] if "DATE_UTC " in stdout else "",
        "gpus": parse_nvidia_smi(section(stdout, "NVIDIA_BEGIN", "NVIDIA_END")),
        "disk": parse_df(section(stdout, "DF_BEGIN", "DF_END")),
        "paths": path_rows,
        "required_paths_ok": required_ok if result["passed"] else False,
        "ssh_result": {
            "returncode": result["returncode"],
            "stderr_tail": result["stderr"][-2000:],
            "timeout": result["timeout"],
        },
    }


def local_artifact_rows() -> list[dict[str, Any]]:
    rows = []
    for path in LOCAL_ARTIFACTS:
        rows.append({"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
    return rows


def build_recommendations(servers: list[dict[str, Any]]) -> list[str]:
    recommendations = [
        "Use direct SSH with -F /dev/null and the verified 10.0.8.114 ports until local SSH aliases are repaired.",
        "Keep dataset package review local; do not copy large PLY/JSONL artifacts into git.",
    ]
    by_name = {server["name"]: server for server in servers}
    train = by_name.get("scan-train", {})
    vlm = by_name.get("scan-vlm", {})
    if train.get("reachable") and train.get("required_paths_ok"):
        recommendations.append("scan-train is usable for controlled main-route and side-track GPU jobs.")
    if vlm.get("reachable"):
        recommendations.append("scan-vlm is reachable for L20 VLM side-track jobs, but keep caches and downloads on /root/epfs.")
    for server in servers:
        for disk in server.get("disk", []):
            if disk.get("mounted_on") == "/" and disk.get("use_percent", "0%").rstrip("%").isdigit():
                if int(disk["use_percent"].rstrip("%")) >= 85:
                    recommendations.append(f"{server['name']} root disk is tight at {disk['use_percent']}; avoid root-backed cache growth.")
            if disk.get("mounted_on") == "/root/epfs" and disk.get("use_percent", "0%").rstrip("%").isdigit():
                if int(disk["use_percent"].rstrip("%")) >= 90:
                    recommendations.append(f"{server['name']} /root/epfs is high at {disk['use_percent']}; clean transient datasets before large expansions.")
    return recommendations


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Infrastructure Readiness",
        "",
        f"- generated at: `{report['generated_at']}`",
        f"- passed: `{report['passed']}`",
        f"- all reachable: `{report['all_reachable']}`",
        f"- all required remote paths ok: `{report['all_required_paths_ok']}`",
        f"- local delivery artifacts ok: `{report['local_delivery_artifacts_ok']}`",
        "",
        "## Servers",
        "",
    ]
    for server in report["servers"]:
        gpu_text = ", ".join(
            f"{gpu['index']} {gpu['name']} {gpu['memory_used_mib']}/{gpu['memory_total_mib']}MiB util={gpu['utilization_gpu_percent']}%"
            for gpu in server.get("gpus", [])
        )
        disk_text = ", ".join(f"{row['mounted_on']} {row['used']}/{row['size']} {row['use_percent']}" for row in server.get("disk", []))
        lines.extend(
            [
                f"### {server['name']}",
                "",
                f"- role: `{server['role']}`",
                f"- endpoint: `ssh -F /dev/null -p {server['endpoint']['port']} root@{server['endpoint']['host']}`",
                f"- reachable: `{server['reachable']}`",
                f"- hostname: `{server.get('hostname')}`",
                f"- GPU: `{gpu_text}`",
                f"- disk: `{disk_text}`",
                f"- required paths ok: `{server['required_paths_ok']}`",
                "",
            ]
        )
    lines.extend(["## Recommendations", ""])
    lines.extend(f"- {item}" for item in report["recommendations"])
    lines.append("")
    return "\n".join(lines)


def build_report(timeout: int) -> dict[str, Any]:
    servers = [check_server(server, timeout) for server in SERVERS]
    local_artifacts = local_artifact_rows()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "servers": servers,
        "local_delivery_artifacts": local_artifacts,
        "all_reachable": all(server["reachable"] for server in servers),
        "all_required_paths_ok": all(server["required_paths_ok"] for server in servers),
        "local_delivery_artifacts_ok": all(row["exists"] and row["bytes"] > 0 for row in local_artifacts),
    }
    report["passed"] = (
        report["all_reachable"]
        and report["all_required_paths_ok"]
        and report["local_delivery_artifacts_ok"]
    )
    report["recommendations"] = build_recommendations(servers)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "route_status_20260610")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    report = build_report(args.timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "infra_readiness_20260611.json"
    md_path = args.output_dir / "infra_readiness_20260611.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
