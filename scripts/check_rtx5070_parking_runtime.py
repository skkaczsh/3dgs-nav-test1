#!/usr/bin/env python3
"""Check the RTX 5070Ti parking semantic workspace before running jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_REMOTE_FILES = (
    "frame_targets_guarded_v2_full_s10_geometry_ceiling_rtx5070/frame_targets_refined.jsonl",
    "frame_targets_guarded_v2_full_s10_geometry_ceiling_rtx5070/frame_targets_refined.ply",
    "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_object_points_stride10.ply",
    "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_objects_viewer.jsonl",
    "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_object_viewer_export_report.json",
    "frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_local_object_qa_report.json",
    "guarded_v2_surface_refinement_all_risk_compare/qa_compare.json",
)


def run_local(argv: list[str], timeout: int) -> tuple[int, str, str]:
    proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def run_remote(host: str, script: str, timeout: int) -> tuple[int, str, str]:
    return run_local(["ssh", host, "bash", "-lc", script], timeout=timeout)


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, memory_used, memory_total, util = parts
        try:
            used = int(memory_used)
            total = int(memory_total)
            utilization = int(util)
        except ValueError:
            continue
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "memory_free_mib": total - used,
                "memory_used_ratio": used / total if total else None,
                "utilization_gpu_percent": utilization,
            }
        )
    return gpus


def parse_key_value_lines(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def bool_from_shell(value: str | None) -> bool:
    return value == "1" or value == "true"


def build_status_script(args: argparse.Namespace) -> str:
    required_files = " ".join(shlex.quote(path) for path in args.required_remote_file)
    return f"""
set -euo pipefail
REPO={shlex.quote(args.remote_repo)}
WORK={shlex.quote(args.remote_work)}
VENV={shlex.quote(args.venv)}
TMUX_SESSION={shlex.quote(args.tmux_session)}
echo section=system
hostname | sed 's/^/hostname=/'
date -Is | sed 's/^/remote_time=/'
echo section=gpu
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo section=workspace
printf 'repo_exists=%s\\n' "$([ -d "$REPO/.git" ] && echo 1 || echo 0)"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" rev-parse --short HEAD | sed 's/^/git_head=/'
  git -C "$REPO" status --porcelain | wc -l | tr -d ' ' | sed 's/^/git_dirty_count=/'
fi
printf 'work_exists=%s\\n' "$([ -d "$WORK" ] && echo 1 || echo 0)"
du -sh "$WORK" 2>/dev/null | awk '{{print "work_du=" $1}}' || true
echo section=runtime
printf 'venv_python_exists=%s\\n' "$([ -x "$VENV/bin/python" ] && echo 1 || echo 0)"
if [ -x "$VENV/bin/python" ]; then
  "$VENV/bin/python" - <<'PY'
import importlib.util
mods = ["torch", "cv2", "scipy", "sklearn", "transformers"]
for mod in mods:
    print(f"module_{{mod}}={{1 if importlib.util.find_spec(mod) else 0}}")
try:
    import torch
    print(f"torch_cuda_available={{1 if torch.cuda.is_available() else 0}}")
    print(f"torch_cuda_version={{torch.version.cuda}}")
except Exception as exc:
    print(f"torch_error={{type(exc).__name__}}:{{exc}}")
PY
fi
echo section=tmux
tmux has-session -t "$TMUX_SESSION" 2>/dev/null && echo tmux_session_exists=1 || echo tmux_session_exists=0
tmux ls 2>/dev/null | sed 's/^/tmux_ls=/' || true
echo section=proxy
(ps -eo pid=,comm=,args= | awk '$2 ~ /^(clash|mihomo)$/ {{print "proxy_process=" $0}}' || true) | head -5
(ss -ltn 2>/dev/null || true) | awk '$4 ~ /:7897$/ {{print "proxy_port_7897=1"; found=1}} END {{if (!found) print "proxy_port_7897=0"}}'
echo section=artifacts
for rel in {required_files}; do
  path="$WORK/$rel"
  if [ -s "$path" ]; then
    stat -c 'artifact=%n|%s|ok' "$path"
  else
    printf 'artifact=%s|0|missing\\n' "$path"
  fi
done
"""


def parse_remote_sections(text: str) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    current = "unknown"
    section_lines: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("section="):
            current = line.split("=", 1)[1]
            section_lines.setdefault(current, [])
            continue
        section_lines.setdefault(current, []).append(line)

    sections["system"] = parse_key_value_lines("\n".join(section_lines.get("system", [])))
    sections["workspace"] = parse_key_value_lines("\n".join(section_lines.get("workspace", [])))
    sections["runtime"] = parse_key_value_lines("\n".join(section_lines.get("runtime", [])))
    sections["tmux"] = parse_key_value_lines("\n".join(section_lines.get("tmux", [])))
    sections["proxy"] = parse_key_value_lines("\n".join(section_lines.get("proxy", [])))
    sections["gpu"] = parse_gpu_csv("\n".join(section_lines.get("gpu", [])))
    artifacts = []
    for line in section_lines.get("artifacts", []):
        if not line.startswith("artifact="):
            continue
        payload = line.split("=", 1)[1]
        path, size, status = payload.rsplit("|", 2)
        artifacts.append({"path": path, "bytes": int(size), "status": status})
    sections["artifacts"] = artifacts
    return sections


def evaluate(sections: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    workspace = sections.get("workspace", {})
    runtime = sections.get("runtime", {})
    tmux = sections.get("tmux", {})
    proxy = sections.get("proxy", {})
    gpus = sections.get("gpu", [])
    artifacts = sections.get("artifacts", [])

    if not gpus:
        errors.append("nvidia-smi did not return a GPU")
    else:
        gpu = gpus[0]
        if gpu["memory_free_mib"] < args.min_free_vram_mib:
            errors.append(f"GPU free VRAM below threshold: {gpu['memory_free_mib']}MiB < {args.min_free_vram_mib}MiB")

    if not bool_from_shell(workspace.get("repo_exists")):
        errors.append("remote repo is missing")
    if not bool_from_shell(workspace.get("work_exists")):
        errors.append("remote workdir is missing")
    if as_int(workspace.get("git_dirty_count"), 0) > args.max_git_dirty_count:
        warnings.append(f"remote git dirty count is {workspace.get('git_dirty_count')}")

    if not bool_from_shell(runtime.get("venv_python_exists")):
        errors.append("remote venv python is missing")
    for mod in ("torch", "cv2", "scipy", "sklearn", "transformers"):
        if not bool_from_shell(runtime.get(f"module_{mod}")):
            errors.append(f"remote python module missing: {mod}")
    if not bool_from_shell(runtime.get("torch_cuda_available")):
        errors.append("torch cuda is not available in remote venv")

    if args.require_tmux and not bool_from_shell(tmux.get("tmux_session_exists")):
        errors.append(f"required tmux session is missing: {args.tmux_session}")
    if args.require_proxy and not (proxy.get("proxy_process") or bool_from_shell(proxy.get("proxy_port_7897"))):
        warnings.append("remote clash/mihomo proxy was not detected")

    missing_artifacts = [row["path"] for row in artifacts if row["status"] != "ok" or row["bytes"] <= 0]
    if missing_artifacts:
        errors.append(f"missing required remote artifacts: {missing_artifacts}")

    return errors, warnings


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="scan-rtx5070")
    parser.add_argument("--remote-repo", default="/home/zsh/Work/SCAN/new_route")
    parser.add_argument("--remote-work", default="/home/zsh/Work/SCAN/work_MT20260616-175807")
    parser.add_argument("--venv", default="/home/zsh/Work/SCAN/.venvs/scan-semantic")
    parser.add_argument("--tmux-session", default="scan_migrate")
    parser.add_argument("--required-remote-file", action="append", default=list(DEFAULT_REQUIRED_REMOTE_FILES))
    parser.add_argument("--min-free-vram-mib", type=int, default=8000)
    parser.add_argument("--max-git-dirty-count", type=int, default=1)
    parser.add_argument("--require-tmux", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-proxy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    code, stdout, stderr = run_remote(args.host, build_status_script(args), timeout=args.timeout)
    if code != 0:
        result = {
            "passed": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": args.host,
            "errors": [f"ssh command failed with exit code {code}"],
            "warnings": [],
            "stderr": stderr,
            "stdout": stdout,
        }
    else:
        sections = parse_remote_sections(stdout)
        errors, warnings = evaluate(sections, args)
        result = {
            "passed": not errors,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": args.host,
            "errors": errors,
            "warnings": warnings,
            "sections": sections,
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
