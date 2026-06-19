#!/usr/bin/env python3
"""Probe TensorRT readiness on the RTX 5070Ti host."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PY_MODULES = ("tensorrt", "torch_tensorrt", "onnx", "onnxsim", "polygraphy", "onnxruntime")


def run(argv: list[str], timeout: int, *, stdin: str | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(argv, input=stdin, text=True, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def remote_script(args: argparse.Namespace) -> str:
    module_list = ", ".join(repr(name) for name in PY_MODULES)
    return f"""
set -euo pipefail
VENV={shlex.quote(args.venv)}
echo section=system
hostname | sed 's/^/hostname=/'
date -Is | sed 's/^/remote_time=/'
echo section=gpu
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits || true
echo section=cuda
printf 'cuda_home_exists=%s\\n' "$([ -d /usr/local/cuda ] && echo 1 || echo 0)"
printf 'nvcc_path=%s\\n' "$(command -v nvcc || true)"
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -1 | sed 's/^/nvcc_version=/'; fi
echo section=trtexec
for candidate in "$(command -v trtexec || true)" /usr/src/tensorrt/bin/trtexec /usr/local/tensorrt/bin/trtexec /opt/tensorrt/bin/trtexec; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    printf 'trtexec_path=%s\\n' "$candidate"
    "$candidate" --version 2>&1 | head -5 | sed 's/^/trtexec_version=/'
    break
  fi
done
echo section=cpp
for header in /usr/include/NvInfer.h /usr/include/x86_64-linux-gnu/NvInfer.h /usr/local/tensorrt/include/NvInfer.h /opt/tensorrt/include/NvInfer.h; do
  if [ -f "$header" ]; then printf 'header=%s\\n' "$header"; fi
done
(ldconfig -p 2>/dev/null | grep -E 'libnvinfer|libnvonnxparser|libnvinfer_plugin' || true) | sed 's/^/lib=/'
echo section=python
printf 'venv_python_exists=%s\\n' "$([ -x "$VENV/bin/python" ] && echo 1 || echo 0)"
if [ -x "$VENV/bin/python" ]; then
  "$VENV/bin/python" - <<'PY'
import importlib.util
mods = [{module_list}]
for mod in mods:
    spec = importlib.util.find_spec(mod)
    print(f"module_{{mod}}={{1 if spec else 0}}")
    if spec and mod in ("tensorrt", "onnxruntime"):
        try:
            imported = __import__(mod)
            print(f"version_{{mod}}={{getattr(imported, '__version__', '')}}")
            if mod == "onnxruntime":
                print("onnxruntime_providers=" + ",".join(imported.get_available_providers()))
        except Exception as exc:
            print(f"error_{{mod}}={{type(exc).__name__}}:{{exc}}")
try:
    import torch
    print(f"torch_version={{torch.__version__}}")
    print(f"torch_cuda_available={{1 if torch.cuda.is_available() else 0}}")
    print(f"torch_cuda_version={{torch.version.cuda}}")
except Exception as exc:
    print(f"torch_error={{type(exc).__name__}}:{{exc}}")
PY
fi
"""


def parse_kv(lines: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    repeated: dict[str, list[str]] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in out:
            repeated.setdefault(key, [out.pop(key)]).append(value)
        elif key in repeated:
            repeated[key].append(value)
        else:
            out[key] = value
    out.update(repeated)
    return out


def parse_sections(text: str) -> dict[str, Any]:
    current = "unknown"
    raw: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("section="):
            current = line.split("=", 1)[1]
            raw.setdefault(current, [])
        else:
            raw.setdefault(current, []).append(line)
    sections = {name: parse_kv(lines) for name, lines in raw.items()}
    gpu_rows = []
    for line in raw.get("gpu", []):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            gpu_rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mib": parts[3],
                }
            )
    sections["gpu"] = gpu_rows
    return sections


def bool_kv(section: dict[str, Any], key: str) -> bool:
    return section.get(key) in ("1", "true", True)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def evaluate(sections: dict[str, Any]) -> dict[str, Any]:
    python = sections.get("python", {})
    cpp = sections.get("cpp", {})
    trtexec = sections.get("trtexec", {})
    cuda = sections.get("cuda", {})
    providers = str(python.get("onnxruntime_providers", ""))
    libs = as_list(cpp.get("lib"))
    headers = as_list(cpp.get("header"))

    torch_cuda = bool_kv(python, "torch_cuda_available")
    has_onnx = bool_kv(python, "module_onnx")
    has_polygraphy = bool_kv(python, "module_polygraphy")
    has_trt_py = bool_kv(python, "module_tensorrt")
    has_ort = bool_kv(python, "module_onnxruntime")
    has_trtexec = bool(trtexec.get("trtexec_path"))
    has_cpp_headers = bool(headers)
    has_cpp_libs = any("libnvinfer" in item for item in libs)
    has_ort_trt = "TensorrtExecutionProvider" in providers

    return {
        "torch_cuda_ready": torch_cuda,
        "onnx_export_ready": torch_cuda and has_onnx,
        "python_tensorrt_ready": has_trt_py,
        "cpp_tensorrt_ready": has_trtexec and has_cpp_headers and has_cpp_libs,
        "onnxruntime_tensorrt_ready": has_ort and has_ort_trt,
        "recommended_next_action": (
            "install TensorRT runtime/dev tools and Python ONNX helpers"
            if not (has_trtexec and has_trt_py and has_onnx)
            else "run tiny ONNX TensorRT smoke before SAM2 engine work"
        ),
        "missing": {
            "trtexec": not has_trtexec,
            "cpp_headers": not has_cpp_headers,
            "cpp_libs": not has_cpp_libs,
            "python_tensorrt": not has_trt_py,
            "python_onnx": not has_onnx,
            "python_polygraphy": not has_polygraphy,
            "python_onnxruntime": not has_ort,
            "onnxruntime_trt_provider": not has_ort_trt,
            "cuda_home": not bool_kv(cuda, "cuda_home_exists"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="scan-rtx5070")
    parser.add_argument("--venv", default="/home/zsh/Work/SCAN/.venvs/scan-semantic")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-cpp-tensorrt", action="store_true")
    parser.add_argument("--require-python-tensorrt", action="store_true")
    args = parser.parse_args()

    command = ["bash", "-s"] if args.host in ("local", "localhost", "127.0.0.1") else ["ssh", args.host, "bash", "-s"]
    code, stdout, stderr = run(command, timeout=args.timeout, stdin=remote_script(args))
    sections = parse_sections(stdout) if stdout else {}
    readiness = evaluate(sections) if code == 0 else {}
    errors = []
    if code != 0:
        errors.append(f"ssh probe failed with exit code {code}")
    if args.require_cpp_tensorrt and not readiness.get("cpp_tensorrt_ready"):
        errors.append("C++ TensorRT is not ready")
    if args.require_python_tensorrt and not readiness.get("python_tensorrt_ready"):
        errors.append("Python TensorRT is not ready")

    result = {
        "passed": not errors,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "errors": errors,
        "stderr": stderr,
        "readiness": readiness,
        "sections": sections,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
