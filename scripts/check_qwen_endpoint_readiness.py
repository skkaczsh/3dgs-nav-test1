#!/usr/bin/env python3
"""Check Qwen OpenAI-compatible endpoints on the two scan servers."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")
MODEL = "Qwen3.6-35B-A3B-Q4_K_M-00001-of-00002.gguf"


SERVERS = (
    {"name": "scan-train", "host": "10.0.8.114", "port": 31909, "expected": "qwen_train_ready"},
    {"name": "scan-vlm", "host": "10.0.8.114", "port": 31079, "expected": "qwen_vlm_ready"},
)


def run(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout}s",
            "passed": False,
        }
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
    }


def remote_script(expected: str, max_tokens: int, timeout: int) -> str:
    return f'''
import json
import urllib.request

model = {MODEL!r}
expected = {expected!r}
payload = {{
    "model": model,
    "messages": [
        {{"role": "user", "content": "Return exactly one short final answer after thinking: " + expected}}
    ],
    "max_tokens": {max_tokens},
    "temperature": 0,
}}
result = {{
    "models_reachable": False,
    "chat_reachable": False,
    "expected": expected,
    "max_tokens": {max_tokens},
}}
try:
    with urllib.request.urlopen("http://127.0.0.1:8001/v1/models", timeout=10) as resp:
        models = json.load(resp)
    result["models_reachable"] = True
    result["model_count"] = len(models.get("data") or models.get("models") or [])
except Exception as exc:
    result["models_error"] = repr(exc)

try:
    req = urllib.request.Request(
        "http://127.0.0.1:8001/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={{"Content-Type": "application/json"}},
    )
    with urllib.request.urlopen(req, timeout={timeout}) as resp:
        data = json.load(resp)
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    result.update({{
        "chat_reachable": True,
        "content": content,
        "reasoning_len": len(reasoning),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "content_matches_expected": content.strip() == expected,
    }})
except Exception as exc:
    result["chat_error"] = repr(exc)

print(json.dumps(result, ensure_ascii=False, indent=2))
'''


def check_server(server: dict[str, Any], max_tokens: int, timeout: int) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "ConnectTimeout=8",
        "-p",
        str(server["port"]),
        f"root@{server['host']}",
        f"python3 - <<'PY'\n{remote_script(server['expected'], max_tokens, timeout)}\nPY",
    ]
    result = run(cmd, timeout=timeout + 20)
    row = {
        "name": server["name"],
        "endpoint": {
            "ssh": f"ssh -F /dev/null -p {server['port']} root@{server['host']}",
            "models": "http://127.0.0.1:8001/v1/models",
            "chat": "http://127.0.0.1:8001/v1/chat/completions",
        },
        "ssh_passed": result["passed"],
    }
    if result["passed"]:
        try:
            row.update(json.loads(result["stdout"]))
        except json.JSONDecodeError:
            row.update({"parse_error": True, "stdout_tail": result["stdout"][-2000:]})
    else:
        row.update({"stderr_tail": result["stderr"][-2000:]})
    row["passed"] = bool(row.get("models_reachable") and row.get("chat_reachable") and row.get("content_matches_expected"))
    return row


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen Endpoint Readiness",
        "",
        f"- generated at: `{report['generated_at']}`",
        f"- passed: `{report['passed']}`",
        f"- max tokens: `{report['max_tokens']}`",
        "",
    ]
    for row in report["servers"]:
        lines.extend(
            [
                f"## {row['name']}",
                "",
                f"- passed: `{row.get('passed')}`",
                f"- models reachable: `{row.get('models_reachable')}`",
                f"- chat reachable: `{row.get('chat_reachable')}`",
                f"- content matches expected: `{row.get('content_matches_expected')}`",
                f"- finish reason: `{row.get('finish_reason')}`",
                f"- reasoning length: `{row.get('reasoning_len')}`",
                f"- content: `{row.get('content')}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "route_status_20260610/qwen_endpoint_readiness_20260611.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "route_status_20260610/qwen_endpoint_readiness_20260611.md")
    args = parser.parse_args()

    servers = [check_server(server, args.max_tokens, args.timeout) for server in SERVERS]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_tokens": args.max_tokens,
        "passed": all(row.get("passed") for row in servers),
        "servers": servers,
        "interpretation": {
            "min_recommended_max_tokens": args.max_tokens,
            "reason": "Qwen3.6 endpoints emit reasoning_content before final content; low max_tokens can falsely look like empty content.",
            "recommended_concurrency": 4,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown), "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
