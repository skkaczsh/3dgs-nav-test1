#!/usr/bin/env python3
"""Audit shell runners that upload Python scripts to a remote temporary path.

This catches a common offline failure mode: a shell script copies
`foo.py` to `/tmp/foo.py` and executes it remotely, but `foo.py` imports another
repo-local module that was not copied with it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


SCP_PY_RE = re.compile(r"scp\b[^\n]*?\$\{?([A-Z0-9_]+)\}?[^\n]*?:([^\"'\s;]+\.py)")
PYTHON_TMP_RE = re.compile(r"python3\s+(/tmp/[A-Za-z0-9_./-]+\.py)")
ASSIGN_RE = re.compile(r'^([A-Z0-9_]+)="\$\{[A-Z0-9_]+:-([^"}]+\.py)\}"')


def shell_assignments(path: Path) -> dict[str, str]:
    assignments = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ASSIGN_RE.match(line.strip())
        if match:
            assignments[match.group(1)] = match.group(2)
    return assignments


def shell_uploaded_python(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    assignments = shell_assignments(path)
    uploads = {}
    for match in SCP_PY_RE.finditer(text):
        var_name, remote_path = match.groups()
        local = assignments.get(var_name, "")
        uploads[Path(remote_path).name] = {
            "var": var_name,
            "local": local,
            "remote": remote_path,
        }
    return uploads


def shell_executed_tmp_python(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {Path(match.group(1)).name for match in PYTHON_TMP_RE.finditer(text)}


def local_module_imports(path: Path, scripts_dir: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".", 1)[0]]
        else:
            continue
        for name in names:
            if (scripts_dir / f"{name}.py").exists():
                modules.add(f"{name}.py")
    return modules


def audit_runner(path: Path, scripts_dir: Path) -> dict:
    uploads = shell_uploaded_python(path)
    executed = shell_executed_tmp_python(path)
    uploaded_names = set(uploads)
    issues = []
    for name in sorted(executed & uploaded_names):
        local = uploads[name].get("local", "")
        local_path = Path(local)
        if not local_path.is_absolute():
            local_path = scripts_dir / name
        if not local_path.exists():
            issues.append({"script": name, "error": "local_upload_source_missing", "local": str(local_path)})
            continue
        for dependency in sorted(local_module_imports(local_path, scripts_dir)):
            if dependency == name:
                continue
            if dependency not in uploaded_names:
                issues.append(
                    {
                        "script": name,
                        "dependency": dependency,
                        "error": "missing_uploaded_local_dependency",
                    }
                )
    return {
        "runner": str(path),
        "uploaded_python": sorted(uploaded_names),
        "executed_tmp_python": sorted(executed),
        "issues": issues,
    }


def audit(scripts_dir: Path) -> dict:
    runners = []
    for path in sorted(scripts_dir.glob("*.sh")):
        result = audit_runner(path, scripts_dir)
        if result["uploaded_python"] or result["executed_tmp_python"]:
            runners.append(result)
    issues = [issue | {"runner": row["runner"]} for row in runners for issue in row["issues"]]
    return {
        "scripts_dir": str(scripts_dir),
        "runner_count": len(runners),
        "issue_count": len(issues),
        "issues": issues,
        "runners": runners,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = audit(args.scripts_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
