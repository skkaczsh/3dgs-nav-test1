#!/usr/bin/env python3
"""Run or print validated semantic viewer PLY export.

This launcher recolors an existing viewer PLY from fused object labels only
after object semantic evidence-fusion validation has passed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.current_mainline_contract import reject_forbidden_production_input


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def existing_file(path: Path, name: str) -> None:
    reject_forbidden_production_input(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} missing: {path}")
    if not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")


def validation_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "status": "missing", "errors": [f"missing_validation_report={path}"]}
    data = read_json(path)
    errors = [str(item) for item in data.get("errors", [])]
    return {
        "schema": data.get("schema"),
        "passed": data.get("schema") == "object-semantic-evidence-fusion-validation/v1" and data.get("passed") is True,
        "status": "pass" if data.get("passed") is True else "fail",
        "errors": errors,
        "path": str(path),
    }


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "scripts/rewrite_viewer_ply_semantics.py",
        "--source-ply",
        str(args.source_ply),
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--output-ply",
        str(args.output_ply),
    ]
    if args.report_json:
        command.extend(["--report-json", str(args.report_json)])
    return command


def build_plan(args: argparse.Namespace, validation: dict[str, Any]) -> dict[str, Any]:
    command = build_command(args)
    blocked = not validation["passed"] and not args.allow_unvalidated_export
    return {
        "schema": "validated-semantic-viewer-export-plan/v1",
        "status": "blocked" if blocked else "ready",
        "source_ply": str(args.source_ply),
        "objects_jsonl": str(args.objects_jsonl),
        "output_ply": str(args.output_ply),
        "report_json": str(args.report_json) if args.report_json else None,
        "validation": validation,
        "allow_unvalidated_export": bool(args.allow_unvalidated_export),
        "commands": [
            {
                "name": "rewrite_viewer_ply_semantics",
                "argv": command,
                "shell": shell_join(command),
            }
        ],
    }


def run_command(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--fusion-validation", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--python", default="python")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-unvalidated-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing_file(args.source_ply, "source ply")
    existing_file(args.objects_jsonl, "objects jsonl")
    existing_file(args.fusion_validation, "fusion validation")
    reject_forbidden_production_input(args.output_ply)
    if args.report_json:
        reject_forbidden_production_input(args.report_json)
    validation = validation_status(args.fusion_validation)
    plan = build_plan(args, validation)
    plan_path = args.plan_json or (args.output_ply.parent / "validated_semantic_viewer_export_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if plan["status"] == "blocked":
        return 2 if args.run else 0
    if args.run:
        run_command([str(part) for part in plan["commands"][0]["argv"]], Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
