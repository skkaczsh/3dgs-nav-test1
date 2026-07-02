#!/usr/bin/env python3
"""Run or print the object semantic evidence-fusion command.

This launcher is deliberately conservative: object-level semantic fusion is
allowed only after the patch experiment promotion gate passes, unless the
operator explicitly asks for an experimental dry-run plan.  It preserves object
ownership and delegates all label decisions to fuse_object_semantic_evidence.py.
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

from scripts.current_mainline_contract import forbidden_production_input_match


DEFAULT_MAINLINE_HEALTHCHECK = REPO_ROOT / "scripts" / "validate_current_mainline.py"
DEFAULT_PATCH_GATE = REPO_ROOT / "docs" / "patch_experiment_promotion_gate.json"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def reject_forbidden_path(path: Path) -> None:
    forbidden = forbidden_production_input_match(path)
    if forbidden:
        raise ValueError(f"forbidden input path contains {forbidden}: {path}")


def existing_file(path: Path, name: str) -> None:
    reject_forbidden_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} missing: {path}")
    if not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")


def patch_gate_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "passed": False, "reasons": [f"missing_patch_gate={path}"]}
    data = read_json(path)
    reasons = [str(item) for item in data.get("reasons", [])]
    return {
        "status": data.get("status"),
        "passed": data.get("schema") == "patch-experiment-promotion-gate/v1" and data.get("status") == "pass",
        "candidate": data.get("candidate"),
        "reasons": reasons,
        "path": str(path),
    }


def build_fuse_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "scripts/fuse_object_semantic_evidence.py",
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--output-jsonl",
        str(args.output_jsonl),
        "--report",
        str(args.report),
        "--sam-weight",
        str(args.sam_weight),
        "--teacher-weight",
        str(args.teacher_weight),
        "--scene-weight",
        str(args.scene_weight),
        "--min-total-weight",
        str(args.min_total_weight),
        "--min-winner-ratio",
        str(args.min_winner_ratio),
        "--min-scene-supported-ratio",
        str(args.min_scene_supported_ratio),
    ] + (["--allow-scene-only"] if args.allow_scene_only else [])


def build_validate_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "scripts/validate_object_semantic_evidence_fusion.py",
        "--input-objects",
        str(args.objects_jsonl),
        "--output-objects",
        str(args.output_jsonl),
        "--report",
        str(args.report),
        "--output-json",
        str(args.validation_report),
    ]
    if args.allow_scene_only:
        command.append("--allow-scene-only")
    return command


def build_plan(args: argparse.Namespace, gate: dict[str, Any]) -> dict[str, Any]:
    fuse_command = build_fuse_command(args)
    validate_command = build_validate_command(args)
    blocked = not gate["passed"] and not args.allow_unpromoted_patch_experiment
    return {
        "schema": "object-semantic-evidence-fusion-plan/v1",
        "status": "blocked" if blocked else "ready",
        "objects_jsonl": str(args.objects_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "report": str(args.report),
        "validation_report": str(args.validation_report),
        "patch_gate": gate,
        "allow_unpromoted_patch_experiment": bool(args.allow_unpromoted_patch_experiment),
        "commands": [
            {
                "name": "fuse_object_semantic_evidence",
                "argv": fuse_command,
                "shell": shell_join(fuse_command),
            },
            {
                "name": "validate_object_semantic_evidence_fusion",
                "argv": validate_command,
                "shell": shell_join(validate_command),
            }
        ],
    }


def run_command(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def run_mainline_healthcheck(args: argparse.Namespace) -> None:
    if args.skip_mainline_healthcheck:
        return
    script = args.mainline_healthcheck
    if not script.exists():
        raise FileNotFoundError(f"mainline healthcheck missing: {script}")
    subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--python", default="python")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--patch-gate", type=Path, default=DEFAULT_PATCH_GATE)
    parser.add_argument("--allow-unpromoted-patch-experiment", action="store_true")
    parser.add_argument("--mainline-healthcheck", type=Path, default=DEFAULT_MAINLINE_HEALTHCHECK)
    parser.add_argument("--skip-mainline-healthcheck", action="store_true")
    parser.add_argument("--sam-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=1.25)
    parser.add_argument("--scene-weight", type=float, default=0.35)
    parser.add_argument("--min-total-weight", type=float, default=3.0)
    parser.add_argument("--min-winner-ratio", type=float, default=0.58)
    parser.add_argument("--min-scene-supported-ratio", type=float, default=0.52)
    parser.add_argument("--allow-scene-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing_file(args.objects_jsonl, "objects jsonl")
    reject_forbidden_path(args.output_jsonl)
    reject_forbidden_path(args.report)
    if args.validation_report is None:
        args.validation_report = args.report.with_name(args.report.stem + "_validation.json")
    reject_forbidden_path(args.validation_report)
    gate = patch_gate_status(args.patch_gate)
    plan = build_plan(args, gate)
    plan_path = args.plan_json or (args.report.parent / "object_semantic_evidence_fusion_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if plan["status"] == "blocked":
        return 2 if args.run else 0
    if args.run:
        run_mainline_healthcheck(args)
        cwd = Path.cwd()
        for item in plan["commands"]:
            run_command([str(part) for part in item["argv"]], cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
