#!/usr/bin/env python3
"""Validate that approved mainline runners enforce their stage contracts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = REPO_ROOT / "docs" / "current_dense_patch_state.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def imports_name(tree: ast.AST, module_names: set[str], imported_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in module_names:
            if any(alias.name == imported_name for alias in node.names):
                return True
    return False


def calls_name(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
            return True
    return False


def has_function(tree: ast.AST, name: str) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == name for node in ast.walk(tree))


def has_argparse_flag(text: str, flag: str) -> bool:
    return flag in text


def validate_python_runner(path: Path, stage: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    errors: list[str] = []
    imports_guard = imports_name(
        tree,
        {"scripts.current_mainline_contract", "current_mainline_contract"},
        "reject_forbidden_production_input",
    )
    if not imports_guard:
        errors.append("missing_forbidden_input_guard_import")
    if not calls_name(tree, "reject_forbidden_production_input"):
        errors.append("missing_forbidden_input_guard_call")

    if stage == "object_building":
        if not has_function(tree, "run_mainline_healthcheck") or not calls_name(tree, "run_mainline_healthcheck"):
            errors.append("missing_mainline_healthcheck_call")
        if not has_function(tree, "validate_current_dense_inputs") or not calls_name(tree, "validate_current_dense_inputs"):
            errors.append("missing_current_dense_input_allowlist_call")
        if "validate_production_inputs" not in text:
            errors.append("missing_validate_production_inputs_reference")
    elif stage == "semantic_evidence":
        if "patch_gate_status" not in text:
            errors.append("missing_patch_promotion_gate")
        if not has_function(tree, "run_mainline_healthcheck") or not calls_name(tree, "run_mainline_healthcheck"):
            errors.append("missing_mainline_healthcheck_call")
        if not has_argparse_flag(text, "--allow-unpromoted-patch-experiment"):
            errors.append("missing_explicit_unpromoted_experiment_flag")
    elif stage == "qa_viewer_export":
        if "validation_status" not in text:
            errors.append("missing_fusion_validation_gate")
        if not has_argparse_flag(text, "--allow-unvalidated-export"):
            errors.append("missing_explicit_unvalidated_export_flag")
        if "rewrite_viewer_ply_semantics.py" not in text:
            errors.append("missing_validated_rewrite_command")
    else:
        errors.append(f"unknown_runner_stage={stage}")
    return errors


def validate_shell_runner(path: Path, stage: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if stage != "object_building":
        errors.append(f"shell_runner_unexpected_stage={stage}")
    required_fragments = {
        "RUN_PREFLIGHT": "missing_run_preflight_switch",
        "validate_current_mainline.py": "missing_mainline_preflight",
        "validate_production_inputs.py": "missing_production_input_preflight",
        "scripts/geometry_input_contract.py": "missing_geometry_contract_rsync",
        "--require-current-dense": "missing_current_dense_allowlist_preflight",
        "rsync -az docs/current_dense_patch_state.json": "missing_dense_state_rsync",
        "tmux new-session": "missing_tmux_launch",
    }
    for fragment, error in required_fragments.items():
        if fragment not in text:
            errors.append(error)
    return errors


def validate_runner(row: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    rel = str(row.get("path", ""))
    stage = str(row.get("stage", ""))
    path = repo_root / rel
    errors: list[str] = []
    if not rel:
        errors.append("missing_runner_path")
    if not stage:
        errors.append("missing_runner_stage")
    if not path.exists():
        errors.append(f"missing_runner_file={rel}")
    elif path.suffix == ".py":
        errors.extend(validate_python_runner(path, stage))
    elif path.suffix == ".sh":
        errors.extend(validate_shell_runner(path, stage))
    else:
        errors.append(f"unsupported_runner_extension={path.suffix}")
    return {
        "path": rel,
        "stage": stage,
        "passed": not errors,
        "errors": errors,
    }


def validate(state_path: Path = DEFAULT_STATE) -> dict[str, Any]:
    data = load_json(state_path)
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    rows = [row for row in data.get("approved_runners", []) if isinstance(row, dict)]
    for row in rows:
        report = validate_runner(row)
        reports.append(report)
        errors.extend(f"{report['path']}:{error}" for error in report["errors"])
    if not rows:
        errors.append("missing_approved_runners")
    return {
        "schema": "approved-mainline-runner-usage/v1",
        "passed": not errors,
        "state": str(state_path),
        "approved_runner_count": len(rows),
        "reports": reports,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()
    report = validate(args.state)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for error in report["errors"]:
            print(error)
        if report["passed"]:
            print(f"approved mainline runner usage ok: {report['approved_runner_count']} runners")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
