#!/usr/bin/env python3
"""Validate that production scripts use the shared forbidden-input guard."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from scripts.current_mainline_contract import PROTECTED_PRODUCTION_GUARD_SCRIPT_PATHS


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_MODULES: frozenset[str] = frozenset(
    {
        "scripts.current_mainline_contract",
        "current_mainline_contract",
    }
)


def imports_shared_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in CONTRACT_MODULES:
            continue
        if any(alias.name == "reject_forbidden_production_input" for alias in node.names):
            return True
    return False


def imports_direct_matcher(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in CONTRACT_MODULES:
            continue
        if any(alias.name == "forbidden_production_input_match" for alias in node.names):
            return True
    return False


def defines_local_legacy_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reject_forbidden_path":
            return True
    return False


def calls_shared_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "reject_forbidden_production_input":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "reject_forbidden_production_input":
            return True
    return False


def validate_script(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    errors: list[str] = []
    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path)

    if not imports_shared_guard(tree):
        errors.append("missing_reject_forbidden_production_input_import")
    if not calls_shared_guard(tree):
        errors.append("missing_reject_forbidden_production_input_call")
    if imports_direct_matcher(tree):
        errors.append("direct_forbidden_production_input_match_import")
    if defines_local_legacy_guard(tree):
        errors.append("local_reject_forbidden_path_definition")

    return {
        "path": display_path,
        "passed": not errors,
        "errors": errors,
    }


def validate(paths: tuple[str, ...] = PROTECTED_PRODUCTION_GUARD_SCRIPT_PATHS) -> dict[str, Any]:
    reports = []
    errors: list[str] = []
    for rel in paths:
        path = REPO_ROOT / rel
        if not path.exists():
            reports.append({"path": rel, "passed": False, "errors": ["missing_script"]})
            errors.append(f"{rel}:missing_script")
            continue
        report = validate_script(path)
        reports.append(report)
        errors.extend(f"{report['path']}:{error}" for error in report["errors"])
    return {
        "schema": "production-input-guard-usage/v1",
        "passed": not errors,
        "protected_script_count": len(paths),
        "reports": reports,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()
    report = validate()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for error in report["errors"]:
            print(error)
        if report["passed"]:
            print(f"production input guard usage ok: {report['protected_script_count']} scripts")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
