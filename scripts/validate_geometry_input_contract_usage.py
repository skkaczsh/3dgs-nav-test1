#!/usr/bin/env python3
"""Validate that semantic stages respect geometry-only object inputs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from scripts.current_mainline_contract import PROTECTED_GEOMETRY_INPUT_CONTRACT_SCRIPT_PATHS


REPO_ROOT = Path(__file__).resolve().parents[1]


def imports_geometry_contract(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "scripts.geometry_input_contract",
            "geometry_input_contract",
        }:
            return True
    return False


def function_named(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def calls_name(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
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

    if not imports_geometry_contract(tree):
        errors.append("missing_geometry_input_contract_import")

    normalizer = function_named(tree, "normalized_original_label")
    if normalizer is None:
        errors.append("missing_normalized_original_label")
    elif not calls_name(normalizer, "is_geometry_only_row"):
        errors.append("normalized_original_label_missing_geometry_only_guard")

    return {
        "path": display_path,
        "passed": not errors,
        "errors": errors,
    }


def validate(paths: tuple[str, ...] = PROTECTED_GEOMETRY_INPUT_CONTRACT_SCRIPT_PATHS) -> dict[str, Any]:
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
        "schema": "geometry-input-contract-usage/v1",
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
            print(f"geometry input contract usage ok: {report['protected_script_count']} scripts")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
