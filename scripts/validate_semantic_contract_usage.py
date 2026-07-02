#!/usr/bin/env python3
"""Validate that current-mainline scripts reuse the shared semantic contract."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from scripts.current_mainline_contract import PROTECTED_SEMANTIC_CONTRACT_SCRIPT_PATHS


REPO_ROOT = Path(__file__).resolve().parents[1]

SENSITIVE_ASSIGNMENTS: frozenset[str] = frozenset(
    {
        "LABELS",
        "LABEL_COLORS",
        "LABEL_TO_SEMANTIC",
        "SEMANTIC_COLORS",
        "SEMANTIC_IDS",
        "SEMANTIC_NAMES",
        "SEMANTIC_TO_LABEL",
    }
)

ALLOWED_CONTRACT_REFERENCES: frozenset[str] = frozenset(
    {
        "LABEL_TO_SEMANTIC",
        "SEMANTIC_COLORS",
        "SEMANTIC_TO_LABEL",
    }
)


def assignment_targets(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def imports_semantic_contract(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "scripts.semantic_label_contract":
            return True
    return False


def value_uses_allowed_contract_name(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Name) and child.id in ALLOWED_CONTRACT_REFERENCES for child in ast.walk(node))


def validate_script(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    errors: list[str] = []
    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path)

    if not imports_semantic_contract(tree):
        errors.append("missing_semantic_label_contract_import")

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target_names = [name for name in assignment_targets(node) if name in SENSITIVE_ASSIGNMENTS]
        if not target_names:
            continue
        value = node.value
        if value is not None and value_uses_allowed_contract_name(value):
            continue
        errors.append(f"local_semantic_contract_assignment:{','.join(target_names)}:line={node.lineno}")

    return {
        "path": display_path,
        "passed": not errors,
        "errors": errors,
    }


def validate(paths: tuple[str, ...] = PROTECTED_SEMANTIC_CONTRACT_SCRIPT_PATHS) -> dict[str, Any]:
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
        "schema": "semantic-contract-usage/v1",
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
            print(f"semantic contract usage ok: {report['protected_script_count']} scripts")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
