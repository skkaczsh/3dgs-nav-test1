#!/usr/bin/env python3
"""Reject paths that must not enter dense production pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.current_mainline_contract import forbidden_production_input_match


DEFAULT_STATE = REPO_ROOT / "docs" / "current_dense_patch_state.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def iter_declared_dense_paths(data: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if key in {"local_paths", "remote_paths"} and isinstance(value, list):
                found.update(str(item) for item in value)
            else:
                found.update(iter_declared_dense_paths(value))
    elif isinstance(data, list):
        for item in data:
            found.update(iter_declared_dense_paths(item))
    return found


def normalize_path_text(path: str) -> str:
    return str(Path(path).expanduser()) if path.startswith(("~", ".")) else str(path)


def load_dense_allowlist(state_path: Path) -> set[str]:
    state = read_json(state_path)
    if state.get("schema") != "current-dense-patch-state/v1":
        raise ValueError(f"unexpected dense patch state schema: {state.get('schema')!r}")
    return {normalize_path_text(path) for path in iter_declared_dense_paths(state)}


def validate_paths(paths: list[str], *, allowed_paths: set[str] | None = None) -> dict:
    errors = []
    checked = []
    for path in paths:
        normalized = normalize_path_text(path)
        forbidden = forbidden_production_input_match(normalized)
        allowlist_match = allowed_paths is None or normalized in allowed_paths
        checked.append({"path": path, "normalized": normalized, "forbidden_match": forbidden, "allowlist_match": allowlist_match})
        if forbidden:
            errors.append(f"forbidden_production_input={forbidden}:{path}")
        if allowed_paths is not None and not allowlist_match:
            errors.append(f"not_current_dense_input:{path}")
    return {
        "schema": "production-input-validation/v1",
        "passed": not errors,
        "checked_count": len(paths),
        "require_current_dense": allowed_paths is not None,
        "checked": checked,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--require-current-dense", action="store_true", help="Require every path to be declared in current_dense_patch_state.json")
    args = parser.parse_args()

    allowed_paths = load_dense_allowlist(args.state) if args.require_current_dense else None
    report = validate_paths([str(path) for path in args.paths], allowed_paths=allowed_paths)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        print(f"production inputs ok: {report['checked_count']} path(s)")
    else:
        for error in report["errors"]:
            print(error, file=sys.stderr)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
