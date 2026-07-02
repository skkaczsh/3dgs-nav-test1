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


def iter_list_paths(row: dict[str, Any], key: str) -> set[str]:
    value = row.get(key)
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


def iter_current_dense_input_paths(data: dict[str, Any]) -> set[str]:
    """Return only canonical dense inputs, not baseline outputs.

    current_dense_patch_state.json also records local_paths for current patch and
    object baselines. Those are review/output artifacts and must not become
    production inputs merely because they are documented in the state file.
    """

    found: set[str] = set()
    for key in ("authoritative_source", "derived_dense_input", "remote_executable_baseline"):
        row = data.get(key)
        if not isinstance(row, dict):
            continue
        found.update(iter_list_paths(row, "local_paths"))
        found.update(iter_list_paths(row, "remote_paths"))

    latest = data.get("latest_remote_run")
    if isinstance(latest, dict):
        inputs = latest.get("inputs")
        if isinstance(inputs, dict):
            found.update(str(item) for item in inputs.values())
    return found


def normalize_path_text(path: str) -> str:
    return str(Path(path).expanduser()) if path.startswith(("~", ".")) else str(path)


def load_dense_allowlist(state_path: Path) -> set[str]:
    state = read_json(state_path)
    if state.get("schema") != "current-dense-patch-state/v1":
        raise ValueError(f"unexpected dense patch state schema: {state.get('schema')!r}")
    return {normalize_path_text(path) for path in iter_current_dense_input_paths(state)}


def iter_output_artifact_paths(data: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for key in ("current_patch_baseline", "current_object_baseline"):
        row = data.get(key)
        if not isinstance(row, dict):
            continue
        found.update(iter_list_paths(row, "local_paths"))
        found.update(iter_list_paths(row, "remote_paths"))
    return found


def validate_dense_allowlist(state_path: Path = DEFAULT_STATE) -> dict[str, Any]:
    state = read_json(state_path)
    if state.get("schema") != "current-dense-patch-state/v1":
        raise ValueError(f"unexpected dense patch state schema: {state.get('schema')!r}")
    allowed = {normalize_path_text(path) for path in iter_current_dense_input_paths(state)}
    output_artifacts = {normalize_path_text(path) for path in iter_output_artifact_paths(state)}
    errors: list[str] = []
    warnings: list[str] = []

    if not allowed:
        errors.append("dense_allowlist_empty")
    for path in sorted(allowed):
        forbidden = forbidden_production_input_match(path)
        if forbidden:
            errors.append(f"dense_allowlist_contains_forbidden={forbidden}:{path}")
        if "stride10" in path:
            errors.append(f"dense_allowlist_contains_stride_preview={path}")
    for path in sorted(allowed & output_artifacts):
        errors.append(f"dense_allowlist_contains_output_artifact={path}")
    if len(allowed) < 4:
        warnings.append(f"dense_allowlist_small={len(allowed)}")

    return {
        "schema": "dense-production-input-allowlist/v1",
        "passed": not errors,
        "state": str(state_path),
        "allowed_count": len(allowed),
        "output_artifact_count": len(output_artifacts),
        "errors": errors,
        "warnings": warnings,
    }


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
