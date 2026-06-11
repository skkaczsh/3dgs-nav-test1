#!/usr/bin/env python3
"""Validate the manual visual acceptance record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/Users/skkac/Work/SCAN")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(path: Path, require_accepted: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    data = read_json(path)
    checks = data.get("checks", [])
    if data.get("review_version") != 1:
        errors.append("review_version must be 1")
    if not checks:
        errors.append("checks is empty")
    required = [row for row in checks if row.get("required")]
    if not required:
        errors.append("no required checks")
    bad_status = [
        row.get("id", "<unknown>")
        for row in checks
        if row.get("status") not in {"pending", "accepted", "rejected", "blocked"}
    ]
    if bad_status:
        errors.append(f"invalid check status: {bad_status}")
    pending_required = [row.get("id", "<unknown>") for row in required if row.get("status") == "pending"]
    failed_required = [row.get("id", "<unknown>") for row in required if row.get("status") in {"rejected", "blocked"}]
    if pending_required:
        warnings.append(f"pending required checks: {pending_required}")
    if failed_required:
        warnings.append(f"failed required checks: {failed_required}")
    accepted = not pending_required and not failed_required and not errors
    if data.get("allow_next_increment") != accepted:
        errors.append("allow_next_increment must match all required checks accepted")
    if require_accepted and not accepted:
        errors.append("visual acceptance is not accepted")
    return {
        "path": str(path),
        "passed": not errors,
        "accepted": accepted,
        "status": data.get("status"),
        "allow_next_increment": data.get("allow_next_increment"),
        "required_check_count": len(required),
        "pending_required": pending_required,
        "failed_required": failed_required,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=ROOT / "route_status_20260610/visual_acceptance_review_20260611.json")
    parser.add_argument("--output", type=Path, default=ROOT / "route_status_20260610/visual_acceptance_review_20260611_validation.json")
    parser.add_argument("--require-accepted", action="store_true")
    args = parser.parse_args()

    result = validate(args.review, args.require_accepted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
