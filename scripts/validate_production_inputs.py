#!/usr/bin/env python3
"""Reject paths that must not enter dense production pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.current_mainline_contract import forbidden_production_input_match


def validate_paths(paths: list[str]) -> dict:
    errors = []
    checked = []
    for path in paths:
        forbidden = forbidden_production_input_match(path)
        checked.append({"path": path, "forbidden_match": forbidden})
        if forbidden:
            errors.append(f"forbidden_production_input={forbidden}:{path}")
    return {
        "schema": "production-input-validation/v1",
        "passed": not errors,
        "checked_count": len(paths),
        "checked": checked,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    report = validate_paths([str(path) for path in args.paths])
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
