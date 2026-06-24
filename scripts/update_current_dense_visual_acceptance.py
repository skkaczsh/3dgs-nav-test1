#!/usr/bin/env python3
"""Safely update one check in the current dense visual acceptance record."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTANCE = REPO_ROOT / "docs" / "current_dense_visual_acceptance.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "current_dense_visual_acceptance.md"
DEFAULT_GATE = REPO_ROOT / "docs" / "current_dense_promotion_gate.json"
VALID_STATUSES = {"pending", "accepted", "rejected", "blocked"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def recompute_status(record: dict[str, Any]) -> str:
    checks = record.get("checks") or []
    required = [row for row in checks if isinstance(row, dict) and row.get("required")]
    if any(row.get("status") in {"rejected", "blocked"} for row in required):
        return "blocked"
    if required and all(row.get("status") == "accepted" for row in required):
        return "accepted"
    return "pending"


def update_record(args: argparse.Namespace) -> dict[str, Any]:
    if args.status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {args.status}")
    record = read_json(args.acceptance)
    if record.get("schema") != "current-dense-visual-acceptance/v1":
        raise ValueError("unexpected visual acceptance schema")

    checks = record.get("checks")
    if not isinstance(checks, list):
        raise ValueError("checks must be a list")
    matches = [row for row in checks if isinstance(row, dict) and row.get("id") == args.check_id]
    if len(matches) != 1:
        raise ValueError(f"check id must match exactly one row: {args.check_id}")
    row = matches[0]
    row["status"] = args.status
    if args.notes is not None:
        row["notes"] = args.notes
    if args.evidence:
        row["evidence"] = [*list(row.get("evidence") or []), *args.evidence]

    if args.reviewer:
        record["reviewer"] = args.reviewer
    if args.summary is not None:
        record["summary"] = args.summary
    record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    record["status"] = recompute_status(record)
    return record


def format_md(record: dict[str, Any]) -> str:
    lines = [
        "# Current Dense Visual Acceptance",
        "",
        f"Status: `{record['status']}`",
        f"Candidate: `{record['accepted_candidate']}`",
        f"Review index: {record['review_index_url']}",
        f"Reviewer: `{record.get('reviewer', '')}`",
        f"Reviewed at: `{record.get('reviewed_at', '')}`",
        "",
        "## Required Checks",
        "",
    ]
    for row in record["checks"]:
        required = "required" if row.get("required") else "optional"
        notes = f" Notes: {row.get('notes', '')}" if row.get("notes") else ""
        lines.append(f"- `{row['id']}` [{required}] `{row['status']}`: {row['question']}{notes}")
    lines.extend(
        [
            "",
            "## Promotion",
            "",
            "Run `python3 scripts/gate_current_dense_mainline_promotion.py --qa-json docs/current_dense_mainline_qa.json --visual-acceptance docs/current_dense_visual_acceptance.json --output docs/current_dense_promotion_gate.json` after updating checks.",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "gate_current_dense_mainline_promotion.py"),
        "--qa-json",
        str(REPO_ROOT / "docs" / "current_dense_mainline_qa.json"),
        "--visual-acceptance",
        str(args.acceptance),
        "--output",
        str(args.gate_output),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--gate-output", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--check-id", required=True)
    parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    parser.add_argument("--notes")
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--reviewer")
    parser.add_argument("--summary")
    parser.add_argument("--run-gate", action="store_true")
    args = parser.parse_args()

    record = update_record(args)
    args.acceptance.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(format_md(record), encoding="utf-8")
    print(json.dumps({"acceptance": str(args.acceptance), "status": record["status"]}, ensure_ascii=False, indent=2))
    if args.run_gate:
        return run_gate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
