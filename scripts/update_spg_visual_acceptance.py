#!/usr/bin/env python3
"""Update one check in the current SPG visual acceptance record."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTANCE = REPO_ROOT / "docs" / "superpoint_graph_v4_visual_acceptance.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "superpoint_graph_v4_visual_acceptance.md"
VALID_STATUSES = {"pending", "accepted", "failed"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def recompute_status(record: dict[str, Any]) -> str:
    required = [row for row in record.get("checks", []) if isinstance(row, dict) and row.get("required")]
    if any(row.get("status") == "failed" for row in required):
        return "failed"
    if required and all(row.get("status") == "accepted" for row in required):
        return "accepted"
    return "pending"


def update_record(args: argparse.Namespace) -> dict[str, Any]:
    record = read_json(args.acceptance)
    if record.get("schema") != "superpoint-graph-visual-acceptance/v1":
        raise ValueError("unexpected SPG visual acceptance schema")
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
    record["reviewer"] = args.reviewer or record.get("reviewer", "")
    record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    record["status"] = recompute_status(record)
    return record


def format_md(record: dict[str, Any]) -> str:
    title = record.get("title") or f"Superpoint Graph Visual Acceptance: {record['candidate']}"
    lines = [
        f"# {title}",
        "",
        f"Status: `{record['status']}`",
        f"Candidate: `{record['candidate']}`",
        f"Review doc: `{record.get('review_doc', '')}`",
        f"Viewer: {record.get('viewer_url', '')}",
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
    lines.extend(["", "Run `python3 scripts/validate_current_mainline.py` after updating checks.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check-id", required=True)
    parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    parser.add_argument("--notes")
    parser.add_argument("--reviewer")
    args = parser.parse_args()

    record = update_record(args)
    args.acceptance.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(format_md(record), encoding="utf-8")
    print(json.dumps({"acceptance": str(args.acceptance), "status": record["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
