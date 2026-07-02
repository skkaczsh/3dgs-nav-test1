#!/usr/bin/env python3
"""Prepare the manual visual acceptance record for the dense mainline review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA = REPO_ROOT / "docs" / "current_dense_mainline_qa.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "current_dense_visual_acceptance.json"
DEFAULT_MD = REPO_ROOT / "docs" / "current_dense_visual_acceptance.md"
DEFAULT_REVIEW_URL = "http://127.0.0.1:8765/docs/current_dense_review_index.html"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def default_checks() -> list[dict[str, Any]]:
    return [
        {
            "id": "v8_fragmentation_improves",
            "required": True,
            "status": "pending",
            "question": "v8 visibly reduces object fragmentation compared with v7 in the same areas.",
            "evidence": ["v7 Object Refinement", "v8 Object Refinement"],
            "artifact_ids": ["v7_object_refinement", "v8_object_refinement"],
            "notes": "",
        },
        {
            "id": "v8_no_obvious_overmerge",
            "required": True,
            "status": "pending",
            "question": "v8 does not visibly merge unrelated large structures such as ground/building/tree into one object.",
            "evidence": ["v8 Object Refinement object mode", "v8 Object Refinement semantic mode"],
            "artifact_ids": ["v8_object_refinement"],
            "notes": "",
        },
        {
            "id": "surface_guard_no_unknown_regression",
            "required": True,
            "status": "pending",
            "question": "v17 keeps floor/wall visible and does not reproduce the v15/v16 unknown spike.",
            "evidence": ["v9 Teacher Semantic", "v17 Surface Preserve Guard"],
            "artifact_ids": ["v9_teacher_semantic", "v17_surface_preserve_guard"],
            "notes": "",
        },
        {
            "id": "semantic_not_promoted_from_object_view",
            "required": True,
            "status": "pending",
            "question": "Object refinement is only promoted as geometry ownership; semantic labels remain evidence/QA references.",
            "evidence": ["current_dense_mainline_qa", "current_dense_review_index"],
            "artifact_ids": ["v8_object_refinement", "v9_teacher_semantic", "v17_surface_preserve_guard"],
            "notes": "",
        },
    ]


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    qa = read_json(args.qa_json)
    existing = read_json(args.output) if args.output.exists() and not args.force else {}
    checks = existing.get("checks") or default_checks()
    required = [row for row in checks if row.get("required")]
    all_required_accepted = bool(required) and all(row.get("status") == "accepted" for row in required)
    blocked = any(row.get("status") in {"rejected", "blocked"} for row in required)
    return {
        "schema": "current-dense-visual-acceptance/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if all_required_accepted else ("blocked" if blocked else "pending"),
        "accepted_candidate": "v8_object_refinement",
        "review_index_url": args.review_index_url,
        "qa_json": str(args.qa_json),
        "qa_summary": {
            "accepted_delta": qa["object_refinement"]["metrics"]["delta_v8_minus_v7"]["accepted_candidate_rows"],
            "output_object_delta": qa["object_refinement"]["metrics"]["delta_v8_minus_v7"]["output_object_count"],
            "overlap_delta": qa["object_refinement"]["metrics"]["delta_v8_minus_v7"]["mixed_object_voxel_ratio_020"],
            "surface_guard_label_delta": qa["surface_guard"]["label_point_counts"]["delta_v17_minus_v9"],
        },
        "reviewer": existing.get("reviewer", ""),
        "reviewed_at": existing.get("reviewed_at", ""),
        "summary": existing.get("summary", ""),
        "checks": checks,
    }


def format_md(record: dict[str, Any]) -> str:
    lines = [
        "# Current Dense Visual Acceptance",
        "",
        f"Status: `{record['status']}`",
        f"Candidate: `{record['accepted_candidate']}`",
        f"Review index: {record['review_index_url']}",
        "",
        "## QA Summary",
        "",
    ]
    for key, value in record["qa_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Required Checks", ""])
    for row in record["checks"]:
        required = "required" if row.get("required") else "optional"
        artifact_ids = ", ".join(f"`{item}`" for item in row.get("artifact_ids", []))
        suffix = f" Artifacts: {artifact_ids}" if artifact_ids else ""
        lines.append(f"- `{row['id']}` [{required}] `{row['status']}`: {row['question']}{suffix}")
    lines.extend(
        [
            "",
            "## Promotion",
            "",
            "Promotion remains blocked until every required check is set to `accepted` in `docs/current_dense_visual_acceptance.json` and `gate_current_dense_mainline_promotion.py` passes.",
            "",
            "## Update Commands",
            "",
            "After visual inspection, update each required check with:",
            "",
            "```bash",
            "python3 scripts/update_current_dense_visual_acceptance.py \\",
            "  --check-id v8_fragmentation_improves \\",
            "  --status accepted \\",
            "  --reviewer \"<name>\" \\",
            "  --notes \"<brief evidence>\"",
            "```",
            "",
            "Valid statuses are `pending`, `accepted`, `rejected`, and `blocked`. Promotion only passes after all required checks are `accepted` and:",
            "",
            "```bash",
            "python3 scripts/gate_current_dense_mainline_promotion.py \\",
            "  --qa-json docs/current_dense_mainline_qa.json \\",
            "  --visual-acceptance docs/current_dense_visual_acceptance.json \\",
            "  --output docs/current_dense_promotion_gate.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--review-index-url", default=DEFAULT_REVIEW_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    record = build_record(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(format_md(record), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "output_md": str(args.output_md), "status": record["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
