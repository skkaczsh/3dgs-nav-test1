#!/usr/bin/env python3
"""Prepare the manual visual acceptance record for patch experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = (
    REPO_ROOT
    / "server_parking_priority_s10"
    / "geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623"
    / "geo_patch_run_comparison_v2_v5_20260702"
    / "comparison.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "patch_experiment_visual_acceptance.json"
DEFAULT_MD = REPO_ROOT / "docs" / "patch_experiment_visual_acceptance.md"
DEFAULT_REVIEW_URL = "http://127.0.0.1:8765/docs/patch_experiment_review_index.html"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def default_checks() -> list[dict[str, Any]]:
    return [
        {
            "id": "metric_comparison_reviewed",
            "required": True,
            "status": "pending",
            "question": "The v2/v5 metric comparison has been reviewed and the selected candidate is intentional.",
            "evidence": ["geo_patch_run_comparison_v2_v5_20260702", "patch_experiment_review_index"],
            "notes": "",
        },
        {
            "id": "no_major_structure_overmerge",
            "required": True,
            "status": "pending",
            "question": "The selected candidate does not visibly merge unrelated large structures such as ground/building/tree.",
            "evidence": ["selected candidate object-color viewer"],
            "notes": "",
        },
        {
            "id": "small_fragment_tradeoff_accepted",
            "required": True,
            "status": "pending",
            "question": "Residual small-fragment behavior is acceptable for the next object/semantic layer.",
            "evidence": ["selected candidate object-color viewer", "comparison QA"],
            "notes": "",
        },
        {
            "id": "semantic_layer_input_decision",
            "required": True,
            "status": "pending",
            "question": "The selected patch run is explicitly approved as geometry input only, not as semantic truth.",
            "evidence": ["patch_bucket_split_attach_eval_20260702.md"],
            "notes": "",
        },
    ]


def compact_run_summary(row: dict[str, Any]) -> dict[str, Any]:
    keep_keys = {
        "patch_count",
        "total_voxels",
        "high_entropy_count",
        "large_patch_count",
        "large_high_entropy_count",
        "large_low_purity_count",
        "large_extreme_aspect_count",
        "merge_accept_count",
        "accepted_attachment_count",
        "voxel_count_p50",
        "voxel_count_p90",
        "voxel_count_p99",
        "voxel_count_max",
        "bucket_entropy_p50",
        "bucket_entropy_p90",
        "bucket_entropy_p99",
    }
    return {key: row[key] for key in keep_keys if key in row}


def extract_run_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    runs = comparison.get("runs") or []
    if isinstance(runs, dict):
        return {str(name): compact_run_summary(row) for name, row in runs.items() if isinstance(row, dict)}
    if isinstance(runs, list):
        return {
            str(row.get("name") or row.get("run") or idx): compact_run_summary(row)
            for idx, row in enumerate(runs)
            if isinstance(row, dict)
        }
    summaries = comparison.get("summary") or comparison.get("summaries") or {}
    if not isinstance(summaries, dict):
        return {}
    return {
        str(name): compact_run_summary(row) if isinstance(row, dict) else row
        for name, row in summaries.items()
    }


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    existing = read_json(args.output) if args.output.exists() and not args.force else {}
    comparison = read_json(args.comparison_json) if args.comparison_json.exists() else {}
    checks = existing.get("checks") or default_checks()
    required = [row for row in checks if row.get("required")]
    all_required_accepted = bool(required) and all(row.get("status") == "accepted" for row in required)
    blocked = any(row.get("status") in {"rejected", "blocked"} for row in required)
    selected_candidate = existing.get("selected_candidate") or args.selected_candidate
    return {
        "schema": "patch-experiment-visual-acceptance/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if all_required_accepted else ("blocked" if blocked else "pending"),
        "selected_candidate": selected_candidate,
        "candidate_policy": "geometry_input_only",
        "review_index_url": args.review_index_url,
        "comparison_json": str(args.comparison_json),
        "comparison_summary": extract_run_summary(comparison),
        "reviewer": existing.get("reviewer", ""),
        "reviewed_at": existing.get("reviewed_at", ""),
        "summary": existing.get("summary", ""),
        "checks": checks,
    }


def format_md(record: dict[str, Any]) -> str:
    lines = [
        "# Patch Experiment Visual Acceptance",
        "",
        f"Status: `{record['status']}`",
        f"Selected candidate: `{record['selected_candidate']}`",
        f"Candidate policy: `{record['candidate_policy']}`",
        f"Review index: {record['review_index_url']}",
        "",
        "## Required Checks",
        "",
    ]
    for row in record["checks"]:
        required = "required" if row.get("required") else "optional"
        lines.append(f"- `{row['id']}` [{required}] `{row['status']}`: {row['question']}")
    lines.extend(
        [
            "",
            "## Promotion",
            "",
            "This experiment remains blocked from semantic/object promotion until every required check is set to `accepted` and `gate_patch_experiment_promotion.py` passes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--review-index-url", default=DEFAULT_REVIEW_URL)
    parser.add_argument("--selected-candidate", default="v2_bucket_attach")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    record = build_record(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(format_md(record), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "output_md": str(args.output_md), "status": record["status"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
