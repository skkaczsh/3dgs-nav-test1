#!/usr/bin/env python3
"""Gate promotion of the current dense mainline candidate.

Metrics can make a candidate eligible for visual QA, but they cannot promote it
alone.  This gate requires both quantitative invariants and an explicit manual
visual acceptance record for the review index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA = REPO_ROOT / "docs" / "current_dense_mainline_qa.json"
DEFAULT_VISUAL = REPO_ROOT / "docs" / "current_dense_visual_acceptance.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "current_dense_promotion_gate.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def fail(reasons: list[str], message: str) -> None:
    reasons.append(message)


def numeric(data: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def validate_visual_acceptance(path: Path | None, require_visual_acceptance: bool) -> dict[str, Any]:
    if not require_visual_acceptance:
        return {"required": False, "accepted": True, "status": "not_required", "errors": []}
    if path is None or not path.exists():
        return {
            "required": True,
            "accepted": False,
            "status": "missing",
            "errors": [f"missing_visual_acceptance={path}"],
        }
    data = read_json(path)
    errors: list[str] = []
    if data.get("schema") != "current-dense-visual-acceptance/v1":
        errors.append("unexpected_visual_acceptance_schema")
    if data.get("status") != "accepted":
        errors.append(f"visual_status_not_accepted={data.get('status')}")
    if data.get("accepted_candidate") != "v8_object_refinement":
        errors.append(f"accepted_candidate_not_v8={data.get('accepted_candidate')}")
    checks = data.get("checks") or []
    if not isinstance(checks, list) or not checks:
        errors.append("visual_checks_empty")
    else:
        required = [row for row in checks if isinstance(row, dict) and row.get("required")]
        if not required:
            errors.append("visual_checks_no_required_rows")
        bad = [row.get("id", "<unknown>") for row in required if row.get("status") != "accepted"]
        if bad:
            errors.append(f"visual_required_checks_not_accepted={bad}")
    review_url = str(data.get("review_index_url") or "")
    if not review_url.endswith("/docs/current_dense_review_index.html"):
        errors.append("visual_review_index_url_not_current")
    return {
        "required": True,
        "accepted": not errors,
        "status": data.get("status"),
        "errors": errors,
        "reviewer": data.get("reviewer", ""),
        "reviewed_at": data.get("reviewed_at", ""),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    qa = read_json(args.qa_json)
    reasons: list[str] = []

    accepted_delta = numeric(qa, ("object_refinement", "metrics", "delta_v8_minus_v7", "accepted_candidate_rows"))
    output_delta = numeric(qa, ("object_refinement", "metrics", "delta_v8_minus_v7", "output_object_count"))
    overlap_delta = numeric(
        qa, ("object_refinement", "metrics", "delta_v8_minus_v7", "mixed_object_voxel_ratio_020")
    )
    surface_delta = qa.get("surface_guard", {}).get("label_point_counts", {}).get("delta_v17_minus_v9", {})
    nonzero_surface_delta = {str(k): v for k, v in surface_delta.items() if int(v) != 0}
    unknown_delta = numeric(qa, ("surface_guard", "unknown_point_delta_v17_minus_v9"))

    if accepted_delta < args.min_accepted_delta:
        fail(reasons, f"accepted_delta {accepted_delta:g} < {args.min_accepted_delta:g}")
    if output_delta > args.max_output_object_delta:
        fail(reasons, f"output_object_delta {output_delta:g} > {args.max_output_object_delta:g}")
    if overlap_delta > args.max_overlap_delta:
        fail(reasons, f"overlap_delta {overlap_delta:.6f} > {args.max_overlap_delta:.6f}")
    if nonzero_surface_delta:
        fail(reasons, f"surface_guard_changed_labels={nonzero_surface_delta}")
    if unknown_delta > args.max_unknown_point_delta:
        fail(reasons, f"unknown_point_delta {unknown_delta:g} > {args.max_unknown_point_delta:g}")

    visual = validate_visual_acceptance(args.visual_acceptance, not args.no_require_visual_acceptance)
    if not visual["accepted"]:
        reasons.extend(str(item) for item in visual["errors"])

    return {
        "schema": "current-dense-promotion-gate/v1",
        "status": "pass" if not reasons else "fail",
        "candidate": "v8_object_refinement",
        "qa_json": str(args.qa_json),
        "visual_acceptance": str(args.visual_acceptance) if args.visual_acceptance else None,
        "thresholds": {
            "min_accepted_delta": args.min_accepted_delta,
            "max_output_object_delta": args.max_output_object_delta,
            "max_overlap_delta": args.max_overlap_delta,
            "max_unknown_point_delta": args.max_unknown_point_delta,
            "require_visual_acceptance": not args.no_require_visual_acceptance,
        },
        "metrics": {
            "accepted_delta": accepted_delta,
            "output_object_delta": output_delta,
            "overlap_delta": overlap_delta,
            "unknown_point_delta": unknown_delta,
            "nonzero_surface_delta": nonzero_surface_delta,
        },
        "visual": visual,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument("--visual-acceptance", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-accepted-delta", type=float, default=1.0)
    parser.add_argument("--max-output-object-delta", type=float, default=0.0)
    parser.add_argument("--max-overlap-delta", type=float, default=0.0)
    parser.add_argument("--max-unknown-point-delta", type=float, default=0.0)
    parser.add_argument("--no-require-visual-acceptance", action="store_true")
    args = parser.parse_args()

    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
