#!/usr/bin/env python3
"""Gate promotion of patch experiments into the object/semantic pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISUAL = REPO_ROOT / "docs" / "patch_experiment_visual_acceptance.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "patch_experiment_promotion_gate.json"
ALLOWED_CANDIDATES = {"v2_bucket_attach", "v5_fragment_evidence"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate_visual_acceptance(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "accepted": False,
            "status": "missing",
            "selected_candidate": None,
            "errors": [f"missing_visual_acceptance={path}"],
        }
    data = read_json(path)
    errors: list[str] = []
    if data.get("schema") != "patch-experiment-visual-acceptance/v1":
        errors.append("unexpected_visual_acceptance_schema")
    if data.get("status") != "accepted":
        errors.append(f"visual_status_not_accepted={data.get('status')}")
    selected_candidate = data.get("selected_candidate")
    if selected_candidate not in ALLOWED_CANDIDATES:
        errors.append(f"candidate_not_allowed={selected_candidate}")
    if data.get("candidate_policy") != "geometry_input_only":
        errors.append(f"candidate_policy_not_geometry_input_only={data.get('candidate_policy')}")
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
    if not review_url.endswith("/docs/patch_experiment_review_index.html"):
        errors.append("visual_review_index_url_not_patch_experiment")
    return {
        "accepted": not errors,
        "status": data.get("status"),
        "selected_candidate": selected_candidate,
        "errors": errors,
        "reviewer": data.get("reviewer", ""),
        "reviewed_at": data.get("reviewed_at", ""),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    visual = validate_visual_acceptance(args.visual_acceptance)
    reasons = list(visual["errors"])
    return {
        "schema": "patch-experiment-promotion-gate/v1",
        "status": "pass" if not reasons else "fail",
        "candidate": visual.get("selected_candidate"),
        "visual_acceptance": str(args.visual_acceptance),
        "visual": visual,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-acceptance", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
