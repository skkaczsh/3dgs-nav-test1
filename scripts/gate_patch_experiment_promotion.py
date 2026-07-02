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
CANDIDATE_RUN_NAMES = {
    "v2_bucket_attach": "v2",
    "v5_fragment_evidence": "v5",
}
METRIC_KEYS = (
    "patch_count",
    "high_entropy_count",
    "large_high_entropy_count",
    "large_low_purity_count",
)


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
        "raw": data,
    }


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return whether run a is no worse than b on all gate metrics and better on one."""

    better = False
    for key in METRIC_KEYS:
        av = _metric_value(a, key)
        bv = _metric_value(b, key)
        if av is None or bv is None:
            return False
        if av > bv:
            return False
        if av < bv:
            better = True
    return better


def validate_metric_acceptance(visual_data: dict[str, Any]) -> dict[str, Any]:
    """Validate selected patch candidate against quantitative geometry metrics.

    Visual review is necessary but not sufficient.  A candidate that is
    dominated by another reviewed run on patch count and mixed-bucket indicators
    should not be promoted into semantic evidence stages, because it carries a
    strictly worse fragmentation/mixing tradeoff before any human label enters
    the system.
    """

    selected_candidate = visual_data.get("selected_candidate")
    selected_run = CANDIDATE_RUN_NAMES.get(str(selected_candidate))
    summary = visual_data.get("comparison_summary") or {}
    errors: list[str] = []
    if not isinstance(summary, dict) or not summary:
        errors.append("metric_comparison_summary_missing")
        return {"accepted": False, "selected_run": selected_run, "errors": errors, "dominated_by": []}
    if selected_run is None:
        errors.append(f"metric_candidate_run_unknown={selected_candidate}")
        return {"accepted": False, "selected_run": selected_run, "errors": errors, "dominated_by": []}
    selected_metrics = summary.get(selected_run)
    if not isinstance(selected_metrics, dict):
        errors.append(f"metric_selected_run_missing={selected_run}")
        return {"accepted": False, "selected_run": selected_run, "errors": errors, "dominated_by": []}
    missing = [key for key in METRIC_KEYS if _metric_value(selected_metrics, key) is None]
    if missing:
        errors.append(f"metric_selected_run_missing_keys={missing}")

    dominated_by: list[str] = []
    for name, row in summary.items():
        if name == selected_run or not isinstance(row, dict):
            continue
        row_missing = [key for key in METRIC_KEYS if _metric_value(row, key) is None]
        if row_missing:
            continue
        if _dominates(row, selected_metrics):
            dominated_by.append(str(name))
    if dominated_by:
        errors.append(f"metric_selected_run_dominated_by={dominated_by}")

    return {
        "accepted": not errors,
        "selected_run": selected_run,
        "metric_keys": list(METRIC_KEYS),
        "selected_metrics": selected_metrics,
        "dominated_by": dominated_by,
        "errors": errors,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    visual = validate_visual_acceptance(args.visual_acceptance)
    raw_visual = visual.pop("raw", {}) if isinstance(visual.get("raw"), dict) else {}
    metrics = validate_metric_acceptance(raw_visual)
    reasons = list(visual["errors"])
    reasons.extend(metrics["errors"])
    return {
        "schema": "patch-experiment-promotion-gate/v1",
        "status": "pass" if not reasons else "fail",
        "candidate": visual.get("selected_candidate"),
        "visual_acceptance": str(args.visual_acceptance),
        "visual": visual,
        "metrics": metrics,
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
