#!/usr/bin/env python3
"""Plan the state edits for promoting the current dense object candidate.

This is intentionally a planner, not an editor.  The promotion gate and visual
acceptance record decide whether the candidate is eligible; this script turns
that decision into an explicit, reviewable state-change plan so operators do not
hand-edit current_dense_patch_state.json from memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = REPO_ROOT / "docs" / "current_dense_patch_state.json"
DEFAULT_QA = REPO_ROOT / "docs" / "current_dense_mainline_qa.json"
DEFAULT_GATE = REPO_ROOT / "docs" / "current_dense_promotion_gate.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def candidate_local_paths(source_run_id: str) -> dict[str, list[str]]:
    base = (
        REPO_ROOT
        / "server_parking_priority_s10"
        / "geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623"
        / source_run_id
        / "objects_v7_structural_multimaterial"
    )
    production = [
        str(base / "geo_patch_objects_v7_structural_multimaterial_report.json"),
        str(base / "geo_patch_objects_v7_structural_multimaterial.jsonl"),
    ]
    qa_only = [
        str(base / "geo_patch_objects_v7_structural_multimaterial_stride10.ply"),
        str(base / "voxel_overlap_020_report.json"),
    ]
    return {"production": production, "qa_only": qa_only}


def existing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if Path(path).exists()]


def build_plan(state: dict[str, Any], qa: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    candidate = state.get("current_promotion_candidate", {})
    if not isinstance(candidate, dict):
        errors.append("current_promotion_candidate_missing")
        candidate = {}
    candidate_id = str(candidate.get("id", ""))
    qa_candidate_id = str(candidate.get("qa_candidate_id", ""))
    source_run_id = str(candidate.get("source_run_id", ""))

    if gate.get("candidate") != candidate_id:
        errors.append(f"gate_candidate_mismatch={gate.get('candidate')}!={candidate_id}")
    if gate.get("status") != "pass":
        errors.append(f"promotion_gate_not_passed={gate.get('status')}")
    if qa.get("object_refinement", {}).get("candidate") != qa_candidate_id:
        errors.append(
            "qa_candidate_mismatch="
            f"{qa.get('object_refinement', {}).get('candidate')}!={qa_candidate_id}"
        )

    metrics = qa.get("object_refinement", {}).get("metrics", {}).get("v8", {})
    if not isinstance(metrics, dict) or not metrics:
        errors.append("qa_v8_metrics_missing")
        metrics = {}

    paths = candidate_local_paths(source_run_id)
    missing_production = sorted(set(paths["production"]) - set(existing_paths(paths["production"])))
    if missing_production:
        errors.extend(f"candidate_production_path_missing={path}" for path in missing_production)

    current_object = state.get("current_object_baseline", {})
    proposed_object_baseline = {
        "id": candidate_id,
        "status": "promoted_dense_object_geometry_baseline",
        "reason": (
            "Promoted after current_dense_promotion_gate passed and fixed visual checks accepted. "
            "This promotes geometry/object ownership only; semantic labels remain evidence."
        ),
        "local_paths": paths["production"],
        "qa_only_paths": paths["qa_only"],
        "metrics": {
            "input_patch_count": int(metrics.get("output_object_count", 0)) + int(metrics.get("accepted_candidate_rows", 0)),
            "output_object_count": int(metrics.get("output_object_count", 0)),
            "accepted_candidate_rows": int(metrics.get("accepted_candidate_rows", 0)),
            "candidate_count": int(metrics.get("candidate_count", 0)),
            "mixed_object_voxel_ratio_020": float(metrics.get("mixed_object_voxel_ratio_020", 0.0)),
            "object_count_in_overlap_preview": int(metrics.get("object_count_in_overlap_preview", 0)),
        },
    }
    proposed_promotion_candidate = {
        **candidate,
        "status": "promoted",
        "promoted_object_baseline_id": candidate_id,
    }
    return {
        "schema": "current-dense-promotion-plan/v1",
        "passed": not errors,
        "candidate": candidate_id,
        "gate_status": gate.get("status"),
        "errors": errors,
        "current_object_baseline": current_object,
        "proposed_object_baseline": proposed_object_baseline,
        "proposed_current_promotion_candidate": proposed_promotion_candidate,
        "manual_steps": [
            "Apply proposed_object_baseline to docs/current_dense_patch_state.json current_object_baseline.",
            "Mark current_promotion_candidate status as promoted only after the gate stays pass.",
            "Run python3 scripts/validate_current_mainline.py before committing.",
            "Do not promote qa_only_paths to production geometry inputs.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument("--promotion-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = build_plan(read_json(args.state), read_json(args.qa_json), read_json(args.promotion_gate))
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if plan["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
