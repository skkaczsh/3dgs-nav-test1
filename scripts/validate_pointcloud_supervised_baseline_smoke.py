#!/usr/bin/env python3
"""Validate the supervised point-cloud baseline smoke contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SCHEMA = "pointcloud-supervised-baseline-smoke/v1"
REQUIRED_INPUT_ID = "dense_las_voxel003_binary"
REQUIRED_ABLATIONS = {
    "xyz",
    "xyz_rgb",
    "xyz_normal",
    "xyz_rgb_normal_height",
}
REQUIRED_OUTPUTS = {
    "per_voxel_semantic_logits_or_labels",
    "per_patch_vote_summary",
    "domain_gap_report",
    "viewer_preview_for_qa",
}
REQUIRED_METRICS = {
    "label_histogram",
    "unknown_ratio",
    "surface_label_ratio",
    "patch_vote_entropy",
    "large_surface_conflict_count",
    "feature_ablation_delta",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate(path: Path) -> dict[str, Any]:
    data = load_json(path)
    errors: list[str] = []
    warnings: list[str] = []

    if data.get("schema") != REQUIRED_SCHEMA:
        errors.append(f"unexpected_schema={data.get('schema')!r}")
    if data.get("role") != "domain_gap_and_semantic_evidence_baseline":
        errors.append(f"unexpected_role={data.get('role')!r}")

    input_source = data.get("input_source") or {}
    if input_source.get("id") != REQUIRED_INPUT_ID:
        errors.append(f"unexpected_input_source={input_source.get('id')!r}")
    if abs(float(input_source.get("required_voxel_size_m", 0.0)) - 0.03) > 1e-9:
        errors.append("input_source_not_voxel003")

    ablations = {
        str(item.get("id"))
        for item in data.get("feature_ablations", [])
        if isinstance(item, dict)
    }
    for missing in sorted(REQUIRED_ABLATIONS - ablations):
        errors.append(f"missing_ablation={missing}")

    outputs = {str(item) for item in data.get("expected_outputs", [])}
    for missing in sorted(REQUIRED_OUTPUTS - outputs):
        errors.append(f"missing_expected_output={missing}")

    metrics = {str(item) for item in data.get("minimum_report_metrics", [])}
    for missing in sorted(REQUIRED_METRICS - metrics):
        errors.append(f"missing_metric={missing}")

    hard = data.get("hard_constraints") or {}
    for key in (
        "may_create_patch_boundaries",
        "may_merge_or_split_objects",
        "may_override_geometry_owner",
        "may_write_canonical_label_without_evidence_fusion",
    ):
        if hard.get(key) is not False:
            errors.append(f"hard_constraint_must_be_false={key}")
    if hard.get("must_map_outputs_to_existing_patch_or_voxel_ids") is not True:
        errors.append("must_map_outputs_to_existing_patch_or_voxel_ids")

    forbidden = " ".join(str(item).lower() for item in data.get("forbidden_use", []))
    if "stride viewer ply" not in forbidden:
        warnings.append("forbidden_use_should_mention_stride_viewer_ply")
    if "taxonomy" not in forbidden:
        warnings.append("forbidden_use_should_mention_taxonomy")

    return {
        "passed": not errors,
        "path": str(path),
        "ablation_count": len(ablations),
        "expected_output_count": len(outputs),
        "metric_count": len(metrics),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/pointcloud_supervised_baseline_smoke_20260708.json"),
    )
    args = parser.parse_args()
    report = validate(args.contract)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
