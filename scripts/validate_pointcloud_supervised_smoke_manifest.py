#!/usr/bin/env python3
"""Validate the supervised point-cloud smoke crop manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SCHEMA = "pointcloud-supervised-baseline-smoke-manifest/v1"
REQUIRED_GEOMETRY = {"horizontal", "vertical", "rough_mixed", "thin_linear"}


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
    dense = data.get("dense_input") or {}
    if dense.get("id") != "dense_las_voxel003_binary":
        errors.append(f"unexpected_dense_input={dense.get('id')!r}")
    if abs(float(dense.get("voxel_size_m", 0.0)) - 0.03) > 1e-9:
        errors.append("dense_input_not_voxel003")
    if int(dense.get("voxel_count", 0)) != 14482557:
        errors.append(f"unexpected_voxel_count={dense.get('voxel_count')}")
    crops = [item for item in data.get("crops", []) if isinstance(item, dict)]
    if len(crops) < 5:
        errors.append("too_few_crops")
    geometry = {str(item.get("geometry_type")) for item in crops}
    for missing in sorted(REQUIRED_GEOMETRY - geometry):
        errors.append(f"missing_geometry_crop={missing}")
    if "mixed_risk" not in geometry:
        errors.append("missing_known_risk_crop")
    for crop in crops:
        bbox = crop.get("bbox_3d") or {}
        mn = bbox.get("min")
        mx = bbox.get("max")
        if not (isinstance(mn, list) and isinstance(mx, list) and len(mn) == 3 and len(mx) == 3):
            errors.append(f"invalid_bbox={crop.get('id')}")
            continue
        if any(float(a) >= float(b) for a, b in zip(mn, mx)):
            errors.append(f"non_positive_bbox={crop.get('id')}")
        count = int(crop.get("dense_voxel_count_in_crop", 0))
        if count <= 0:
            errors.append(f"empty_crop={crop.get('id')}")
        elif count < 100:
            warnings.append(f"small_crop={crop.get('id')}:{count}")
    forbidden = " ".join(str(item) for item in (data.get("runner_contract") or {}).get("forbidden_outputs", []))
    if "new_patch_boundaries" not in forbidden or "overwritten_geometry_owner" not in forbidden:
        errors.append("runner_contract_missing_ownership_forbidden_outputs")
    return {
        "passed": not errors,
        "path": str(path),
        "crop_count": len(crops),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("docs/pointcloud_supervised_baseline_smoke_manifest_20260708.json"))
    args = parser.parse_args()
    report = validate(args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
