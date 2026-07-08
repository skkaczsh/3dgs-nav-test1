#!/usr/bin/env python3
"""Validate exported supervised smoke crop PLYs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_SCHEMA = "pointcloud-supervised-smoke-crop-export/v1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> dict[str, Any]:
    data = load_json(path)
    errors: list[str] = []
    if data.get("schema") != REQUIRED_SCHEMA:
        errors.append(f"unexpected_schema={data.get('schema')!r}")
    crops = [crop for crop in data.get("crops", []) if isinstance(crop, dict)]
    if len(crops) < 5:
        errors.append("too_few_exported_crops")
    for crop in crops:
        ply = Path(str(crop.get("output_ply", "")))
        if not ply.exists():
            errors.append(f"missing_crop_ply={ply}")
            continue
        if int(crop.get("point_count", 0)) <= 0:
            errors.append(f"empty_crop={crop.get('id')}")
        if crop.get("count_matches_manifest") is not True:
            errors.append(f"count_mismatch={crop.get('id')}")
        expected_sha = str(crop.get("sha256", ""))
        if not expected_sha:
            errors.append(f"missing_sha256={crop.get('id')}")
        elif sha256_file(ply) != expected_sha:
            errors.append(f"sha256_mismatch={crop.get('id')}")
    return {
        "passed": not errors,
        "path": str(path),
        "crop_count": len(crops),
        "errors": errors,
        "warnings": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("server_parking_priority_s10/pointcloud_supervised_baseline_smoke_crops_20260708/crop_export_report.json"),
    )
    args = parser.parse_args()
    report = validate(args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
