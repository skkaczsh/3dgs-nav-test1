#!/usr/bin/env python3
"""Validate the dataset delivery manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = []
    warnings = []

    if manifest.get("dataset", {}).get("projection_route") != "img_pos.txt + cam_in_ex.txt + Tcl + Til":
        errors.append("projection route is not the validated scanner-native route")
    if manifest.get("dataset", {}).get("semantic_combo") != "sam2_prompt_v3_sky_label_merge_completion":
        errors.append("semantic combo is not the current best baseline")
    if not manifest.get("passed"):
        errors.append("manifest passed flag is false")

    for row in manifest.get("files", []):
        if row.get("required") and not row.get("exists"):
            errors.append(f"required file missing in manifest: {row.get('role')} {row.get('path')}")
        if row.get("required") and int(row.get("bytes") or 0) <= 0:
            errors.append(f"required file has zero bytes in manifest: {row.get('role')} {row.get('path')}")
        path = Path(row.get("path", ""))
        if row.get("required") and not path.exists():
            errors.append(f"required file missing on disk: {row.get('role')} {path}")

    for check in manifest.get("checks", []):
        if not check.get("passed"):
            errors.append(f"threshold failed: {check}")

    viewer_inputs = manifest.get("recommended_viewer_inputs", [])
    if not viewer_inputs:
        errors.append("no recommended viewer inputs")
    for path_text in viewer_inputs:
        if not Path(path_text).exists():
            errors.append(f"recommended viewer input missing: {path_text}")

    result = {
        "manifest": str(args.manifest),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "required_file_count": sum(1 for row in manifest.get("files", []) if row.get("required")),
        "recommended_viewer_inputs": viewer_inputs,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
