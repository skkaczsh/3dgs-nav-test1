#!/usr/bin/env python3
"""Validate the current parking semantic architecture decision file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.current_mainline_contract import REQUIRED_ACTIVE_BASELINE_IDS, REQUIRED_REJECTED_ARTIFACT_IDS

REQUIRED_TOP_LEVEL = {
    "schema",
    "updated_at",
    "dataset",
    "objective",
    "current_diagnosis",
    "dense_sources",
    "active_baselines",
    "rejected_artifacts",
    "architecture_principles",
    "next_mainline",
}

REQUIRED_REJECTED_IDS = set(REQUIRED_REJECTED_ARTIFACT_IDS)
REQUIRED_ACTIVE_IDS = set(REQUIRED_ACTIVE_BASELINE_IDS)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("architecture file must contain a JSON object")
    return data


def _iter_local_paths(data: Any) -> list[str]:
    found: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "local_paths" and isinstance(value, list):
                found.extend(str(item) for item in value)
            else:
                found.extend(_iter_local_paths(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(_iter_local_paths(item))
    return found


def validate(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    errors: list[str] = []
    warnings: list[str] = []

    missing_top = sorted(REQUIRED_TOP_LEVEL - set(data))
    errors.extend(f"missing_top_level={key}" for key in missing_top)

    if data.get("schema") != "current-project-architecture/v1":
        errors.append(f"unexpected_schema={data.get('schema')!r}")

    active_ids = {str(item.get("id")) for item in data.get("active_baselines", []) if isinstance(item, dict)}
    rejected_ids = {str(item.get("id")) for item in data.get("rejected_artifacts", []) if isinstance(item, dict)}

    missing_active = sorted(REQUIRED_ACTIVE_IDS - active_ids)
    errors.extend(f"missing_active_baseline={item}" for item in missing_active)

    missing_rejected = sorted(REQUIRED_REJECTED_IDS - rejected_ids)
    errors.extend(f"missing_rejected_artifact={item}" for item in missing_rejected)

    active_rejected_overlap = sorted(active_ids & rejected_ids)
    errors.extend(f"active_baseline_is_rejected={item}" for item in active_rejected_overlap)

    local_paths = _iter_local_paths(data)
    missing_paths = [item for item in local_paths if item.startswith("/") and not Path(item).exists()]
    errors.extend(f"missing_local_path={item}" for item in missing_paths)

    if len(data.get("architecture_principles", [])) < 5:
        warnings.append("architecture_principles_too_sparse")
    if len(data.get("next_mainline", [])) < 3:
        warnings.append("next_mainline_too_sparse")

    return {
        "passed": not errors,
        "path": str(path),
        "active_baseline_count": len(active_ids),
        "rejected_artifact_count": len(rejected_ids),
        "checked_local_path_count": len(local_paths),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        default="docs/current_project_architecture.json",
        help="Path to current_project_architecture.json",
    )
    args = parser.parse_args()

    report = validate(Path(args.architecture))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
