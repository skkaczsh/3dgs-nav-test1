#!/usr/bin/env python3
"""Validate object semantic evidence-fusion outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_OUTPUT_FIELDS = {
    "semantic_label",
    "semantic_id",
    "semantic_fusion_status",
    "semantic_fusion_confidence",
    "semantic_evidence_scores",
    "semantic_evidence_source_scores",
    "semantic_vetoed_scores",
}
OWNERSHIP_FIELDS = (
    "geometry_type",
    "bbox_3d",
    "centroid",
    "mean_normal",
    "mean_rgb",
    "bucket_counts",
    "voxel_count",
    "patch_count",
)
MEMBERSHIP_FIELDS = (
    "patch_ids",
    "patch_ids_truncated",
    "target_ids",
    "target_indices",
    "voxel_indices",
    "point_indices",
    "merged_point_indices",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def object_key(row: dict[str, Any]) -> int | None:
    for key in ("object_id", "viewer_object_id"):
        try:
            return int(row.get(key))
        except (TypeError, ValueError):
            continue
    return None


def by_object_id(rows: list[dict[str, Any]], side: str, errors: list[str]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        oid = object_key(row)
        if oid is None:
            errors.append(f"{side}:missing_object_id:index={index}")
            continue
        if oid in out:
            errors.append(f"{side}:duplicate_object_id={oid}")
            continue
        out[oid] = row
    return out


def scene_only_promoted(row: dict[str, Any]) -> bool:
    if row.get("semantic_fusion_status") != "evidence_fusion_applied":
        return False
    label = str(row.get("semantic_label") or "unknown")
    by_source = row.get("semantic_evidence_source_scores")
    if not isinstance(by_source, dict):
        return False
    scene_scores = by_source.get("scene") if isinstance(by_source.get("scene"), dict) else {}
    sam_scores = by_source.get("sam") if isinstance(by_source.get("sam"), dict) else {}
    teacher_scores = by_source.get("teacher") if isinstance(by_source.get("teacher"), dict) else {}
    scene_score = float(scene_scores.get(label, 0.0) or 0.0)
    sam_score = float(sam_scores.get(label, 0.0) or 0.0)
    teacher_score = float(teacher_scores.get(label, 0.0) or 0.0)
    return scene_score > 0 and sam_score <= 0 and teacher_score <= 0


def validate(
    input_objects: Path,
    output_objects: Path,
    report_path: Path | None = None,
    *,
    allow_scene_only: bool = False,
) -> dict[str, Any]:
    input_rows = read_jsonl(input_objects)
    output_rows = read_jsonl(output_objects)
    errors: list[str] = []
    warnings: list[str] = []
    before = by_object_id(input_rows, "input", errors)
    after = by_object_id(output_rows, "output", errors)

    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    if missing:
        errors.append(f"missing_output_object_ids={missing[:20]}")
    if extra:
        errors.append(f"extra_output_object_ids={extra[:20]}")

    status_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for oid in sorted(set(before) & set(after)):
        src = before[oid]
        out = after[oid]
        missing_fields = sorted(field for field in REQUIRED_OUTPUT_FIELDS if field not in out)
        if missing_fields:
            errors.append(f"object={oid}:missing_fields={missing_fields}")
        for field in OWNERSHIP_FIELDS:
            if field in src and out.get(field) != src.get(field):
                errors.append(f"object={oid}:ownership_field_changed={field}")
        for field in MEMBERSHIP_FIELDS:
            if field in src and out.get(field) != src.get(field):
                errors.append(f"object={oid}:membership_field_changed={field}")
        if not isinstance(out.get("semantic_evidence_scores"), dict):
            errors.append(f"object={oid}:semantic_evidence_scores_not_object")
        if not isinstance(out.get("semantic_evidence_source_scores"), dict):
            errors.append(f"object={oid}:semantic_evidence_source_scores_not_object")
        if not isinstance(out.get("semantic_vetoed_scores"), dict):
            errors.append(f"object={oid}:semantic_vetoed_scores_not_object")
        if not allow_scene_only and scene_only_promoted(out):
            errors.append(f"object={oid}:scene_only_promotion")
        status_counts[str(out.get("semantic_fusion_status") or "missing")] += 1
        label_counts[str(out.get("semantic_label") or "unknown")] += 1

    report: dict[str, Any] | None = None
    if report_path is not None:
        if not report_path.exists():
            errors.append(f"missing_report={report_path}")
        else:
            report = read_json(report_path)
            if report.get("schema") != "object-semantic-evidence-fusion/v1":
                errors.append(f"unexpected_report_schema={report.get('schema')!r}")
            if int(report.get("object_count") or -1) != len(output_rows):
                errors.append("report_object_count_mismatch")

    return {
        "schema": "object-semantic-evidence-fusion-validation/v1",
        "passed": not errors,
        "input_objects": str(input_objects),
        "output_objects": str(output_objects),
        "report": str(report_path) if report_path else None,
        "object_count": len(output_rows),
        "status_counts": dict(status_counts),
        "label_counts": dict(label_counts),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-objects", type=Path, required=True)
    parser.add_argument("--output-objects", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--allow-scene-only", action="store_true")
    args = parser.parse_args()
    result = validate(
        args.input_objects,
        args.output_objects,
        args.report,
        allow_scene_only=args.allow_scene_only,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
