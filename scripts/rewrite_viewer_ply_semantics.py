#!/usr/bin/env python3
"""Rewrite semantic/RGB fields in an existing viewer PLY from object JSONL labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_frame_target_objects_for_viewer import LABEL_TO_SEMANTIC, SEMANTIC_COLORS
from scripts.current_mainline_contract import reject_forbidden_production_input


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def object_key(row: dict[str, Any]) -> int | None:
    for key in ("viewer_object_id", "object_id"):
        value = row.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def load_object_metadata_map(objects_jsonl: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(objects_jsonl):
        oid = object_key(row)
        if oid is not None:
            out[oid] = row
    return out


def load_object_label_map(objects_jsonl: Path) -> dict[int, str]:
    return {oid: str(row.get("semantic_label") or "unknown") for oid, row in load_object_metadata_map(objects_jsonl).items()}


def source_support_kind(row: dict[str, Any]) -> str:
    label = str(row.get("semantic_label") or "unknown")
    source_scores = row.get("semantic_evidence_source_scores")
    if not isinstance(source_scores, dict):
        return "missing_source_scores"
    active_sources: list[str] = []
    for source in ("sam", "teacher", "scene"):
        scores = source_scores.get(source)
        if not isinstance(scores, dict):
            continue
        try:
            score = float(scores.get(label, 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > 0:
            active_sources.append(source)
    return "+".join(active_sources) if active_sources else "no_label_source_support"


def read_header(path: Path) -> tuple[list[str], list[str], int]:
    header: list[str] = []
    props: list[str] = []
    in_vertex = False
    header_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            header.append(line)
            if stripped == "end_header":
                break
    return header, props, header_lines


def rewrite_ply(
    source_ply: Path,
    objects_jsonl: Path,
    output_ply: Path,
    *,
    allow_qa_preview_source: bool = False,
) -> dict[str, Any]:
    reject_forbidden_production_input(source_ply, allow_qa_preview=allow_qa_preview_source)
    reject_forbidden_production_input(objects_jsonl)
    reject_forbidden_production_input(output_ply)
    objects = load_object_metadata_map(objects_jsonl)
    header, props, header_lines = read_header(source_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"red", "green", "blue", "object", "semantic"}
    missing = required - set(idx)
    if missing:
        raise ValueError(f"PLY missing required fields: {sorted(missing)}")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    label_counts = Counter()
    object_counts = Counter()
    object_label_counts = Counter()
    point_source_support_counts = Counter()
    object_source_support_counts = Counter()
    fusion_status_counts = Counter()
    conflict_flag_counts = Counter()
    unknown_objects = Counter()
    rows = 0
    with source_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in header:
            dst.write(line)
        for _ in range(header_lines):
            next(src)
        for line in src:
            if not line.strip():
                continue
            parts = line.strip().split()
            oid = int(float(parts[idx["object"]]))
            obj = objects.get(oid)
            label = str(obj.get("semantic_label") or "unknown") if obj else "unknown"
            if obj is None:
                unknown_objects[oid] += 1
                support_kind = "missing_object"
                fusion_status = "missing_object"
                conflict_flags: list[str] = []
            else:
                support_kind = source_support_kind(obj)
                fusion_status = str(obj.get("semantic_fusion_status") or "missing")
                conflict_flags = [str(flag) for flag in obj.get("conflict_flags") or []]
            semantic = LABEL_TO_SEMANTIC.get(label, 0)
            color = SEMANTIC_COLORS.get(semantic, SEMANTIC_COLORS[0])
            parts[idx["red"]] = str(color[0])
            parts[idx["green"]] = str(color[1])
            parts[idx["blue"]] = str(color[2])
            parts[idx["semantic"]] = str(semantic)
            dst.write(" ".join(parts) + "\n")
            label_counts[label] += 1
            object_counts[oid] += 1
            point_source_support_counts[support_kind] += 1
            fusion_status_counts[fusion_status] += 1
            for flag in conflict_flags:
                conflict_flag_counts[flag] += 1
            rows += 1
    for oid in object_counts:
        obj = objects.get(oid)
        if obj is None:
            object_source_support_counts["missing_object"] += 1
            object_label_counts["unknown"] += 1
            continue
        object_source_support_counts[source_support_kind(obj)] += 1
        object_label_counts[str(obj.get("semantic_label") or "unknown")] += 1
    return {
        "source_ply": str(source_ply),
        "objects_jsonl": str(objects_jsonl),
        "output_ply": str(output_ply),
        "rows": rows,
        "object_count": len(object_counts),
        "label_counts": dict(label_counts),
        "object_label_counts": dict(object_label_counts),
        "point_source_support_counts": dict(point_source_support_counts),
        "object_source_support_counts": dict(object_source_support_counts),
        "fusion_status_counts": dict(fusion_status_counts),
        "conflict_flag_counts": dict(conflict_flag_counts),
        "unknown_object_count": len(unknown_objects),
        "unknown_object_examples": [oid for oid, _ in unknown_objects.most_common(20)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--allow-qa-preview-source",
        action="store_true",
        help="Allow a stride-sampled viewer PLY as QA source. Output remains QA-only and cannot be a production input.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = rewrite_ply(
        args.source_ply,
        args.objects_jsonl,
        args.output_ply,
        allow_qa_preview_source=args.allow_qa_preview_source,
    )
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["unknown_object_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
