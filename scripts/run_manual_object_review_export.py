#!/usr/bin/env python3
"""Normalize manual object decisions, apply them, and export a reviewed viewer artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import apply_manual_object_review_decisions as apply_review
from scripts import export_frame_target_objects_for_viewer as export_viewer
from scripts import normalize_manual_object_review_decisions as normalize_review
from scripts import qa_viewer_candidate


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_reviewed_viewer(args: argparse.Namespace, reviewed_objects: Path) -> dict[str, Any]:
    target_index_to_id = export_viewer.load_target_index_map(args.targets_jsonl)
    objects_by_id, target_to_object = export_viewer.load_object_maps(reviewed_objects)
    output_ply = args.output_dir / args.ply_name
    output_objects = args.output_dir / args.objects_name
    report = export_viewer.export_ply(
        args.target_ply,
        output_ply,
        target_index_to_id,
        target_to_object,
        objects_by_id,
        args.stride,
    )
    export_viewer.export_objects_jsonl(objects_by_id, output_objects, keep_targets=args.keep_target_list)
    report.update(
        {
            "targets_jsonl": str(args.targets_jsonl),
            "target_ply": str(args.target_ply),
            "objects_jsonl": str(reviewed_objects),
            "output_ply": str(output_ply),
            "output_objects_jsonl": str(output_objects),
            "object_records": len(objects_by_id),
            "target_records": len(target_index_to_id),
        }
    )
    write_json(args.output_dir / "frame_object_viewer_export_report.json", report)
    return report


def run_qa(args: argparse.Namespace) -> dict[str, Any]:
    qa_args = argparse.Namespace(
        ply=args.output_dir / args.ply_name,
        objects_jsonl=args.output_dir / args.objects_name,
        output_json=args.output_dir / "viewer_candidate_qa.json",
        output_md=args.output_dir / "viewer_candidate_qa.md",
        top_n=args.qa_top_n,
        ambiguous_report=None,
        consolidation_report=None,
    )
    report = qa_viewer_candidate.build_report(qa_args)
    write_json(qa_args.output_json, report)
    qa_viewer_candidate.write_markdown(qa_args.output_md, report, qa_args.top_n)
    return report


def copy_review_inputs(args: argparse.Namespace) -> None:
    if not args.copy_review_inputs:
        return
    for path in [args.decisions_csv, args.review_index_json]:
        if path.exists():
            shutil.copy2(path, args.output_dir / path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--review-index-json", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--target-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--ply-name", default="frame_object_points_stride10.ply")
    parser.add_argument("--objects-name", default="frame_objects_viewer.jsonl")
    parser.add_argument("--keep-target-list", action="store_true")
    parser.add_argument("--qa-top-n", type=int, default=20)
    parser.add_argument("--copy-review-inputs", action="store_true")
    parser.add_argument("--allow-normalize-errors", action="store_true")
    parser.add_argument("--allow-apply-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalized_jsonl = args.output_dir / "manual_object_review_decisions.normalized.jsonl"
    normalize_report_json = args.output_dir / "manual_object_review_decisions.report.json"
    normalized_rows, normalize_errors = normalize_review.normalize(args.decisions_csv, args.review_index_json)
    normalize_review.write_jsonl(normalized_jsonl, normalized_rows)
    normalize_report = {
        "schema": "manual-object-review-normalization-report/v1",
        "decisions_csv": str(args.decisions_csv),
        "review_index_json": str(args.review_index_json),
        "output_jsonl": str(normalized_jsonl),
        "accepted_count": len(normalized_rows),
        "error_count": len(normalize_errors),
        "errors": normalize_errors,
    }
    write_json(normalize_report_json, normalize_report)
    if normalize_errors and not args.allow_normalize_errors:
        print(json.dumps(normalize_report, ensure_ascii=False))
        return 2

    objects = apply_review.read_jsonl(args.objects_jsonl)
    decisions = apply_review.read_jsonl(normalized_jsonl)
    reviewed_objects, apply_report = apply_review.apply_decisions(objects, decisions)
    reviewed_objects_jsonl = args.output_dir / "frame_objects_viewer.manual_reviewed.jsonl"
    apply_review.write_jsonl(reviewed_objects_jsonl, reviewed_objects)
    apply_report.update(
        {
            "objects_jsonl": str(args.objects_jsonl),
            "decisions_jsonl": str(normalized_jsonl),
            "output_objects_jsonl": str(reviewed_objects_jsonl),
        }
    )
    write_json(args.output_dir / "manual_object_review_apply_report.json", apply_report)
    if apply_report["error_count"] and not args.allow_apply_errors:
        print(json.dumps(apply_report, ensure_ascii=False))
        return 3

    export_report = export_reviewed_viewer(args, reviewed_objects_jsonl)
    qa_report = run_qa(args)
    copy_review_inputs(args)

    summary = {
        "schema": "manual-object-reviewed-viewer-export/v1",
        "output_dir": str(args.output_dir),
        "normalized_decisions": str(normalized_jsonl),
        "reviewed_objects_jsonl": str(reviewed_objects_jsonl),
        "viewer_ply": str(args.output_dir / args.ply_name),
        "viewer_objects_jsonl": str(args.output_dir / args.objects_name),
        "normalize": {
            "accepted_count": normalize_report["accepted_count"],
            "error_count": normalize_report["error_count"],
        },
        "apply": {
            "applied_count": apply_report["applied_count"],
            "error_count": apply_report["error_count"],
            "decision_counts": apply_report["decision_counts"],
        },
        "export": {
            "output_vertices": export_report["output_vertices"],
            "object_records": export_report["object_records"],
            "missing_target_points": export_report["missing_target_points"],
        },
        "qa": {
            "status": qa_report["status"],
            "warnings": qa_report["warnings"],
            "errors": qa_report["errors"],
        },
    }
    write_json(args.output_dir / "manual_object_review_export_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if qa_report["status"] == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
