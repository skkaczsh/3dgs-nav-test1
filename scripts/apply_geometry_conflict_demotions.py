#!/usr/bin/env python3
"""Demote geometry-conflicting objects and rewrite the viewer PLY.

This pass is intentionally conservative: it only changes objects that a prior
QA report has already marked with a specific suggested_action. The source label
is preserved in metadata so manual review can still recover or relabel it.
"""

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

from scripts import qa_viewer_candidate
from scripts import rewrite_viewer_ply_semantics
from scripts.apply_manual_object_review_decisions import read_jsonl, write_jsonl
from scripts.export_frame_target_objects_for_viewer import LABEL_TO_SEMANTIC


def selected_conflicts(conflicts_jsonl: Path, source_label: str, action: str) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(conflicts_jsonl):
        if str(row.get("semantic_label") or "unknown") != source_label:
            continue
        if str(row.get("suggested_action") or "") != action:
            continue
        selected[int(row["object_id"])] = row
    return selected


def apply_demotions(
    objects_jsonl: Path,
    conflicts_jsonl: Path,
    source_label: str,
    action: str,
    target_label: str,
    status: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_label not in LABEL_TO_SEMANTIC:
        raise ValueError(f"Unknown target label: {target_label}")
    conflicts = selected_conflicts(conflicts_jsonl, source_label, action)
    output: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for obj in read_jsonl(objects_jsonl):
        oid = int(obj.get("viewer_object_id", 0) or 0)
        out = dict(obj)
        conflict = conflicts.get(oid)
        if conflict and str(out.get("semantic_label") or "unknown") == source_label:
            old_label = str(out.get("semantic_label") or "unknown")
            out["semantic_label_original"] = out.get("semantic_label_original") or old_label
            out["semantic_label"] = target_label
            out["semantic_id"] = LABEL_TO_SEMANTIC[target_label]
            out["status"] = status
            out["geometry_demotion_status"] = "applied"
            out["geometry_demotion_action"] = action
            out["geometry_demotion_source_label"] = source_label
            out["geometry_demotion_target_label"] = target_label
            out["geometry_demotion_reasons"] = conflict.get("reasons") or []
            out["geometry_demotion_metrics"] = conflict.get("metrics") or {}
            out["geometry_demotion_source"] = str(conflicts_jsonl)
            applied.append(
                {
                    "object_id": oid,
                    "source_object_id": out.get("object_id"),
                    "old_label": old_label,
                    "new_label": target_label,
                    "reasons": out["geometry_demotion_reasons"],
                    "point_count": out.get("point_count"),
                }
            )
        output.append(out)

    found = {row["object_id"] for row in applied}
    missing = sorted(set(conflicts) - found)
    return output, {
        "schema": "geometry-conflict-demotion/v1",
        "objects_jsonl": str(objects_jsonl),
        "conflicts_jsonl": str(conflicts_jsonl),
        "source_label": source_label,
        "target_label": target_label,
        "action": action,
        "candidate_count": len(conflicts),
        "applied_count": len(applied),
        "missing_object_ids": missing,
        "label_counts_after": dict(Counter(str(row.get("semantic_label") or "unknown") for row in output)),
        "applied": applied,
    }


def run_qa(output_dir: Path, ply_name: str, objects_name: str, top_n: int) -> dict[str, Any]:
    qa_args = argparse.Namespace(
        ply=output_dir / ply_name,
        objects_jsonl=output_dir / objects_name,
        output_json=output_dir / "viewer_candidate_qa.json",
        output_md=output_dir / "viewer_candidate_qa.md",
        top_n=top_n,
        ambiguous_report=None,
        consolidation_report=None,
    )
    report = qa_viewer_candidate.build_report(qa_args)
    qa_args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    qa_viewer_candidate.write_markdown(qa_args.output_md, report, top_n)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--conflicts-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ply-name", default="frame_object_points_stride10.ply")
    parser.add_argument("--objects-name", default="frame_objects_viewer.jsonl")
    parser.add_argument("--source-label", default="car")
    parser.add_argument("--target-label", default="unknown")
    parser.add_argument("--action", default="demote_or_visual_review")
    parser.add_argument("--status", default="geometry_demoted_visual_review")
    parser.add_argument("--qa-top-n", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects, demotion_report = apply_demotions(
        args.objects_jsonl,
        args.conflicts_jsonl,
        args.source_label,
        args.action,
        args.target_label,
        args.status,
    )
    output_objects = args.output_dir / args.objects_name
    output_ply = args.output_dir / args.ply_name
    write_jsonl(output_objects, objects)
    rewrite_report = rewrite_viewer_ply_semantics.rewrite_ply(args.source_ply, output_objects, output_ply)
    qa_report = run_qa(args.output_dir, args.ply_name, args.objects_name, args.qa_top_n)
    report = {
        **demotion_report,
        "output_objects_jsonl": str(output_objects),
        "output_ply": str(output_ply),
        "rewrite": rewrite_report,
        "qa": {"status": qa_report["status"], "warnings": qa_report["warnings"], "errors": qa_report["errors"]},
    }
    (args.output_dir / "geometry_conflict_demotions_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if qa_report["status"] == "ok" and not demotion_report["missing_object_ids"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
