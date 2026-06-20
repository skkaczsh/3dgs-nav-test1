#!/usr/bin/env python3
"""Apply normalized manual object review decisions to a viewer object JSONL.

This stage updates object metadata only. Re-export the viewer PLY from the
updated objects JSONL to change semantic colors in point-level PLY output.
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

from scripts.export_frame_target_objects_for_viewer import LABEL_TO_SEMANTIC


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def object_review_key(row: dict[str, Any]) -> str:
    value = row.get("viewer_object_id")
    if value is None:
        value = row.get("object_id")
    return str(value)


def decisions_by_object(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    out: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row.get("object_id") or "")
        if not oid:
            errors.append({"error": "missing_object_id", "decision": row})
            continue
        if oid in out:
            errors.append({"object_id": oid, "error": "duplicate_decision", "decision": row})
            continue
        out[oid] = row
    return out, errors


def apply_decision(obj: dict[str, Any], decision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(obj)
    old_label = str(out.get("semantic_label") or "unknown")
    action = str(decision.get("decision") or "")
    final_label = str(decision.get("final_label") or old_label)
    status_before = str(out.get("status") or "")

    if action in {"relabel", "demote_unknown"} and final_label != old_label:
        out["semantic_label_original"] = out.get("semantic_label_original") or old_label
        out["semantic_label"] = final_label
        out["semantic_id"] = LABEL_TO_SEMANTIC.get(final_label, 0)
        out["manual_review_relabel_from"] = old_label
        out["manual_review_relabel_to"] = final_label
    elif action == "reject_artifact":
        out["semantic_label_original"] = out.get("semantic_label_original") or old_label
        out["semantic_label"] = "unknown"
        out["semantic_id"] = LABEL_TO_SEMANTIC["unknown"]
        out["status"] = "manual_rejected_artifact"
    elif action == "split_review":
        out["status"] = "manual_split_review"

    out["manual_review_status"] = action
    out["manual_review_confidence"] = decision.get("confidence")
    out["manual_review_reviewer"] = decision.get("reviewer") or ""
    out["manual_review_notes"] = decision.get("notes") or ""

    return out, {
        "object_id": object_review_key(obj),
        "source_object_id": obj.get("object_id"),
        "decision": action,
        "old_label": old_label,
        "new_label": str(out.get("semantic_label") or "unknown"),
        "status_before": status_before,
        "status_after": out.get("status") or "",
    }


def apply_decisions(objects: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decision_map, errors = decisions_by_object(decisions)
    seen: set[str] = set()
    applied: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    for obj in objects:
        key = object_review_key(obj)
        decision = decision_map.get(key)
        if not decision:
            output.append(dict(obj))
            continue
        updated, event = apply_decision(obj, decision)
        output.append(updated)
        applied.append(event)
        seen.add(key)

    for oid in sorted(set(decision_map) - seen):
        errors.append({"object_id": oid, "error": "decision_object_not_found"})

    return output, {
        "schema": "manual-object-review-apply-report/v1",
        "input_object_count": len(objects),
        "output_object_count": len(output),
        "decision_count": len(decisions),
        "applied_count": len(applied),
        "error_count": len(errors),
        "errors": errors,
        "applied": applied,
        "label_counts_before": dict(Counter(str(row.get("semantic_label") or "unknown") for row in objects)),
        "label_counts_after": dict(Counter(str(row.get("semantic_label") or "unknown") for row in output)),
        "decision_counts": dict(Counter(row["decision"] for row in applied)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--decisions-jsonl", type=Path, required=True)
    parser.add_argument("--output-objects-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = read_jsonl(args.objects_jsonl)
    decisions = read_jsonl(args.decisions_jsonl)
    output, report = apply_decisions(objects, decisions)
    write_jsonl(args.output_objects_jsonl, output)
    report.update(
        {
            "objects_jsonl": str(args.objects_jsonl),
            "decisions_jsonl": str(args.decisions_jsonl),
            "output_objects_jsonl": str(args.output_objects_jsonl),
        }
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["error_count"] == 0 or args.allow_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
