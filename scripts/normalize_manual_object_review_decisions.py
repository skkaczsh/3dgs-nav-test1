#!/usr/bin/env python3
"""Validate manual object review decisions from the semantic object review page."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VALID_DECISIONS = {
    "keep",
    "relabel",
    "demote_unknown",
    "split_review",
    "reject_artifact",
}

VALID_LABELS = {
    "floor",
    "ground",
    "wall",
    "grass",
    "car",
    "railing",
    "tree_or_shrub",
    "equipment",
    "hvac_outdoor_unit",
    "traffic_cone",
    "pipe_or_pole",
    "door_or_window",
    "sign_or_box",
    "curb_or_low_barrier",
    "building_part",
    "unknown",
}


def read_review_index(path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    for item in report.get("objects", []):
        oid = str(item.get("object_id"))
        if oid and oid != "None":
            rows[oid] = item
    return rows


def parse_confidence(value: str) -> tuple[float | None, str]:
    if not value.strip():
        return None, ""
    try:
        confidence = float(value)
    except ValueError:
        return None, "invalid_confidence"
    if confidence < 0.0 or confidence > 1.0:
        return None, "invalid_confidence"
    return confidence, ""


def normalize_row(row: dict[str, str], review_objects: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    oid = str(row.get("object_id") or "").strip()
    if oid not in review_objects:
        return None, {"object_id": oid, "error": "unknown_object_id", "row": row}

    decision = str(row.get("decision") or "").strip().lower()
    if decision in {"", "pending"}:
        return None, {"object_id": oid, "error": "pending", "row": row}
    if decision not in VALID_DECISIONS:
        return None, {"object_id": oid, "error": "invalid_decision", "row": row}

    current_label = str(row.get("current_label") or review_objects[oid].get("label") or "unknown").strip()
    new_label = str(row.get("new_label") or "").strip()
    if decision == "keep":
        final_label = current_label
    elif decision == "demote_unknown":
        final_label = "unknown"
    elif decision == "relabel":
        final_label = new_label
        if final_label not in VALID_LABELS:
            return None, {"object_id": oid, "error": "invalid_new_label", "row": row}
    else:
        final_label = new_label or current_label
        if final_label and final_label not in VALID_LABELS:
            return None, {"object_id": oid, "error": "invalid_new_label", "row": row}

    confidence, conf_error = parse_confidence(str(row.get("confidence") or ""))
    if conf_error:
        return None, {"object_id": oid, "error": conf_error, "row": row}

    return {
        "schema": "manual-object-review-decision/v1",
        "object_id": oid,
        "source_object_id": row.get("source_object_id") or review_objects[oid].get("source_object_id"),
        "current_label": current_label,
        "decision": decision,
        "final_label": final_label,
        "confidence": confidence,
        "reviewer": row.get("reviewer") or "",
        "notes": row.get("notes") or "",
    }, None


def normalize(csv_path: Path, review_index: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    review_objects = read_review_index(review_index)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            normalized, error = normalize_row(row, review_objects)
            if normalized:
                rows.append(normalized)
            if error:
                errors.append(error)
    return rows, errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--review-index-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, errors = normalize(args.decisions_csv, args.review_index_json)
    write_jsonl(args.output_jsonl, rows)
    report = {
        "schema": "manual-object-review-normalization-report/v1",
        "decisions_csv": str(args.decisions_csv),
        "review_index_json": str(args.review_index_json),
        "output_jsonl": str(args.output_jsonl),
        "accepted_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors or args.allow_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
