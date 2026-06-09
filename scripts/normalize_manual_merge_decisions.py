#!/usr/bin/env python3
"""Normalize manual cross-candidate merge decisions into review JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VALID_DECISIONS = {"merge", "keep_split", "uncertain", "pending"}


def load_review_items(path: Path) -> dict[str, dict]:
    items = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            items[str(row["review_id"])] = row
    return items


def parse_confidence(value: str, decision: str) -> float:
    value = str(value or "").strip()
    if not value:
        return 0.0 if decision in {"pending", "uncertain"} else 1.0
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid confidence {value!r}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"confidence out of range: {parsed}")
    return parsed


def normalize_row(row: dict, review_items: dict[str, dict]) -> tuple[dict | None, dict | None]:
    review_id = str(row.get("review_id", "")).strip()
    item = review_items.get(review_id)
    if item is None:
        return None, {"review_id": review_id, "error": "unknown_review_id"}
    proposal = item["proposal"]
    object_a = str(row.get("object_a", "")).strip()
    object_b = str(row.get("object_b", "")).strip()
    if object_a != proposal["object_a"] or object_b != proposal["object_b"]:
        return None, {"review_id": review_id, "error": "object_pair_mismatch"}
    decision = str(row.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        return None, {"review_id": review_id, "error": f"invalid_decision:{decision}"}
    try:
        confidence = parse_confidence(str(row.get("confidence", "")), decision)
    except ValueError as exc:
        return None, {"review_id": review_id, "error": str(exc)}
    if decision == "pending":
        return None, {"review_id": review_id, "error": "pending"}
    notes = str(row.get("notes", "")).strip()
    reviewer = str(row.get("reviewer", "")).strip()
    return (
        {
            "review_id": review_id,
            "object_a": object_a,
            "object_b": object_b,
            "candidate_a": proposal.get("candidate_a", ""),
            "candidate_b": proposal.get("candidate_b", ""),
            "vlm": {
                "decision": decision,
                "confidence": confidence,
                "physical_relation": "manual_review",
                "reason": notes,
                "evidence": ["manual cross-candidate review"],
                "risk": "",
            },
            "reviewer": reviewer,
            "status": "manual",
        },
        None,
    )


def normalize(csv_path: Path, review_items_path: Path) -> tuple[list[dict], list[dict]]:
    review_items = load_review_items(review_items_path)
    rows = []
    errors = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            normalized, error = normalize_row(row, review_items)
            if normalized is not None:
                rows.append(normalized)
            if error is not None:
                errors.append(error)
    return rows, errors


def write_outputs(rows: list[dict], errors: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "manual_merge_reviews.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    errors_path = output_dir / "manual_merge_review_errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as f:
        for row in errors:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "review_jsonl": str(jsonl_path),
        "error_jsonl": str(errors_path),
        "accepted_review_count": len(rows),
        "error_count": len(errors),
        "decision_counts": {},
    }
    for row in rows:
        decision = row["vlm"]["decision"]
        report["decision_counts"][decision] = report["decision_counts"].get(decision, 0) + 1
    (output_dir / "manual_merge_review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-csv", type=Path, required=True)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, errors = normalize(args.manual_csv, args.review_jsonl)
    write_outputs(rows, errors, args.output_dir)
    print(json.dumps({"reviews": len(rows), "errors": len(errors), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
