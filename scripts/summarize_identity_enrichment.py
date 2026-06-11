#!/usr/bin/env python3
"""Summarize coarse-label plus identity-description coverage.

The semantic route keeps labels constrained while using description,
identity_hint, and attributes for object identity. This report makes that
coverage auditable across 2D label records and fused objects.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def label_record_paths(semantic_eval_dir: Path, combo: str) -> list[Path]:
    return sorted((semantic_eval_dir / "images").glob(f"*/{combo}/label_records.json"))


def iter_label_records(semantic_eval_dir: Path, combo: str):
    for path in label_record_paths(semantic_eval_dir, combo):
        try:
            raw = read_json(path)
        except Exception as exc:
            yield {"_error": repr(exc), "_path": str(path)}
            continue
        values = raw.values() if isinstance(raw, dict) else raw
        for value in values:
            if isinstance(value, dict):
                yield {**value, "_path": str(path)}


def has_identity(row: dict) -> bool:
    return bool(row.get("description") or row.get("identity_hint") or row.get("attributes"))


def summarize_records(rows: list[dict]) -> dict:
    labels = Counter()
    enriched = Counter()
    descriptions: dict[str, Counter] = defaultdict(Counter)
    errors = []
    for row in rows:
        if "_error" in row:
            errors.append({"path": row.get("_path"), "error": row["_error"]})
            continue
        label = str(row.get("label") or "unknown")
        labels[label] += 1
        if has_identity(row):
            enriched[label] += 1
        description = str(row.get("description") or row.get("identity_hint") or "").strip()
        if description:
            descriptions[label][description] += 1
    return {
        "record_count": int(sum(labels.values())),
        "enriched_count": int(sum(enriched.values())),
        "enriched_ratio": float(sum(enriched.values()) / max(sum(labels.values()), 1)),
        "label_counts": dict(labels),
        "enriched_by_label": dict(enriched),
        "top_descriptions_by_label": {
            label: [{"description": desc, "count": count} for desc, count in counter.most_common(10)]
            for label, counter in descriptions.items()
        },
        "errors": errors[:50],
        "error_count": len(errors),
    }


def iter_objects(path: Path):
    if not path or not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                yield {"_error": repr(exc), "_line": line_no}
                continue
            if isinstance(row, dict):
                yield row


def summarize_objects(rows: list[dict]) -> dict:
    labels = Counter()
    enriched = Counter()
    statuses = Counter()
    desc_ratios = []
    errors = []
    for row in rows:
        if "_error" in row:
            errors.append({"line": row.get("_line"), "error": row["_error"]})
            continue
        label = str(row.get("semantic_label") or row.get("dominant_label") or "unknown")
        labels[label] += 1
        statuses[str(row.get("status") or "")] += 1
        if has_identity(row) or row.get("description_votes") or row.get("dominant_attributes"):
            enriched[label] += 1
        if row.get("description_vote_ratio") is not None:
            try:
                desc_ratios.append(float(row["description_vote_ratio"]))
            except (TypeError, ValueError):
                pass
    desc_ratios.sort()
    median = desc_ratios[len(desc_ratios) // 2] if desc_ratios else None
    return {
        "object_count": int(sum(labels.values())),
        "enriched_count": int(sum(enriched.values())),
        "enriched_ratio": float(sum(enriched.values()) / max(sum(labels.values()), 1)),
        "label_counts": dict(labels),
        "status_counts": dict(statuses),
        "enriched_by_label": dict(enriched),
        "description_vote_ratio_median": median,
        "errors": errors[:50],
        "error_count": len(errors),
    }


def write_description_csv(path: Path, record_summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "description", "count"])
        writer.writeheader()
        for label, rows in record_summary.get("top_descriptions_by_label", {}).items():
            for row in rows:
                writer.writerow({"label": label, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--objects-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--description-csv", type=Path, default=None)
    args = parser.parse_args()

    record_rows = list(iter_label_records(args.semantic_eval_dir, args.combo))
    object_rows = list(iter_objects(args.objects_jsonl)) if args.objects_jsonl else []
    record_summary = summarize_records(record_rows)
    object_summary = summarize_objects(object_rows)
    report = {
        "semantic_eval_dir": str(args.semantic_eval_dir),
        "combo": args.combo,
        "label_record_files": len(label_record_paths(args.semantic_eval_dir, args.combo)),
        "records": record_summary,
        "objects_jsonl": str(args.objects_jsonl) if args.objects_jsonl else None,
        "objects": object_summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.description_csv:
        write_description_csv(args.description_csv, record_summary)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
