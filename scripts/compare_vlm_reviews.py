#!/usr/bin/env python3
"""Compare two object-level VLM review JSONL files by object_id."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["object_id"]): row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row}


def review_value(row: dict[str, Any]) -> tuple[str, float]:
    parsed = row.get("parsed") or {}
    return str(parsed.get("controlled_label") or ""), float(parsed.get("confidence") or 0.0)


def compare(old: dict[int, dict[str, Any]], new: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common = sorted(old.keys() & new.keys())
    changed = []
    transitions: Counter[str] = Counter()
    for object_id in common:
        old_label, old_confidence = review_value(old[object_id])
        new_label, new_confidence = review_value(new[object_id])
        if old_label != new_label:
            transitions[f"{old_label}->{new_label}"] += 1
            changed.append({
                "object_id": object_id,
                "old_label": old_label,
                "new_label": new_label,
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
                "confidence_delta": round(new_confidence - old_confidence, 6),
                "old": old[object_id],
                "new": new[object_id],
            })
    report = {
        "old_rows": len(old),
        "new_rows": len(new),
        "common_rows": len(common),
        "old_only": len(old.keys() - new.keys()),
        "new_only": len(new.keys() - old.keys()),
        "changed_labels": len(changed),
        "changed_label_ratio": len(changed) / max(len(common), 1),
        "transitions": dict(transitions.most_common()),
    }
    return report, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--changed", type=Path, required=True)
    args = parser.parse_args()
    report, changed = compare(read_rows(args.old), read_rows(args.new))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.changed.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changed), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
