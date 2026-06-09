#!/usr/bin/env python3
"""QA reviewed object merge outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_ids(row: dict) -> list[str]:
    ids = row.get("source_long_object_ids")
    if isinstance(ids, list):
        return [str(v) for v in ids]
    return [str(row["long_object_id"])]


def qa(input_objects: list[dict], output_objects: list[dict], decisions: list[dict]) -> dict:
    input_ids = [str(row["long_object_id"]) for row in input_objects]
    output_sources = [src for row in output_objects for src in source_ids(row)]
    input_point_count = sum(int(row.get("point_count", 0)) for row in input_objects)
    output_point_count = sum(int(row.get("point_count", 0)) for row in output_objects)
    accepted = [row for row in decisions if row.get("accepted") is True]
    merged_groups = [row for row in output_objects if len(source_ids(row)) > 1]
    missing = sorted(set(input_ids) - set(output_sources))
    extra = sorted(set(output_sources) - set(input_ids))
    duplicate_sources = sorted({src for src in output_sources if output_sources.count(src) > 1})
    expected_output_count = len(input_objects) - len(accepted)
    checks = {
        "point_count_preserved": input_point_count == output_point_count,
        "all_input_objects_covered": not missing,
        "no_extra_source_objects": not extra,
        "no_duplicate_source_objects": not duplicate_sources,
        "object_count_matches_accepted_merges": len(output_objects) == expected_output_count,
        "merged_group_count_matches": len(merged_groups) == len(accepted),
    }
    return {
        "input_object_count": len(input_objects),
        "output_object_count": len(output_objects),
        "input_point_count": input_point_count,
        "output_point_count": output_point_count,
        "accepted_merge_count": len(accepted),
        "merged_group_count": len(merged_groups),
        "expected_output_object_count": expected_output_count,
        "missing_source_objects": missing,
        "extra_source_objects": extra,
        "duplicate_source_objects": duplicate_sources,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-objects", type=Path, required=True)
    parser.add_argument("--output-objects", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    report = qa(load_jsonl(args.input_objects), load_jsonl(args.output_objects), load_jsonl(args.decisions))
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
