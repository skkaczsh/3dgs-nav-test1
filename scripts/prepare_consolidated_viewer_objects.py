#!/usr/bin/env python3
"""Prepare consolidated object JSONL for the semantic PLY viewer.

`remap_ply_object_ids.py` rewrites PLY object ids to compact integers and emits
a sidecar mapping of consolidated object id -> integer id. This script applies
that same mapping to the consolidated objects JSONL so the viewer can join PLY
points with object metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_sidecar(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = data.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"Sidecar missing mapping object: {path}")
    return {str(key): int(value) for key, value in mapping.items()}


def convert_objects(rows: list[dict[str, Any]], mapping: dict[str, int]) -> tuple[list[dict[str, Any]], list[str]]:
    out = []
    missing = []
    for row in rows:
        source_id = str(row.get("object_id") or "")
        numeric_id = mapping.get(source_id)
        if numeric_id is None:
            missing.append(source_id)
            continue
        obj = dict(row)
        obj["viewer_object_id"] = int(numeric_id)
        obj["original_object_id"] = source_id
        obj["object_id"] = int(numeric_id)
        out.append(obj)
    return out, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--remap-sidecar", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    mapping = load_sidecar(args.remap_sidecar)
    rows = read_jsonl(args.objects_jsonl)
    converted, missing = convert_objects(rows, mapping)
    write_jsonl(args.output_jsonl, converted)
    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "remap_sidecar": str(args.remap_sidecar),
        "output_jsonl": str(args.output_jsonl),
        "input_objects": len(rows),
        "output_objects": len(converted),
        "missing_objects": len(missing),
        "missing_examples": missing[:50],
        "mapping_objects": len(mapping),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
