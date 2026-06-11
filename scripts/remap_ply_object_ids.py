#!/usr/bin/env python3
"""Remap the object field in an ASCII PLY using source->consolidated JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def object_number(object_id: str) -> int:
    match = re.search(r"(\d+)$", object_id or "")
    return int(match.group(1)) if match else 0


def load_mapping(path: Path) -> tuple[dict[int, int], dict[str, int]]:
    consolidated_to_int: dict[str, int] = {}
    source_to_consolidated: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source = object_number(str(row.get("source_object_id", "")))
            consolidated = str(row.get("consolidated_object_id", ""))
            if consolidated not in consolidated_to_int:
                consolidated_to_int[consolidated] = len(consolidated_to_int) + 1
            if source:
                source_to_consolidated[source] = consolidated_to_int[consolidated]
    return source_to_consolidated, consolidated_to_int


def read_header(path: Path) -> tuple[list[str], list[str], int]:
    header = []
    props = []
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            stripped = line.strip()
            if stripped.startswith("format ") and stripped.split()[1] != "ascii":
                raise ValueError(f"Only ASCII PLY is supported: {path}")
            if stripped.startswith("element vertex"):
                in_vertex = True
            elif stripped.startswith("element "):
                in_vertex = False
            elif in_vertex and stripped.startswith("property "):
                props.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
    if "object" not in props:
        raise ValueError(f"PLY has no object field: {path}")
    return header, props, len(header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--unmapped-object", type=int, default=0)
    args = parser.parse_args()

    source_to_consolidated, consolidated_to_int = load_mapping(args.mapping)
    header, props, header_lines = read_header(args.input)
    obj_idx = props.index("object")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    remapped = 0
    unmapped = 0
    with args.input.open("r", encoding="utf-8", errors="replace") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in header:
            dst.write(line)
        for _ in range(header_lines):
            next(src)
        for line in src:
            parts = line.split()
            if len(parts) <= obj_idx:
                continue
            source_obj = int(float(parts[obj_idx]))
            new_obj = source_to_consolidated.get(source_obj)
            if new_obj is None:
                new_obj = args.unmapped_object
                unmapped += 1
            else:
                remapped += 1
            parts[obj_idx] = str(new_obj)
            dst.write(" ".join(parts) + "\n")
    sidecar = args.output.with_suffix(args.output.suffix + ".mapping.json")
    sidecar.write_text(
        json.dumps(
            {
                "consolidated_object_count": len(consolidated_to_int),
                "remapped_vertices": remapped,
                "unmapped_vertices": unmapped,
                "mapping": consolidated_to_int,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"consolidated_object_count={len(consolidated_to_int)}")
    print(f"remapped_vertices={remapped}")
    print(f"unmapped_vertices={unmapped}")
    print(f"output={args.output}")
    print(f"sidecar={sidecar}")


if __name__ == "__main__":
    main()
